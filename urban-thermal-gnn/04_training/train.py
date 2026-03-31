"""
04_training/train.py
════════════════════════════════════════════════════════════════
PIN-ST-GNN Training main program

Run:
  cd 04_training
  python train.py
  python train.py --config ../../00_config/urbangraph_params.yaml
"""
from __future__ import annotations
import sys, json, math, argparse, warnings
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Try to import PyTorch Lightning (optional)
try:
    import lightning as L
    from lightning.pytorch.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateMonitor
    )
    _LIGHTNING = True
except ImportError:
    _LIGHTNING = False
    warnings.warn("lightning not installed, using manual training loop.")

from dataset import UTCIGraphDataset
from urbangraph import UrbanGraph, build_model
from shared import EPWData, solar_position
from live_loss_plot import LiveLossPlotter


# ════════════════════════════════════════════════════════════════
# 1. Global Environment & Time Feature Construction
# ════════════════════════════════════════════════════════════════
def build_env_time_seq(epw: EPWData, sim_hours: list,
                        month: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build global climate sequence (env_seq) and time feature sequence (time_seq).
    env_seq  : (T, 7) [Ta, RH, WS, WDsin, WDcos, GHI, SolAlt]
    time_seq : (T, 2) [sin_hour, cos_hour]
    """
    typical  = epw.get_typical_day(month=month, stat="hottest")
    clim_map = {h.hour: h for h in typical}
    T        = len(sim_hours)

    env  = np.zeros((T, 7), dtype=np.float32)
    time = np.zeros((T, 2), dtype=np.float32)

    for i, hr in enumerate(sim_hours):
        clim = clim_map.get(hr)
        if clim is None:
            continue
        sol_alt, _ = solar_position(
            epw.latitude, epw.longitude, epw.timezone,
            clim.month, clim.day if clim.day else 15, hr
        )
        wd_r = math.radians(clim.wind_dir)
        env[i]  = [clim.ta/35.0, clim.rh/100.0,
                    clim.wind_speed/8.0,
                    math.sin(wd_r), math.cos(wd_r),
                    clim.ghi/1000.0, max(0.0, sol_alt)/90.0]
        time[i] = [math.sin(2*math.pi*hr/24),
                    math.cos(2*math.pi*hr/24)]

    return torch.from_numpy(env), torch.from_numpy(time)


# ════════════════════════════════════════════════════════════════
# 2. Batch Collate Function
# ════════════════════════════════════════════════════════════════
def collate_single(batch):
    """Each HeteroData processed independently (no batching, graphs differ in size)."""
    return batch[0]


# ════════════════════════════════════════════════════════════════
# 3. Manual Training Loop
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self,
                 model:      UrbanGraph,
                 train_ds:   UTCIGraphDataset,
                 val_ds:     UTCIGraphDataset,
                 epw:        EPWData,
                 cfg:        dict,
                 device:     str = "cpu",
                 out_dir:    str = "checkpoints",
                 live_plot:  bool = True):

        self.model    = model.to(device)
        self.train_ds = train_ds
        self.val_ds   = val_ds
        self.device   = device
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        tr = cfg.get("training", {})
        self.lr           = float(tr.get("lr",           1e-3))
        self.wd           = float(tr.get("weight_decay", 1e-4))
        self.max_epochs   = int(  tr.get("max_epochs",   200))
        self.patience     = int(  tr.get("early_stopping_patience", 20))
        self.batch_size   = int(  tr.get("batch_size",   1))
        self.lambda_sense = float(tr.get("lambda_sensing", 0.5))

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        # Global environment sequence (fixed, not scenario-specific)
        sim_hours = train_ds.sim_hours
        self.env_seq, self.time_seq = build_env_time_seq(
            epw, sim_hours, month=7)
        self.env_seq  = self.env_seq.to(device)
        self.time_seq = self.time_seq.to(device)

        # Training history
        self.history = {"train_loss": [], "val_loss": [],
                         "val_r2": [], "lr": []}
        self.best_val_loss = float("inf")
        self.no_improve    = 0

        # Real-time loss curve
        self.plotter: Optional[LiveLossPlotter] = None
        if live_plot:
            self.plotter = LiveLossPlotter(
                max_epochs=self.max_epochs,
                save_dir=str(self.out_dir),
            )

    def _forward_one(self, data) -> dict:
        """Perform forward propagation on single HeteroData and compute loss."""
        obj_feat  = data["object"].x.to(self.device)       # (N_obj, 7)
        air_feat  = data["air"].x.to(self.device)           # (N_air, T, dim_air)
        target    = data["air"].y.to(self.device)           # (N_air, T)
        air_pos   = data["air"].pos.to(self.device)

        # Static edges
        static_edges = {}
        for rel in ["semantic", "contiguity"]:
            key = ("object", rel, "object") if rel == "semantic" else ("air", rel, "air")
            if hasattr(data[key], "edge_index"):
                static_edges[rel] = data[key].edge_index.to(self.device)

        # Dynamic edges
        dynamic_edges = getattr(data, "dynamic_edges", [{}] * air_feat.shape[1])

        pred = self.model(
            obj_feat      = obj_feat,
            air_feat      = air_feat,
            dynamic_edges = dynamic_edges,
            static_edges  = static_edges,
            env_seq       = self.env_seq,
            time_seq      = self.time_seq,
        )

        # SVF / shadow / bldg_height (read from npz fields)
        svf       = air_feat[:, :, 4][:, 0]   # (N_air,) static
        in_shadow = air_feat[:, :, 5]          # (N_air, T)
        bh        = (air_feat[:, :, 0].mean(dim=1) * 0.0
                      + 0.3)                   # simplified: fixed 0.3

        # Solar altitude angle sequence (T,)
        T      = air_feat.shape[1]
        sol_alt = self.env_seq[:T, 6] * 90.0  # denormalize

        losses = self.model.compute_loss(
            pred         = pred,
            target       = target,
            svf          = svf,
            in_shadow    = in_shadow,
            sol_alt_seq  = sol_alt,
            bldg_height  = bh,
            quality_w    = None,
            lambda_sense = self.lambda_sense,
        )
        return losses, pred, target

    def _epoch(self, dataset: UTCIGraphDataset, train: bool) -> dict:
        self.model.train(train)
        total = {"loss_data": 0, "loss_physics": 0,
                  "loss_total": 0, "n": 0}

        all_pred, all_tgt = [], []
        n = len(dataset)

        with torch.set_grad_enabled(train):
            for idx in range(n):
                data = dataset.get(idx)
                losses, pred, tgt = self._forward_one(data)
                l = losses["loss_total"]

                if train:
                    self.opt.zero_grad()
                    l.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()

                for k in ["loss_data", "loss_physics", "loss_total"]:
                    total[k] += float(losses[k])
                total["n"] += 1
                all_pred.append(pred.detach().cpu())
                all_tgt.append(tgt.detach().cpu())

        avg = {k: total[k] / max(total["n"], 1)
                for k in ["loss_data", "loss_physics", "loss_total"]}

        # R² calculation
        p_all = torch.cat(all_pred, dim=0).numpy().ravel()
        t_all = torch.cat(all_tgt,  dim=0).numpy().ravel()
        ss_res = np.sum((p_all - t_all)**2)
        ss_tot = np.sum((t_all - t_all.mean())**2) + 1e-9
        avg["r2"] = float(1 - ss_res / ss_tot)
        return avg

    def fit(self):
        print(f"\n[Trainer] Starting training  epochs={self.max_epochs}  "
              f"train={len(self.train_ds)}  val={len(self.val_ds)}")

        for epoch in range(1, self.max_epochs + 1):
            tr_stats = self._epoch(self.train_ds, train=True)
            va_stats = self._epoch(self.val_ds,   train=False)
            self.sched.step(va_stats["loss_total"])

            lr_now = self.opt.param_groups[0]["lr"]
            self.history["train_loss"].append(tr_stats["loss_total"])
            self.history["val_loss"].append(va_stats["loss_total"])
            self.history["val_r2"].append(va_stats["r2"])
            self.history["lr"].append(lr_now)

            print(f"  Epoch {epoch:4d}/{self.max_epochs}  "
                  f"tr_loss={tr_stats['loss_total']:.4f}  "
                  f"va_loss={va_stats['loss_total']:.4f}  "
                  f"va_R²={va_stats['r2']:.4f}  "
                  f"lr={lr_now:.2e}")

            if self.plotter is not None:
                self.plotter.update(
                    epoch      = epoch,
                    train_loss = tr_stats["loss_total"],
                    val_loss   = va_stats["loss_total"],
                    val_r2     = va_stats["r2"],
                    lr         = lr_now,
                )

            # Save best model
            if va_stats["loss_total"] < self.best_val_loss:
                self.best_val_loss = va_stats["loss_total"]
                self.no_improve    = 0
                ckpt = self.out_dir / "best_model.pt"
                torch.save({"epoch": epoch,
                             "model_state": self.model.state_dict(),
                             "val_loss": self.best_val_loss,
                             "val_r2":   va_stats["r2"]}, ckpt)
                print(f"    * Best model saved (val_loss={self.best_val_loss:.4f})")
            else:
                self.no_improve += 1

            if self.no_improve >= self.patience:
                print(f"  [EarlyStopping] No improvement for {self.patience} epochs, stopping training.")
                break

        # Save training history
        hist_path = self.out_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[Trainer] Training completed  best val_loss={self.best_val_loss:.4f}")
        print(f"  Training history saved: {hist_path}")

        if self.plotter is not None:
            self.plotter.save("training_curve_final.png")
            self.plotter.close()

        return self.history


# ════════════════════════════════════════════════════════════════
# 4. Main Program
# ════════════════════════════════════════════════════════════════
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(cfg_path:      str  = "../../00_config/urbangraph_params.yaml",
          h5_path:      str  = "../../01_data_generation/outputs/raw_simulations/ground_truth.h5",
          scenario_pkl: str  = "../../01_data_generation/outputs/raw_simulations/scenarios.pkl",
          epw_pkl:      str  = "../../01_data_generation/outputs/raw_simulations/epw_data.pkl",
          out_dir:      str  = "checkpoints",
          device_str:   str  = "auto",
          live_plot:    bool = True):

    cfg = load_cfg(cfg_path) if Path(cfg_path).exists() else {}

    # Default hyperparameters
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg["model"].setdefault("hidden_dim",    128)
    cfg["model"].setdefault("n_rgcn_layers", 3)
    cfg["model"].setdefault("lstm_hidden",   256)
    cfg["model"].setdefault("out_timesteps", 11)
    cfg["training"].setdefault("lr",           1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("max_epochs",   200)
    cfg["training"].setdefault("early_stopping_patience", 20)

    # Device
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    print(f"[Train] Device: {device}")

    # Dataset
    import pickle, __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC   # fix: pickle stored classes as __main__.X
    __main__.EPWData       = _ED
    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)

    train_ds = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split="train")
    val_ds   = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split="val")

    cfg["model"]["out_timesteps"] = len(train_ds.sim_hours)

    # Model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model Parameters: {n_params:,}")

    # Training
    trainer = Trainer(model, train_ds, val_ds, epw, cfg, device, out_dir,
                      live_plot=live_plot)
    history = trainer.fit()
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="../../00_config/urbangraph_params.yaml")
    ap.add_argument("--h5",       default="../../01_data_generation/outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--scenarios",default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",      default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--out",      default="checkpoints")
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--live-plot",    dest="live_plot", action="store_true",
                    default=True,  help="Show real-time loss curve window (default: on)")
    ap.add_argument("--no-live-plot", dest="live_plot", action="store_false",
                    help="Disable live curve window (headless / SSH mode)")
    args = ap.parse_args()
    main(args.config, args.h5, args.scenarios, args.epw, args.out,
         args.device, args.live_plot)