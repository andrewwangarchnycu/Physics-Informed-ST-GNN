"""
04_training/ablation.py
════════════════════════════════════════════════════════════════
Train all model variants for the ablation study.

Variants
--------
V0  MLP-per-node baseline  (no graph, no LSTM)
V3  Full architecture, no physics loss
V4  + L_rad only
V5  + L_rad + L_temp
V6  Full model — proposed  (L_rad + L_temp + L_wind)

Usage:
    python ablation.py --epochs 200 --device cuda
    python ablation.py --variants V0,V3,V6   # subset
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── path bootstrap ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import pickle, __main__
from shared     import HourlyClimate as _HC, EPWData as _ED
__main__.HourlyClimate = _HC
__main__.EPWData       = _ED

from dataset    import UTCIGraphDataset
from urbangraph import UrbanGraph, build_model
from train      import Trainer, build_env_time_seq
from evaluate   import evaluate


# ── Lambda configs per variant ─────────────────────────────────
VARIANT_LAMBDAS = {
    "V3": {"lambda1": 0.0,  "lambda2": 0.0,  "lambda3": 0.0},
    "V4": {"lambda1": 0.1,  "lambda2": 0.0,  "lambda3": 0.0},
    "V5": {"lambda1": 0.1,  "lambda2": 0.05, "lambda3": 0.0},
    "V6": {"lambda1": 0.1,  "lambda2": 0.05, "lambda3": 0.05},
}


# ── MLP Baseline (V0) ──────────────────────────────────────────
class MLPBaseline(nn.Module):
    """Per-node MLP: no graph, no LSTM. Flattens air_feat and predicts T steps."""

    def __init__(self, dim_air: int = 9, hidden: int = 256, T: int = 11):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(dim_air * T, hidden), nn.ReLU(),
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,      T),
        )

    def forward(self, obj_feat, air_feat, dynamic_edges, static_edges,
                env_seq, time_seq):
        N, T, D = air_feat.shape
        x = air_feat.reshape(N, T * D)
        return self.net(x)          # (N, T)

    def compute_loss(self, pred, target, **kwargs):
        import torch.nn.functional as F
        l = F.mse_loss(pred, target)
        return {
            "loss_total":   l,
            "loss_data":    l,
            "loss_physics": torch.tensor(0.0, device=pred.device),
        }


# ── Build model for a given variant ───────────────────────────
def build_variant(variant_id: str, out_timesteps: int,
                  dim_air: int, device: str) -> nn.Module:
    base_cfg = {
        "hidden_dim":   128,
        "n_rgcn_layers": 3,
        "lstm_hidden":  256,
        "lstm_layers":   1,
        "out_timesteps": out_timesteps,
        "dropout":       0.1,
    }
    if variant_id == "V0":
        return MLPBaseline(dim_air=dim_air, T=out_timesteps).to(device)

    if variant_id in VARIANT_LAMBDAS:
        cfg = {
            **base_cfg,
            "dim_air": dim_air,
            "lambdas": VARIANT_LAMBDAS[variant_id],
        }
        return UrbanGraph(**cfg).to(device)

    raise ValueError(f"Unknown variant: {variant_id}")


# ── Train + evaluate one variant ──────────────────────────────
def run_variant(variant_id: str,
                h5:         str,
                scenarios:  str,
                epw_pkl:    str,
                out_base:   str,
                epochs:     int,
                device:     str) -> dict:

    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)

    dim_air  = 9
    train_ds = UTCIGraphDataset(h5, scenarios, epw_pkl, split="train", dim_air=dim_air)
    val_ds   = UTCIGraphDataset(h5, scenarios, epw_pkl, split="val",   dim_air=dim_air)
    test_ds  = UTCIGraphDataset(h5, scenarios, epw_pkl, split="test",  dim_air=dim_air)

    out_dir = Path(out_base) / variant_id
    model   = build_variant(variant_id, len(train_ds.sim_hours), dim_air, device)

    cfg = {
        "training": {
            "lr":                       1e-3,
            "weight_decay":             1e-4,
            "max_epochs":               epochs,
            "early_stopping_patience": 25,
        }
    }
    trainer = Trainer(model, train_ds, val_ds, epw, cfg, device,
                      out_dir=str(out_dir), live_plot=False)
    history = trainer.fit()

    # Load best checkpoint and evaluate on test set
    ckpt_path = out_dir / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    metrics = evaluate(model, test_ds, epw, device)
    metrics["variant"]       = variant_id
    metrics["best_val_loss"] = min(history["val_loss"])
    return metrics


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Ablation study: train all model variants.")
    ap.add_argument("--h5",
        default="../../01_data_generation/outputs/raw_simulations/ground_truth_v2.h5")
    ap.add_argument("--scenarios",
        default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",
        default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--out",      default="ablation_ckpts")
    ap.add_argument("--epochs",   type=int, default=200)
    ap.add_argument("--variants", default="V0,V3,V4,V5,V6")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    variant_list = args.variants.split(",")
    all_results: list[dict] = []

    for vid in variant_list:
        print(f"\n{'='*60}\nTraining variant {vid}\n{'='*60}")
        metrics = run_variant(vid, args.h5, args.scenarios, args.epw,
                              args.out, args.epochs, args.device)
        all_results.append(metrics)
        print(
            f"  {vid}  R²={metrics['R2']:.4f}  "
            f"RMSE={metrics['RMSE']:.3f}°C  "
            f"MAE={metrics['MAE']:.3f}°C  "
            f"CatAcc={metrics['category_accuracy']*100:.1f}%"
        )

    out_path = Path(args.out) / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
