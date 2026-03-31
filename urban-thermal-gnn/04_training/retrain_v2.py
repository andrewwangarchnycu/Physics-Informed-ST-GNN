"""
04_training/retrain_v2.py
════════════════════════════════════════════════════════════════
Retrain PIN-ST-GNN with 9D air features (V2: surface temperature).

This script:
  1. Loads the dataset with dim_air=9 (adds surface temperature feature)
  2. Optionally warm-starts from an existing V1 (8D) checkpoint
  3. Trains the V2 model and saves to checkpoints_v2/
  4. Runs evaluation and prints metrics

Run:
  cd 04_training
  python retrain_v2.py
  python retrain_v2.py --warm-start checkpoints/best_model.pt
  python retrain_v2.py --no-live-plot  # headless mode
"""
from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import numpy as np
import torch
import yaml

from dataset   import UTCIGraphDataset
from urbangraph import UrbanGraph, build_model, DIM_AIR_V1, DIM_AIR_V2
from train     import Trainer, build_env_time_seq


def load_cfg(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def warm_start_from_v1(model: UrbanGraph, v1_ckpt_path: str, device: str):
    """
    Load V1 (8D) checkpoint into V2 (9D) model with partial weight transfer.

    The air_encoder first Linear layer changes from (hidden, 8) to (hidden, 9).
    We copy the 8 matching columns and zero-init the 9th column.
    All other weights are copied directly.
    """
    v1_state = torch.load(v1_ckpt_path, map_location=device, weights_only=False)
    v1_weights = v1_state["model_state"]
    v2_state = model.state_dict()

    loaded, skipped = 0, 0
    for key in v2_state:
        if key not in v1_weights:
            skipped += 1
            continue

        v1_param = v1_weights[key]
        v2_param = v2_state[key]

        if v1_param.shape == v2_param.shape:
            v2_state[key] = v1_param
            loaded += 1
        elif key == "air_encoder.net.0.weight":
            # V1: (hidden, 8) → V2: (hidden, 9)
            # Copy first 8 columns, zero-init 9th
            hidden = v1_param.shape[0]
            v2_state[key][:, :DIM_AIR_V1] = v1_param
            v2_state[key][:, DIM_AIR_V1:] = 0.0  # zero-init new feature dim
            loaded += 1
            print(f"  [warm-start] {key}: {v1_param.shape} → {v2_param.shape} "
                  f"(padded 9th column with zeros)")
        elif key == "air_encoder.net.0.bias":
            v2_state[key] = v1_param
            loaded += 1
        else:
            print(f"  [warm-start] shape mismatch, skipping {key}: "
                  f"{v1_param.shape} vs {v2_param.shape}")
            skipped += 1

    model.load_state_dict(v2_state)
    epoch = v1_state.get("epoch", "?")
    r2 = v1_state.get("val_r2", "?")
    print(f"  [warm-start] Loaded {loaded} params from V1 "
          f"(epoch={epoch}, val_R²={r2}), skipped {skipped}")
    return model


def main(cfg_path:      str  = "../../00_config/urbangraph_params.yaml",
          h5_path:      str  = "../../01_data_generation/outputs/raw_simulations/ground_truth.h5",
          scenario_pkl: str  = "../../01_data_generation/outputs/raw_simulations/scenarios.pkl",
          epw_pkl:      str  = "../../01_data_generation/outputs/raw_simulations/epw_data.pkl",
          out_dir:      str  = "../checkpoints_v2",
          warm_start:   str  = "",
          device_str:   str  = "auto",
          live_plot:    bool = True):

    cfg = load_cfg(cfg_path)
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})

    # Force dim_air=9 for V2
    cfg["model"]["dim_air"] = DIM_AIR_V2
    cfg["model"].setdefault("hidden_dim",    128)
    cfg["model"].setdefault("n_rgcn_layers", 3)
    cfg["model"].setdefault("lstm_hidden",   256)
    cfg["model"].setdefault("out_timesteps", 11)
    cfg["training"].setdefault("lr",           1e-3)
    cfg["training"].setdefault("weight_decay", 1e-4)
    cfg["training"].setdefault("max_epochs",   200)
    cfg["training"].setdefault("early_stopping_patience", 20)

    # Use lower LR if warm-starting (fine-tuning)
    if warm_start:
        cfg["training"]["lr"] = float(cfg["training"].get("lr", 1e-3)) * 0.5
        print(f"[retrain_v2] Warm-start mode: lr reduced to {cfg['training']['lr']}")

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else device_str
    print(f"[retrain_v2] Device: {device}")
    print(f"[retrain_v2] DIM_AIR: {DIM_AIR_V2} (V2, with surface temperature)")

    # EPW data
    import pickle, __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED
    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)

    # Dataset with dim_air=9
    print(f"[retrain_v2] Loading dataset with dim_air={DIM_AIR_V2}...")
    train_ds = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl,
                                 split="train", dim_air=DIM_AIR_V2)
    val_ds   = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl,
                                 split="val", dim_air=DIM_AIR_V2)

    cfg["model"]["out_timesteps"] = len(train_ds.sim_hours)

    # Verify feature dimension
    sample = train_ds.get(0)
    actual_dim = sample["air"].x.shape[2]
    print(f"[retrain_v2] Dataset air_feat shape: {tuple(sample['air'].x.shape)} "
          f"(dim_air={actual_dim})")
    assert actual_dim == DIM_AIR_V2, \
        f"Expected dim_air={DIM_AIR_V2} but got {actual_dim}"

    # Build V2 model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[retrain_v2] Model parameters: {n_params:,}")
    print(f"[retrain_v2] Air encoder input dim: {model.dim_air}")

    # Warm-start from V1 checkpoint if provided
    if warm_start and Path(warm_start).exists():
        print(f"[retrain_v2] Warm-starting from: {warm_start}")
        model = warm_start_from_v1(model, warm_start, device)
    elif warm_start:
        print(f"[retrain_v2] WARNING: warm-start path not found: {warm_start}")

    # Train
    trainer = Trainer(model, train_ds, val_ds, epw, cfg, device, out_dir,
                      live_plot=live_plot)
    history = trainer.fit()

    # ── Evaluate on test set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[retrain_v2] Evaluating V2 model on test set...")
    print("=" * 60)

    from evaluate import evaluate, compute_metrics

    test_ds = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl,
                                split="test", dim_air=DIM_AIR_V2)

    # Load best checkpoint
    best_ckpt = Path(out_dir) / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best checkpoint: epoch={ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?'):.4f}")

    metrics = evaluate(model, test_ds, epw, device)

    print(f"\n{'─' * 50}")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.3f} °C")
    print(f"  MAE  = {metrics['MAE']:.3f} °C")
    print(f"  Category Accuracy = {metrics['category_accuracy']*100:.1f}%")
    print(f"\n  Per-category accuracy:")
    for label, acc in metrics["per_category_accuracy"].items():
        print(f"    {label}: {acc*100:.1f}%")
    print(f"{'─' * 50}")

    if metrics["R2"] >= 0.90:
        print("\n  ✓ R² ≥ 0.90 — PASSES deployment threshold!")
    else:
        print(f"\n  ✗ R² = {metrics['R2']:.4f} < 0.90 — consider more training or tuning.")

    # Save metrics
    metrics_path = Path(out_dir) / "eval_results_v2.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved: {metrics_path}")

    return history, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Retrain PIN-ST-GNN with 9D air features (surface temperature)")
    ap.add_argument("--config",     default="../../00_config/urbangraph_params.yaml")
    ap.add_argument("--h5",         default="../../01_data_generation/outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--scenarios",  default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",        default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--out",        default="../checkpoints_v2")
    ap.add_argument("--warm-start", default="checkpoints/best_model.pt",
                    help="Path to V1 (8D) checkpoint for warm-start (default: checkpoints/best_model.pt)")
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_const",
                    const="",
                    help="Train from scratch without warm-starting")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--live-plot",  dest="live_plot", action="store_true",
                    default=True)
    ap.add_argument("--no-live-plot", dest="live_plot", action="store_false")
    args = ap.parse_args()

    main(args.config, args.h5, args.scenarios, args.epw, args.out,
         args.warm_start, args.device, args.live_plot)
