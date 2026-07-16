"""
04_training/generate_figures.py
════════════════════════════════════════════════════════════════
Produce all paper figures from saved checkpoints.

Outputs (saved to --out, default figures/):
  fig3_scatter.png        predicted vs. ground-truth UTCI scatter
  fig4_residuals.png      residual histogram
  fig5_hourly_r2.png      per-hour R² bar chart
  fig6_confusion.png      6-class UTCI confusion matrix (normalized)
  fig7_learning_curves.png  train/val loss + R² from training_history.json
  fig8_ablation.png       ablation bar chart from ablation_results.json

Usage:
    python generate_figures.py --ckpt ../checkpoints_v2/best_model.pt
    python generate_figures.py \\
        --ckpt    ../checkpoints_v2/best_model.pt \\
        --ablation ablation_ckpts/ablation_results.json \\
        --history  ../checkpoints_v2/training_history.json \\
        --out      figures/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ── UTCI helpers ──────────────────────────────────────────────
UTCI_THRESHOLDS = [-40, 9, 26, 32, 38, 46, 200]
UTCI_LABELS_SHORT = ["<9", "9–26", "26–32", "32–38", "38–46", ">46"]


def utci_to_class(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=int)
    for i, (lo, hi) in enumerate(
            zip(UTCI_THRESHOLDS[:-1], UTCI_THRESHOLDS[1:])):
        out[(arr >= lo) & (arr < hi)] = i
    return out


# ── Individual figure functions ───────────────────────────────
def fig_scatter(pred: np.ndarray, tgt: np.ndarray,
                out_path: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tgt, pred, s=0.3, alpha=0.3, color="#378ADD", rasterized=True)
    lo = min(tgt.min(), pred.min()) - 1
    hi = max(tgt.max(), pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.0, label="1:1 line")
    r2   = 1 - np.sum((pred - tgt)**2) / (np.sum((tgt - tgt.mean())**2) + 1e-9)
    rmse = np.sqrt(np.mean((pred - tgt)**2))
    mae  = np.mean(np.abs(pred - tgt))
    ax.text(0.05, 0.95,
            f"R² = {r2:.4f}\nRMSE = {rmse:.2f}°C\nMAE = {mae:.2f}°C",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    ax.set_xlabel("Ground-Truth UTCI (°C)")
    ax.set_ylabel("Predicted UTCI (°C)")
    ax.set_title("Test Set: Predicted vs. Ground-Truth UTCI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_residuals(pred: np.ndarray, tgt: np.ndarray,
                  out_path: Path, dpi: int = 300) -> None:
    res = pred - tgt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(res, bins=80, color="#639922", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="red", lw=1)
    ax.set_xlabel("Residual (Predicted − GT) [°C]")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution  (μ={res.mean():.3f}°C, σ={res.std():.3f}°C)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_hourly_r2(pred_shaped: list, tgt_shaped: list,
                  hours: list, out_path: Path, dpi: int = 300) -> None:
    T = pred_shaped[0].shape[1]
    r2s = []
    for t in range(T):
        p_t = np.concatenate([p[:, t] for p in pred_shaped])
        g_t = np.concatenate([g[:, t] for g in tgt_shaped])
        ss_res = np.sum((p_t - g_t)**2)
        ss_tot = np.sum((g_t - g_t.mean())**2) + 1e-9
        r2s.append(1 - ss_res / ss_tot)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(hours, r2s, color="#534AB7", edgecolor="none", alpha=0.85)
    ax.axhline(0.99, color="red", lw=1, ls="--", label="R²=0.99 threshold")
    ax.set_ylim(0.9, 1.01)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("R²")
    ax.set_title("Per-Hour Validation R²")
    ax.set_xticks(hours)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_confusion(pred: np.ndarray, tgt: np.ndarray,
                  out_path: Path, dpi: int = 300) -> None:
    pred_cls = utci_to_class(pred)
    tgt_cls  = utci_to_class(tgt)
    cm = confusion_matrix(tgt_cls, pred_cls,
                           labels=list(range(6)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(UTCI_LABELS_SHORT, fontsize=8)
    ax.set_yticklabels(UTCI_LABELS_SHORT, fontsize=8)
    ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")
    ax.set_title("UTCI Thermal-Stress Classification (Normalized)")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_learning_curves(history_json: str, out_path: Path,
                        dpi: int = 300) -> None:
    with open(history_json) as f:
        h = json.load(f)
    ep = list(range(1, len(h["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogy(ep, h["train_loss"], label="Train loss", color="#58a6ff")
    ax1.semilogy(ep, h["val_loss"],   label="Val loss",   color="#ff7b72")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss (log)")
    ax1.set_title("Loss Curves"); ax1.legend(fontsize=8)
    ax2.plot(ep, h["val_r2"], color="#56d364")
    ax2.axhline(0.99, color="orange", lw=1, ls="--", label="R²=0.99")
    ax2.set_ylim(0.8, 1.01)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val R²")
    ax2.set_title("Validation R²"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_ablation(ablation_json: str, out_path: Path, dpi: int = 300) -> None:
    with open(ablation_json) as f:
        results = json.load(f)
    names = [r["variant"] for r in results]
    r2s   = [r["R2"]      for r in results]
    rmses = [r["RMSE"]    for r in results]
    x     = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x, r2s,   color="#534AB7", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0.8, 1.01)
    ax1.axhline(0.99, color="red", lw=1, ls="--")
    ax1.set_title("Ablation: R² on Test Set"); ax1.set_ylabel("R²")
    ax2.bar(x, rmses, color="#D85A30", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_title("Ablation: RMSE on Test Set (°C)"); ax2.set_ylabel("RMSE (°C)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate all paper figures from saved checkpoints.")
    ap.add_argument("--ckpt",
        default="../checkpoints_v2/best_model.pt")
    ap.add_argument("--h5",
        default="../../01_data_generation/outputs/raw_simulations/ground_truth_v2.h5")
    ap.add_argument("--scenarios",
        default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",
        default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--history",
        default="../checkpoints_v2/training_history.json")
    ap.add_argument("--ablation",
        default="ablation_ckpts/ablation_results.json")
    ap.add_argument("--out",    default="figures")
    ap.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ── bootstrap imports ─────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

    import pickle, __main__
    from shared     import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED

    from dataset    import UTCIGraphDataset
    from urbangraph import build_model
    from train      import build_env_time_seq

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.epw, "rb") as f:
        epw = pickle.load(f)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    # Auto-detect dim_air from checkpoint
    enc_w   = ckpt["model_state"].get("air_encoder.net.0.weight")
    dim_air = int(enc_w.shape[1]) if enc_w is not None else 9
    print(f"  Auto-detected dim_air={dim_air} from checkpoint")

    test_ds = UTCIGraphDataset(
        args.h5, args.scenarios, args.epw, split="test", dim_air=dim_air)

    cfg   = {"model": {"out_timesteps": len(test_ds.sim_hours), "dim_air": dim_air}}
    model = build_model(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env_seq, time_seq = build_env_time_seq(epw, test_ds.sim_hours, month=7)
    env_seq  = env_seq.to(args.device)
    time_seq = time_seq.to(args.device)

    # Collect predictions
    pred_shaped: list = []
    tgt_shaped:  list = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            data     = test_ds.get(idx)
            obj_feat = data["object"].x.to(args.device)
            air_feat = data["air"].x.to(args.device)
            target   = data["air"].y.to(args.device)
            static_edges: dict = {}
            for rel in ["semantic", "contiguity"]:
                key = ("object", rel, "object") if rel == "semantic" \
                      else ("air", rel, "air")
                if hasattr(data[key], "edge_index"):
                    static_edges[rel] = data[key].edge_index.to(args.device)
            dyn  = getattr(data, "dynamic_edges",
                           [{}] * air_feat.shape[1])
            pred = model(obj_feat, air_feat, dyn,
                          static_edges, env_seq, time_seq)
            pred_shaped.append(pred.cpu().numpy())
            tgt_shaped.append(target.cpu().numpy())

    # Denormalize
    ns   = test_ds.norm_stats
    mean = ns["utci"]["mean"]
    std  = ns["utci"]["std"]
    pred_dn = [p * std + mean for p in pred_shaped]
    tgt_dn  = [t * std + mean for t in tgt_shaped]
    pred_flat = np.concatenate([p.ravel() for p in pred_dn])
    tgt_flat  = np.concatenate([t.ravel() for t in tgt_dn])

    print("\n[generate_figures] Producing figures …")

    # fig3 — scatter
    fig_scatter(pred_flat, tgt_flat, out_dir / "fig3_scatter.png")

    # fig4 — residuals
    fig_residuals(pred_flat, tgt_flat, out_dir / "fig4_residuals.png")

    # fig5 — hourly R²
    fig_hourly_r2(pred_dn, tgt_dn, test_ds.sim_hours,
                  out_dir / "fig5_hourly_r2.png")

    # fig6 — confusion matrix
    fig_confusion(pred_flat, tgt_flat, out_dir / "fig6_confusion.png")

    # fig7 — learning curves (skip if history missing)
    hist = Path(args.history)
    if hist.exists():
        fig_learning_curves(str(hist), out_dir / "fig7_learning_curves.png")
    else:
        print(f"  [skip] training history not found: {hist}")

    # fig8 — ablation (skip if json missing)
    abl = Path(args.ablation)
    if abl.exists():
        fig_ablation(str(abl), out_dir / "fig8_ablation.png")
    else:
        print(f"  [skip] ablation results not found: {abl}")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
