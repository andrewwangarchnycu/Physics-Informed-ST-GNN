"""
viz_training.py  — Phase 3+4 Model Training Verification Visualization
Run: cd 04_training && python viz_training.py
"""
from __future__ import annotations
import sys, json, argparse, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_graph_construction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_model"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import torch

from dataset    import UTCIGraphDataset
from urbangraph import UrbanGraph, build_model
from evaluate   import compute_metrics, UTCI_THRESHOLDS, UTCI_LABELS, utci_to_class
from train      import build_env_time_seq


UTCI_CMAP = LinearSegmentedColormap.from_list("utci", [
    "#313695","#4575b4","#74add1","#abd9e9",
    "#ffffbf","#fdae61","#f46d43","#d73027","#a50026"], N=256)


# ════════════════════════════════════════════════════════════════
# Fig A: Training Convergence Curve (loss + R²)
# ════════════════════════════════════════════════════════════════
def fig_training_curves(history_json: str, out_dir: Path):
    with open(history_json, "r") as f:
        hist = json.load(f)

    epochs     = list(range(1, len(hist["train_loss"])+1))
    train_loss = hist["train_loss"]
    val_loss   = hist["val_loss"]
    val_r2     = hist["val_r2"]
    lr_hist    = hist.get("lr", [1e-3]*len(epochs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Phase 4 — Training Convergence Curve", fontsize=13)

    # Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", c="#2c7bb6", lw=1.8)
    axes[0].plot(epochs, val_loss,   label="Val Loss",   c="#d7191c", lw=1.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE + Physics Loss")
    axes[0].set_title("Loss Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # R²
    axes[1].plot(epochs, val_r2, c="#1a9641", lw=1.8, label="Val R²")
    axes[1].axhline(0.90, ls="--", c="orange", lw=1.5, label="Deployment Threshold R²=0.90")
    best_r2 = max(val_r2)
    best_ep = epochs[val_r2.index(best_r2)]
    axes[1].axvline(best_ep, ls=":", c="#d7191c", lw=1.5,
                     label=f"Best Epoch={best_ep}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("R²")
    axes[1].set_title(f"R² Convergence (Best={best_r2:.4f})")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(max(0, min(val_r2)-0.05), 1.0)

    # Learning Rate
    axes[2].semilogy(epochs, lr_hist, c="#7b2d8b", lw=1.8)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule"); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "figA_training_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


# ════════════════════════════════════════════════════════════════
# Fig B: Predicted vs Ground Truth Scatter
# ════════════════════════════════════════════════════════════════
def fig_scatter_pred_vs_gt(model: UrbanGraph, dataset: UTCIGraphDataset,
                             epw, device: str, out_dir: Path,
                             n_samples: int = 20):
    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)

    norm_stats = dataset.norm_stats
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]

    all_pred, all_tgt = [], []
    with torch.no_grad():
        for idx in range(min(n_samples, len(dataset))):
            data = dataset.get(idx)
            obj_feat = data["object"].x.to(device)
            air_feat = data["air"].x.to(device)
            target   = data["air"].y.to(device)
            static_e = {}
            for rel in ["semantic","contiguity"]:
                key = ("object",rel,"object") if rel=="semantic" else ("air",rel,"air")
                if hasattr(data[key],"edge_index"):
                    static_e[rel] = data[key].edge_index.to(device)
            dyn = getattr(data,"dynamic_edges",[{}]*air_feat.shape[1])
            pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)
            all_pred.append((pred.cpu().numpy() * std + mean).ravel())
            all_tgt.append((target.cpu().numpy() * std + mean).ravel())

    p_all = np.concatenate(all_pred)
    t_all = np.concatenate(all_tgt)
    pred_norm_all = (np.concatenate(all_pred) - mean) / std
    tgt_norm_all  = (np.concatenate(all_tgt)  - mean) / std
    metrics = compute_metrics(pred_norm_all, tgt_norm_all, norm_stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Phase 4 — Prediction vs Ground Truth  (n={n_samples} [REMOVED_ZH:2])",
                  fontsize=12)

    # [REMOVED_ZH:3]
    ax  = axes[0]
    lim = [min(t_all.min(), p_all.min())-2, max(t_all.max(), p_all.max())+2]
    ax.scatter(t_all, p_all, c=t_all, cmap=UTCI_CMAP,
                vmin=25, vmax=50, s=3, alpha=0.3)
    ax.plot(lim, lim, "k--", lw=1.5, label="1:1 line")
    ax.set_xlabel("Ground Truth UTCI [°C]")
    ax.set_ylabel("Predicted UTCI [°C]")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f}°C  "
                  f"MAE={metrics['MAE']:.2f}°C")
    ax.legend(); ax.grid(True, alpha=0.3)

    # [REMOVED_ZH:5]
    ax2 = axes[1]
    res = p_all - t_all
    ax2.hist(res, bins=60, color="#2980b9", alpha=0.75, edgecolor="none")
    ax2.axvline(0, c="k", ls="--", lw=1.5)
    ax2.axvline(metrics["RMSE"], c="r", ls=":", lw=1.5,
                 label=f"RMSE={metrics['RMSE']:.2f}°C")
    ax2.axvline(-metrics["RMSE"], c="r", ls=":", lw=1.5)
    ax2.set_xlabel("Residual (Pred − GT) [°C]")
    ax2.set_ylabel("Count")
    ax2.set_title(f"[REMOVED_ZH:4]  bias={res.mean():.3f}°C")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p = out_dir / "figB_scatter_test.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


# ════════════════════════════════════════════════════════════════
# Fig C: [REMOVED_ZH:2] R² [REMOVED_ZH:2] (hour-by-hour)
# ════════════════════════════════════════════════════════════════
def fig_hourly_r2(model: UrbanGraph, dataset: UTCIGraphDataset,
                   epw, device: str, out_dir: Path, n_samples: int = 30):
    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)
    T        = len(sim_hours)
    norm_stats = dataset.norm_stats
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]

    pred_by_t = [[] for _ in range(T)]
    tgt_by_t  = [[] for _ in range(T)]

    model.eval()
    with torch.no_grad():
        for idx in range(min(n_samples, len(dataset))):
            data = dataset.get(idx)
            obj_feat = data["object"].x.to(device)
            air_feat = data["air"].x.to(device)
            target   = data["air"].y.to(device)
            static_e = {}
            for rel in ["semantic","contiguity"]:
                key = ("object",rel,"object") if rel=="semantic" else ("air",rel,"air")
                if hasattr(data[key],"edge_index"):
                    static_e[rel] = data[key].edge_index.to(device)
            dyn  = getattr(data,"dynamic_edges",[{}]*T)
            pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)

            for t in range(T):
                p_t = (pred[:, t].cpu().numpy() * std + mean)
                y_t = (target[:, t].cpu().numpy() * std + mean)
                pred_by_t[t].append(p_t)
                tgt_by_t[t].append(y_t)

    r2_list = []
    for t in range(T):
        p = np.concatenate(pred_by_t[t])
        y = np.concatenate(tgt_by_t[t])
        ss_res = np.sum((p-y)**2)
        ss_tot = np.sum((y-y.mean())**2)+1e-9
        r2_list.append(1 - ss_res/ss_tot)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sim_hours, r2_list, color=[cm.RdYlGn(r) for r in r2_list], alpha=0.85)
    ax.axhline(0.90, ls="--", c="orange", lw=1.5, label="[REMOVED_ZH:2] R²=0.90")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("R²")
    ax.set_title("Phase 4 — [REMOVED_ZH:2] R²（[REMOVED_ZH:11]）", fontsize=11)
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(True, alpha=0.3)
    for hr, r2 in zip(sim_hours, r2_list):
        ax.text(hr, r2+0.01, f"{r2:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    p = out_dir / "figC_hourly_r2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


# ════════════════════════════════════════════════════════════════
# Fig D: UTCI Thermal Stress Class Confusion Matrix
# ════════════════════════════════════════════════════════════════
def fig_confusion_matrix(model: UrbanGraph, dataset: UTCIGraphDataset,
                           epw, device: str, out_dir: Path, n_samples: int = 50):
    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)
    norm_stats = dataset.norm_stats
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]
    n_cls = len(UTCI_LABELS)

    conf = np.zeros((n_cls, n_cls), dtype=int)

    with torch.no_grad():
        for idx in range(min(n_samples, len(dataset))):
            data = dataset.get(idx)
            obj_feat = data["object"].x.to(device)
            air_feat = data["air"].x.to(device)
            target   = data["air"].y.to(device)
            static_e = {}
            for rel in ["semantic","contiguity"]:
                key = ("object",rel,"object") if rel=="semantic" else ("air",rel,"air")
                if hasattr(data[key],"edge_index"):
                    static_e[rel] = data[key].edge_index.to(device)
            dyn  = getattr(data,"dynamic_edges",[{}]*air_feat.shape[1])
            pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)

            p_vals = pred.cpu().numpy().ravel() * std + mean
            t_vals = target.cpu().numpy().ravel() * std + mean
            p_cls  = utci_to_class(p_vals)
            t_cls  = utci_to_class(t_vals)
            for tc, pc in zip(t_cls, p_cls):
                if tc < n_cls and pc < n_cls:
                    conf[tc, pc] += 1

    # [REMOVED_ZH:6]
    conf_norm = conf.astype(float) / (conf.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax   = plt.subplots(figsize=(9, 7))
    im  = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Row-normalized Accuracy")

    short_labels = ["<9", "9–26", "26–32", "32–38", "38–46", ">46"]
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(short_labels)
    ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")
    ax.set_title("UTCI Thermal Stress Class Confusion Matrix ([REMOVED_ZH:5])", fontsize=12)

    for i in range(n_cls):
        for j in range(n_cls):
            v = conf_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if v > 0.5 else "black")

    overall_acc = conf.diagonal().sum() / conf.sum()
    ax.set_xlabel(f"Predicted  (Overall Acc={overall_acc*100:.1f}%)")

    fig.tight_layout()
    p = out_dir / "figD_confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


# ════════════════════════════════════════════════════════════════
# Fig E: [REMOVED_ZH:4] UTCI [REMOVED_ZH:2] vs GT [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def fig_spatial_comparison(model: UrbanGraph, dataset: UTCIGraphDataset,
                             epw, device: str, out_dir: Path):
    """[REMOVED_ZH:3] test [REMOVED_ZH:2]，[REMOVED_ZH:7] GT / Pred / Error [REMOVED_ZH:2]。"""
    model.eval()
    sim_hours = dataset.sim_hours
    env_seq, time_seq = build_env_time_seq(epw, sim_hours)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)
    norm_stats = dataset.norm_stats
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]

    data     = dataset.get(0)
    obj_feat = data["object"].x.to(device)
    air_feat = data["air"].x.to(device)
    target   = data["air"].y.to(device)
    air_pos  = data["air"].pos.numpy()

    static_e = {}
    for rel in ["semantic","contiguity"]:
        key = ("object",rel,"object") if rel=="semantic" else ("air",rel,"air")
        if hasattr(data[key],"edge_index"):
            static_e[rel] = data[key].edge_index.to(device)
    dyn = getattr(data,"dynamic_edges",[{}]*air_feat.shape[1])

    with torch.no_grad():
        pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)

    # [REMOVED_ZH:5] (t_idx=4 → 12:00)
    t_show = min(4, len(sim_hours)-1)
    gt_vals   = target[:, t_show].cpu().numpy() * std + mean
    pred_vals = pred[:,   t_show].cpu().numpy() * std + mean
    err_vals  = pred_vals - gt_vals

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Phase 4 — [REMOVED_ZH:6]  {sim_hours[t_show]:02d}:00", fontsize=12)

    vmin, vmax = min(gt_vals.min(), pred_vals.min()), max(gt_vals.max(), pred_vals.max())
    for ax, vals, title in zip(axes,
                                  [gt_vals, pred_vals, err_vals],
                                  ["Ground Truth", "Prediction", "Error (Pred-GT)"]):
        if "Error" in title:
            cmap  = "RdBu_r"
            vm    = max(abs(err_vals.min()), abs(err_vals.max()))
            vmin2, vmax2 = -vm, vm
        else:
            cmap  = UTCI_CMAP
            vmin2, vmax2 = vmin, vmax
        sc = ax.scatter(air_pos[:,0], air_pos[:,1], c=vals,
                         cmap=cmap, vmin=vmin2, vmax=vmax2, s=25, marker="s")
        plt.colorbar(sc, ax=ax, label="UTCI [°C]" if "Error" not in title else "ΔT [°C]",
                      shrink=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    fig.tight_layout()
    p = out_dir / "figE_spatial_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")


# ════════════════════════════════════════════════════════════════
# Main Program
# ════════════════════════════════════════════════════════════════
def main(ckpt_path:    str = "checkpoints/best_model.pt",
          history_json: str = "checkpoints/training_history.json",
          h5_path:     str = "../../01_data_generation/outputs/raw_simulations/ground_truth.h5",
          scenario_pkl: str = "../../01_data_generation/outputs/raw_simulations/scenarios.pkl",
          epw_pkl:     str = "../../01_data_generation/outputs/raw_simulations/epw_data.pkl",
          out_dir:     str = "viz_output/training"):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("\n[viz_training] ── Phase 3+4 Model Training Verification Visualization ──")

    # [REMOVED_ZH:4]（[REMOVED_ZH:2] JSON）
    if Path(history_json).exists():
        fig_training_curves(history_json, out)
    else:
        print(f"  [[REMOVED_ZH:2]] [REMOVED_ZH:3] {history_json}")

    # [REMOVED_ZH:7]
    if not Path(ckpt_path).exists():
        print(f"  [[REMOVED_ZH:2]] [REMOVED_ZH:3] {ckpt_path}，[REMOVED_ZH:2]Run train.py")
        return

    import __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC
    __main__.EPWData       = _ED
    with open(epw_pkl, "rb") as f:
        epw = pickle.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device)

    ds_test = UTCIGraphDataset(h5_path, scenario_pkl, epw_pkl, split="test")
    cfg = {"model": {"out_timesteps": len(ds_test.sim_hours)}}
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    fig_scatter_pred_vs_gt(model, ds_test, epw, device, out, n_samples=20)
    fig_hourly_r2(model, ds_test, epw, device, out, n_samples=30)
    fig_confusion_matrix(model, ds_test, epw, device, out, n_samples=50)
    fig_spatial_comparison(model, ds_test, epw, device, out)

    print(f"\n[viz_training] [REMOVED_ZH:2]，[REMOVED_ZH:5] {out}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    default="checkpoints/best_model.pt")
    ap.add_argument("--history", default="checkpoints/training_history.json")
    ap.add_argument("--h5",      default="../../01_data_generation/outputs/raw_simulations/ground_truth.h5")
    ap.add_argument("--scenarios",default="../../01_data_generation/outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--epw",     default="../../01_data_generation/outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--out",     default="viz_output/training")
    args = ap.parse_args()
    main(args.ckpt, args.history, args.h5, args.scenarios, args.epw, args.out)