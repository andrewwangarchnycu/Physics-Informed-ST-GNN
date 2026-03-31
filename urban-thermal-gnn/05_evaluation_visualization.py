"""
05_evaluation_visualization.py
════════════════════════════════════════════════════════════════
Physics-Informed ST-GNN — [REMOVED_ZH:10]

[REMOVED_ZH:6]:
  Stage 01 — Dataset Statistics (ground_truth.h5 / scenarios.pkl)
  Stage 02 — Graph Structure Feature Analysis (node features / edge degree)
  Stage 04 — [REMOVED_ZH:4]and[REMOVED_ZH:4] (training_history / best_model.pt)
  Side-by-Side — GT vs [REMOVED_ZH:6] + [REMOVED_ZH:3] (test [REMOVED_ZH:2])

Run[REMOVED_ZH:2]:
  conda activate PytorchGPU
  cd "urban-thermal-gnn"
  python 05_evaluation_visualization.py

[REMOVED_ZH:2]: ./05_visualization_outputs/
  fig1_dataset_overview.png      — Stage 01 Dataset Statistics
  fig2_graph_features.png        — Stage 02 [REMOVED_ZH:5]
  fig3_training_curves.png       — Stage 04 Training Convergence Curve
  fig4_prediction_quality.png    — Stage 04 [REMOVED_ZH:6]
  fig5_spatial_comparison.png    — GT vs [REMOVED_ZH:6] (test [REMOVED_ZH:2])
  fig6_summary_dashboard.png     — [REMOVED_ZH:7]
"""
from __future__ import annotations

import sys
import json
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import h5py

# ── [REMOVED_ZH:4] ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE  # 05_evaluation_visualization.py [REMOVED_ZH:2] urban-thermal-gnn/ [REMOVED_ZH:1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "02_graph_construction"))
sys.path.insert(0, str(_ROOT / "03_model"))
sys.path.insert(0, str(_ROOT / "04_training"))

import torch

try:
    from dataset    import UTCIGraphDataset
    from urbangraph import UrbanGraph, build_model
    from evaluate   import compute_metrics, UTCI_THRESHOLDS, UTCI_LABELS, utci_to_class
    from train      import build_env_time_seq
    from shared     import EPWData, HourlyClimate
    _IMPORTS_OK = True
except ImportError as e:
    warnings.warn(f"Pipeline import failed: {e}\n  Some plots may be skipped.")
    _IMPORTS_OK = False

# ── [REMOVED_ZH:4] ──────────────────────────────────────────────────
H5_PATH      = _ROOT / "01_data_generation/outputs/raw_simulations/ground_truth.h5"
SCENARIOS_PKL = _ROOT / "01_data_generation/outputs/raw_simulations/scenarios.pkl"
EPW_PKL      = _ROOT / "01_data_generation/outputs/raw_simulations/epw_data.pkl"
CKPT_PATH    = _ROOT / "04_training/checkpoints/best_model.pt"
HISTORY_JSON = _ROOT / "04_training/checkpoints/training_history.json"
EVAL_JSON    = _ROOT / "04_training/checkpoints/eval_results.json"
OUT_DIR      = _ROOT / "05_visualization_outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── UTCI [REMOVED_ZH:2]（[REMOVED_ZH:1]→[REMOVED_ZH:1]→[REMOVED_ZH:1]，ISO 15743 [REMOVED_ZH:2]）──────────────────────
# ── Sensing-augmented model paths (v2 / with-sensing checkpoint) ─────────────
CKPT_PATH_V2    = _ROOT / "checkpoints_v2/best_model.pt"
EVAL_JSON_V2    = _ROOT / "checkpoints_v2/eval_results.json"
HISTORY_JSON_V2 = _ROOT / "checkpoints_v2/training_history.json"

UTCI_CMAP = LinearSegmentedColormap.from_list("utci", [
    "#313695", "#4575b4", "#74add1", "#abd9e9",
    "#ffffbf", "#fdae61", "#f46d43", "#d73027", "#a50026"
], N=256)

UTCI_CLASS_COLORS = ["#313695", "#74add1", "#ffffbf",
                     "#fdae61", "#d73027", "#7b0000"]

SIM_HOURS = list(range(8, 19))   # 8:00–18:00 (11 timesteps)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",   # Latin + math; avoids CJK font missing glyphs
    "axes.unicode_minus": False,
    "figure.dpi":         100,
    "mathtext.fontset":   "dejavusans",
})


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════

def _load_epw() -> "EPWData | None":
    if not EPW_PKL.exists():
        return None
    import __main__
    __main__.HourlyClimate = HourlyClimate
    __main__.EPWData       = EPWData
    with open(EPW_PKL, "rb") as f:
        return pickle.load(f)


def _load_model(device: str, prefer_v2: bool = True) -> "UrbanGraph | None":
    """Load best available checkpoint: V2 (with-sensing) preferred, V1 fallback."""
    ckpt_path = CKPT_PATH_V2 if (prefer_v2 and CKPT_PATH_V2.exists()) else CKPT_PATH
    if not ckpt_path.exists():
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return None
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = {"model": {"out_timesteps": len(SIM_HOURS)}}
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tag   = "V2 (sensing)" if ckpt_path == CKPT_PATH_V2 else "V1 (base)"
    epoch = ckpt.get("epoch", "?")
    r2    = ckpt.get("val_r2", "?")
    print(f"  [{tag}] Model loaded — epoch={epoch}  val_R²={r2}")
    return model


def _compute_sensor_utci(ta_field: np.ndarray,
                          rh_field: np.ndarray,
                          va_field: np.ndarray,
                          mrt_field: np.ndarray) -> np.ndarray:
    """
    Compute sensor-estimated UTCI from station observations using the
    pythermalcomfort formula (Bröde 2012). Differs from LBT UTCI which
    uses full CFD radiation modelling.
    ta/rh/va/mrt: (N,) arrays at a single timestep.
    Returns: (N,) UTCI in °C.
    """
    ws  = np.clip(va_field,  0.5, 20.0)
    mrt = np.clip(mrt_field, ta_field - 5, ta_field + 70)
    try:
        from pythermalcomfort.models import utci as _utci_fn
        res = _utci_fn(tdb=ta_field.ravel(), tr=mrt.ravel(),
                       v=ws.ravel(), rh=rh_field.ravel(), units="SI")
        return np.array(res["utci"], dtype=np.float32)
    except Exception:
        # Bröde 2012 linear approximation fallback
        return (ta_field + 0.33 * (mrt - ta_field) - 0.7 * ws - 4.0).astype(np.float32)


def _run_inference(model, dataset, epw, device, indices) -> tuple[list, list, list]:
    """
    [REMOVED_ZH:2] (all_pred_denorm, all_tgt_denorm, sensor_pts_list)
    [REMOVED_ZH:3] shape: (N_air, T)
    """
    norm_stats = dataset.norm_stats
    mean = norm_stats["utci"]["mean"]
    std  = norm_stats["utci"]["std"]

    env_seq, time_seq = build_env_time_seq(epw, dataset.sim_hours)
    env_seq  = env_seq.to(device)
    time_seq = time_seq.to(device)

    preds, tgts, pts = [], [], []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            data = dataset.get(idx)
            obj_feat = data["object"].x.to(device)
            air_feat = data["air"].x.to(device)
            target   = data["air"].y.to(device)
            static_e = {}
            for rel in ["semantic", "contiguity"]:
                key = ("object", rel, "object") if rel == "semantic" \
                      else ("air", rel, "air")
                if hasattr(data[key], "edge_index"):
                    static_e[rel] = data[key].edge_index.to(device)
            dyn  = getattr(data, "dynamic_edges", [{}] * air_feat.shape[1])
            pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)

            preds.append((pred.cpu().numpy() * std + mean))
            tgts.append((target.cpu().numpy() * std + mean))
            pts.append(data["air"].pos.numpy())

    return preds, tgts, pts


def _scatter_with_density(ax, x, y, cmap="Blues", s=2, alpha=0.15, max_pts=50000):
    """[REMOVED_ZH:9]"""
    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y = x[idx], y[idx]
    ax.scatter(x, y, c=x, cmap=cmap, s=s, alpha=alpha, rasterized=True)


def _save(fig: plt.Figure, name: str):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ════════════════════════════════════════════════════════════════
# Figure 1 — Dataset Statistics (Stage 01)
# ════════════════════════════════════════════════════════════════

def fig1_dataset_overview():
    """
    Panel A: [REMOVED_ZH:1]Dataset UTCI [REMOVED_ZH:2]（violin + jitter）
    Panel B: [REMOVED_ZH:2] UTCI [REMOVED_ZH:3] ± IQR
    Panel C: UTCI Thermal Stress[REMOVED_ZH:4]
    Panel D: [REMOVED_ZH:7]（[REMOVED_ZH:7]）
    """
    print("\n[Fig 1] Dataset Overview...")
    if not H5_PATH.exists():
        print(f"  [skip] {H5_PATH} not found")
        return

    with h5py.File(H5_PATH, "r") as hf:
        norm_stats = {}
        for field in hf["normalization"].keys():
            grp = hf[f"normalization/{field}"]
            norm_stats[field] = {
                "mean": float(grp.attrs["mean"]),
                "std":  float(grp.attrs["std"]),
            }
        mean_u = norm_stats["utci"]["mean"]
        std_u  = norm_stats["utci"]["std"]

        train_ids = [str(i) for i in hf["splits/train_ids"][:]]
        val_ids   = [str(i) for i in hf["splits/val_ids"][:]]
        test_ids  = [str(i) for i in hf["splits/test_ids"][:]]
        all_ids   = train_ids + val_ids + test_ids
        sim_hours = list(hf["metadata/sim_hours"][:])

        # [REMOVED_ZH:1]Dataset UTCI ([REMOVED_ZH:3])
        utci_by_hour = [[] for _ in sim_hours]
        n_sensors_per_scene = []
        utci_flat_all = []

        for sid in all_ids:
            grp = hf[f"scenarios/{sid}"]
            utci_norm = grp["utci"][:]         # (T, N)
            utci = utci_norm * std_u + mean_u  # denormalize
            N = utci.shape[1]
            n_sensors_per_scene.append(N)
            for t_idx, _ in enumerate(sim_hours):
                utci_by_hour[t_idx].append(utci[t_idx])
            # [REMOVED_ZH:6] 200 [REMOVED_ZH:1] air node [REMOVED_ZH:2]
            sample = utci.ravel()
            if len(sample) > 200:
                sample = sample[np.random.choice(len(sample), 200, replace=False)]
            utci_flat_all.append(sample)

    utci_all_np    = np.concatenate(utci_flat_all)
    utci_by_hour_m = [np.median(np.concatenate(h)) for h in utci_by_hour]
    utci_by_hour_q1 = [np.percentile(np.concatenate(h), 25) for h in utci_by_hour]
    utci_by_hour_q3 = [np.percentile(np.concatenate(h), 75) for h in utci_by_hour]

    # UTCI [REMOVED_ZH:4]
    classes = utci_to_class(utci_all_np)
    class_counts = np.array([np.sum(classes == i) for i in range(len(UTCI_LABELS))])
    class_pct    = class_counts / class_counts.sum() * 100

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Stage 01 — Dataset Statistics  (LBT Ground-Truth Simulation)\n"
        f"{len(train_ids)} train  /  {len(val_ids)} val  /  {len(test_ids)} test scenarios",
        fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # A: Full-dataset UTCI distribution
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.hist(utci_all_np, bins=80, color="#2980b9", alpha=0.75, edgecolor="none")
    for lo, hi, col, _lbl in zip(UTCI_THRESHOLDS[:-1], UTCI_THRESHOLDS[1:],
                                  UTCI_CLASS_COLORS, UTCI_LABELS):
        ax_a.axvspan(max(lo, utci_all_np.min()-1),
                     min(hi, utci_all_np.max()+1),
                     alpha=0.07, color=col)
    ax_a.set_xlabel("UTCI (°C)"); ax_a.set_ylabel("Count")
    ax_a.set_title(f"Full-Dataset UTCI Distribution  "
                   f"(μ={utci_all_np.mean():.1f} °C,  σ={utci_all_np.std():.1f} °C)")
    ax_a.grid(True, alpha=0.3)

    # B: Diurnal median UTCI ± IQR
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_b.plot(sim_hours, utci_by_hour_m, "o-", color="#c0392b", lw=2,
              ms=6, label="Median UTCI")
    ax_b.fill_between(sim_hours, utci_by_hour_q1, utci_by_hour_q3,
                      alpha=0.25, color="#e74c3c", label="IQR (Q1–Q3)")
    for threshold, label_str in [(32, "Strong (32 °C)"), (38, "Very Strong (38 °C)")]:
        ax_b.axhline(threshold, ls="--", lw=1.2, alpha=0.6)
        ax_b.text(sim_hours[-1]+0.1, threshold, label_str, va="center", fontsize=8)
    ax_b.set_xlabel("Hour of Day"); ax_b.set_ylabel("UTCI (°C)")
    ax_b.set_title("Diurnal UTCI Median ± IQR  (all 300 scenarios)")
    ax_b.set_xticks(sim_hours); ax_b.legend(); ax_b.grid(True, alpha=0.3)

    # C: Thermal Stress Category Distribution
    ax_c = fig.add_subplot(gs[1, :2])
    nonzero = [(pct, lbl, col)
               for pct, lbl, col in zip(class_pct, UTCI_LABELS, UTCI_CLASS_COLORS)
               if pct > 0.5]
    wedge_pcts  = [x[0] for x in nonzero]
    wedge_lbls  = [f"{x[1]}\n{x[0]:.1f}%" for x in nonzero]
    wedge_cols  = [x[2] for x in nonzero]
    ax_c.pie(wedge_pcts, labels=wedge_lbls, colors=wedge_cols,
             autopct="", startangle=90,
             wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax_c.set_title("UTCI Thermal Stress Category Distribution\n(ISO 15743 / Fiala 2012)")

    # D: Air-node count per scenario
    ax_d = fig.add_subplot(gs[1, 2:])
    n_arr = np.array(n_sensors_per_scene)
    train_n = n_arr[:len(train_ids)]
    val_n   = n_arr[len(train_ids):len(train_ids)+len(val_ids)]
    test_n  = n_arr[len(train_ids)+len(val_ids):]
    bins = np.linspace(n_arr.min()-1, n_arr.max()+1, 30)
    ax_d.hist(train_n, bins=bins, color="#2196F3", alpha=0.6, label=f"Train ({len(train_ids)})")
    ax_d.hist(val_n,   bins=bins, color="#FF9800", alpha=0.6, label=f"Val ({len(val_ids)})")
    ax_d.hist(test_n,  bins=bins, color="#4CAF50", alpha=0.6, label=f"Test ({len(test_ids)})")
    ax_d.axvline(n_arr.mean(), ls="--", c="k", lw=1.5,
                 label=f"Mean = {n_arr.mean():.0f}")
    ax_d.set_xlabel("Air Node Count (N_air)"); ax_d.set_ylabel("Scenario Count")
    ax_d.set_title("Air Sensor Node Count per Scenario\n(Train / Val / Test splits)")
    ax_d.legend(); ax_d.grid(True, alpha=0.3)

    _save(fig, "fig1_dataset_overview.png")


# ════════════════════════════════════════════════════════════════
# Figure 2 — Graph Structure Feature Analysis (Stage 02)
# ════════════════════════════════════════════════════════════════

def fig2_graph_features():
    """
    Panel A: Air node [REMOVED_ZH:5] (boxplot × 8 features)
    Panel B: Object node [REMOVED_ZH:5] (boxplot × 7 features)
    Panel C: [REMOVED_ZH:11]（[REMOVED_ZH:1] SVF [REMOVED_ZH:2]）
    Panel D: [REMOVED_ZH:7]（height vs footprint_area）
    """
    print("\n[Fig 2] Graph Feature Analysis...")
    if not H5_PATH.exists() or not SCENARIOS_PKL.exists():
        print("  [skip] h5 or scenarios pkl not found")
        return

    with open(SCENARIOS_PKL, "rb") as f:
        scenarios = pickle.load(f)

    scen_map = {str(s["scenario_id"]): s for s in scenarios}

    AIR_FEAT_NAMES  = ["ta_norm", "mrt_norm", "va_norm", "rh_norm",
                       "SVF", "in_shadow", "bldg_h/50", "tree_h/12"]
    OBJ_FEAT_NAMES  = ["height/50", "floors/12", "footprint/2000",
                       "cx/80", "cy/80", "GFA/20000", "is_L"]

    air_feats_samples = [[] for _ in range(8)]
    obj_feats_samples = [[] for _ in range(7)]
    sample_sid = None
    sample_pts = None
    sample_svf = None
    bldg_heights, bldg_areas = [], []

    with h5py.File(H5_PATH, "r") as hf:
        norm_stats = {}
        for field in hf["normalization"].keys():
            grp = hf[f"normalization/{field}"]
            norm_stats[field] = {
                "mean": float(grp.attrs["mean"]),
                "std":  float(grp.attrs["std"]),
            }
        all_ids = ([str(i) for i in hf["splits/train_ids"][:]] +
                   [str(i) for i in hf["splits/val_ids"][:]] +
                   [str(i) for i in hf["splits/test_ids"][:]])

        # [REMOVED_ZH:2] 60 [REMOVED_ZH:3]
        sample_ids = all_ids[:60]

        for sid in sample_ids:
            grp   = hf[f"scenarios/{sid}"]
            pts   = grp["sensor_pts"][:]        # (N, 2)
            svf   = grp["svf"][:]               # (N,)
            ta_n  = grp["ta"][:].mean(axis=0)   # (N,) time-mean
            mrt_n = grp["mrt"][:].mean(axis=0)
            va_n  = grp["va"][:].mean(axis=0)
            rh_n  = grp["rh"][:].mean(axis=0)
            shadow = grp["in_shadow"][:].mean(axis=0)
            bh    = grp["building_height"][:] / 50.0
            th    = grp["tree_height"][:] / 12.0

            N = min(len(pts), 300)
            idx_s = np.random.choice(len(pts), N, replace=False)
            for i, arr in enumerate([ta_n[idx_s], mrt_n[idx_s], va_n[idx_s],
                                      rh_n[idx_s], svf[idx_s], shadow[idx_s],
                                      bh[idx_s], th[idx_s]]):
                air_feats_samples[i].extend(arr.tolist())

            if sample_sid is None:
                sample_sid = sid
                sample_pts = pts
                sample_svf = svf

            # Object features
            if sid in scen_map:
                for bldg in scen_map[sid].get("buildings", []):
                    h  = float(bldg.get("height", 0))
                    fl = int(bldg.get("floors", 1))
                    fp = bldg.get("footprint")
                    # footprint may be a Shapely Polygon or list of coords
                    if fp is not None and hasattr(fp, "area"):
                        area = float(fp.area)
                    elif fp is not None and hasattr(fp, "__len__") and len(fp) >= 3:
                        fp_arr = np.array(fp)
                        xf, yf = fp_arr[:, 0], fp_arr[:, 1]
                        area = 0.5 * abs(np.dot(xf, np.roll(yf, 1)) -
                                         np.dot(yf, np.roll(xf, 1)))
                    else:
                        area = float(bldg.get("coverage", 0))

                    cen = bldg.get("centroid", (40.0, 40.0))
                    cx, cy = float(cen[0]), float(cen[1])
                    gfa  = float(bldg.get("gfa", h * area))
                    is_l = 1 if bldg.get("shape_type", "rect") != "rect" else 0

                    feat_vals = [h/50.0, fl/12.0, area/2000.0,
                                 cx/80.0, cy/80.0, gfa/20000.0, float(is_l)]
                    for i, v in enumerate(feat_vals):
                        obj_feats_samples[i].append(v)

                    bldg_heights.append(h)
                    bldg_areas.append(area)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Stage 02 — Graph Structure & Node Feature Analysis", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # A: Air node feature boxplot
    ax_a = fig.add_subplot(gs[0, :2])
    data_a = [np.array(x) for x in air_feats_samples]
    bp = ax_a.boxplot(data_a, patch_artist=True, notch=False, vert=True,
                      medianprops={"color": "k", "lw": 2},
                      flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
    colors_box = plt.cm.Set2(np.linspace(0, 1, 8))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax_a.set_xticklabels(AIR_FEAT_NAMES, rotation=20, ha="right", fontsize=8)
    ax_a.set_title("Air Node Feature Distribution  (60 scenarios × ≤300 nodes)")
    ax_a.set_ylabel("Normalized Value"); ax_a.grid(True, alpha=0.3)

    # B: Object node feature boxplot
    ax_b = fig.add_subplot(gs[0, 2:])
    data_b = [np.array(x) for x in obj_feats_samples]
    bp2 = ax_b.boxplot(data_b, patch_artist=True, notch=False, vert=True,
                       medianprops={"color": "k", "lw": 2},
                       flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
    colors_box2 = plt.cm.Set1(np.linspace(0, 1, 7))
    for patch, c in zip(bp2["boxes"], colors_box2):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax_b.set_xticklabels(OBJ_FEAT_NAMES, rotation=20, ha="right", fontsize=8)
    ax_b.set_title("Object (Building) Node Feature Distribution")
    ax_b.set_ylabel("Normalized Value"); ax_b.grid(True, alpha=0.3)

    # C: Sample scene — spatial SVF map
    ax_c = fig.add_subplot(gs[1, :2])
    if sample_pts is not None:
        sc = ax_c.scatter(sample_pts[:, 0], sample_pts[:, 1],
                          c=sample_svf, cmap="RdYlGn", s=12,
                          vmin=0, vmax=1, alpha=0.8)
        plt.colorbar(sc, ax=ax_c, label="Sky View Factor (SVF)")
    ax_c.set_aspect("equal"); ax_c.set_xlabel("X (m)"); ax_c.set_ylabel("Y (m)")
    ax_c.set_title(f"Sample Scene — Air Node Positions (SVF coloured)\nScenario: {sample_sid}")
    ax_c.grid(True, alpha=0.2)

    # D: Building height vs footprint area
    ax_d = fig.add_subplot(gs[1, 2:])
    if bldg_heights:
        h_arr = np.array(bldg_heights)
        a_arr = np.array(bldg_areas)
        ax_d.scatter(a_arr, h_arr, alpha=0.4, s=10, c="#3498db", edgecolors="none")
        ax_d.set_xlabel("Building Footprint Area (m²)"); ax_d.set_ylabel("Building Height (m)")
        ax_d.set_title(f"Building Height vs Footprint Area\n"
                       f"n={len(h_arr)} buildings,  "
                       f"avg_h={h_arr.mean():.1f} m,  avg_fp={a_arr.mean():.0f} m²")
        ax_d.grid(True, alpha=0.3)

    _save(fig, "fig2_graph_features.png")


# ════════════════════════════════════════════════════════════════
# Figure 3 — Training Convergence Curve (Stage 04)
# ════════════════════════════════════════════════════════════════

def fig3_training_curves():
    """
    Overlays V1 (base) and V2 (with sensing) training histories side-by-side.
    Panel A: Train / Val Loss convergence (log scale)
    Panel B: Validation R² convergence
    Panel C: Learning Rate schedule (V2)
    Panel D: Summary metrics card
    """
    print("\n[Fig 3] Training Curves (V1 vs V2)...")

    # ── Load available histories ──────────────────────────────────
    RUN_CFG = [
        ("V1 Base",          HISTORY_JSON,    EVAL_JSON,    "#2c7bb6", "#abd9e9"),
        ("V2 + Sensing",     HISTORY_JSON_V2, EVAL_JSON_V2, "#d7191c", "#fdae61"),
    ]
    runs = []
    for tag, h_path, e_path, c_main, c_light in RUN_CFG:
        if h_path.exists():
            with open(h_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            ev = {}
            if e_path.exists():
                with open(e_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
            runs.append((tag, hist, ev, c_main, c_light))
            print(f"  Loaded: {h_path}")

    if not runs:
        print(f"  [skip] no training history found")
        return

    # Use V2 (last run) for LR schedule and primary metrics card
    primary = runs[-1]
    p_tag, p_hist, p_ev, _, _ = primary
    p_epochs  = list(range(1, len(p_hist["train_loss"]) + 1))
    p_val_r2  = p_hist.get("val_r2", [0.0] * len(p_epochs))
    p_lr      = p_hist.get("lr", [1e-3] * len(p_epochs))
    p_best_ep = int(np.argmin(p_hist["val_loss"])) + 1
    p_best_r2 = float(np.max(p_val_r2))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(
        "Stage 04 — Model Training Convergence  ·  V1 Base vs V2 + Sensing",
        fontsize=13, fontweight="bold")

    # A: Loss curves (all runs overlaid)
    for tag, hist, _, c_main, c_light in runs:
        eps = list(range(1, len(hist["train_loss"]) + 1))
        axes[0].semilogy(eps, hist["train_loss"], c=c_main,  lw=1.8,
                         label=f"Train  {tag}")
        axes[0].semilogy(eps, hist["val_loss"],   c=c_light, lw=1.8, ls="--",
                         label=f"Val    {tag}")
        best_ep_r = int(np.argmin(hist["val_loss"])) + 1
        axes[0].axvline(best_ep_r, ls=":", c=c_main, lw=1.2, alpha=0.6)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Train / Val Loss Convergence")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # B: Val R² (all runs overlaid)
    for tag, hist, _, c_main, _ in runs:
        eps   = list(range(1, len(hist["train_loss"]) + 1))
        r2_h  = hist.get("val_r2", [0.0] * len(eps))
        best_ep_r = int(np.argmax(r2_h)) + 1
        best_r2_r = float(np.max(r2_h))
        axes[1].plot(eps, r2_h, c=c_main, lw=1.8, label=f"{tag}  (peak={best_r2_r:.4f})")
        axes[1].scatter([best_ep_r], [best_r2_r], c=c_main, s=60, zorder=5)
    axes[1].axhline(0.90, ls="--", c="orange", lw=1.4, alpha=0.8, label="Target R²=0.90")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val R²")
    axes[1].set_title("Validation R² Convergence")
    all_r2_flat = [v for _, h, *_ in runs for v in h.get("val_r2", [0])]
    axes[1].set_ylim(max(0.0, min(all_r2_flat) - 0.05), 1.01)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # C: Learning rate schedule (V2 / primary run)
    axes[2].semilogy(p_epochs, p_lr, c="#7b2d8b", lw=1.8)
    if len(p_lr) > 1:
        drops = [i+1 for i in range(1, len(p_lr)) if p_lr[i] < p_lr[i-1] * 0.6]
        for ep in drops:
            axes[2].axvline(ep, ls=":", c="gray", lw=1.0, alpha=0.6)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title(f"Learning Rate Schedule  ({p_tag})")
    axes[2].grid(True, alpha=0.3)

    # D: Summary metrics card (V2)
    axes[3].axis("off")
    lines = [
        f"╔═══  {p_tag}  ═══╗",
        f"  Best Epoch :  {p_best_ep}",
        f"  Peak Val R²:  {p_best_r2:.4f}",
        f"  Total Epochs: {len(p_epochs)}",
        "",
    ]
    if p_ev:
        lines += [
            f"  ── Test Results ──",
            f"  R²           : {p_ev.get('R2', 0):.4f}",
            f"  RMSE         : {p_ev.get('RMSE', 0):.3f} °C",
            f"  MAE          : {p_ev.get('MAE', 0):.3f} °C",
            f"  Cat. Accuracy: {p_ev.get('category_accuracy', 0)*100:.1f} %",
            f"  N samples    : {p_ev.get('n_samples', 'N/A'):,}",
        ]
    axes[3].text(0.05, 0.95, "\n".join(lines),
                 transform=axes[3].transAxes, verticalalignment="top",
                 fontsize=10, fontfamily="monospace",
                 bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f0f0",
                       "edgecolor": "#bbb", "alpha": 0.8})
    axes[3].set_title("Training Summary (Primary Run)")

    fig.tight_layout()
    _save(fig, "fig3_training_curves.png")


# ════════════════════════════════════════════════════════════════
# Figure 4 — [REMOVED_ZH:6] (Stage 04)
# ════════════════════════════════════════════════════════════════

def fig4_prediction_quality(model, dataset_test, epw, device):
    """
    Panel A: Pred vs GT [REMOVED_ZH:3]（hexbin）
    Panel B: [REMOVED_ZH:7]
    Panel C: [REMOVED_ZH:2] R² [REMOVED_ZH:3]
    Panel D: UTCI Thermal Stress Class Confusion Matrix
    """
    print("\n[Fig 4] Prediction Quality...")
    if model is None or dataset_test is None or epw is None:
        print("  [skip] model / dataset / epw not available")
        return

    n_test   = min(len(dataset_test), 40)
    indices  = list(range(n_test))
    preds, tgts, _ = _run_inference(model, dataset_test, epw, device, indices)

    norm_stats = dataset_test.norm_stats
    sim_hours  = dataset_test.sim_hours
    T          = len(sim_hours)

    p_all = np.concatenate([p.ravel() for p in preds])
    t_all = np.concatenate([t.ravel() for t in tgts])
    res   = p_all - t_all

    # per-hour R²
    r2_hour = []
    for t_idx in range(T):
        p_t = np.concatenate([p[:, t_idx] for p in preds])
        t_t = np.concatenate([t[:, t_idx] for t in tgts])
        ss_res = np.sum((p_t - t_t)**2)
        ss_tot = np.sum((t_t - t_t.mean())**2) + 1e-9
        r2_hour.append(float(1 - ss_res / ss_tot))

    # confusion matrix
    p_cls = utci_to_class(p_all)
    t_cls = utci_to_class(t_all)
    n_cls = len(UTCI_LABELS)
    conf  = np.zeros((n_cls, n_cls), dtype=int)
    for tc, pc in zip(t_cls, p_cls):
        if 0 <= tc < n_cls and 0 <= pc < n_cls:
            conf[tc, pc] += 1
    conf_norm = conf.astype(float) / (conf.sum(axis=1, keepdims=True) + 1e-9)

    # Metrics
    mean_u = norm_stats["utci"]["mean"]
    std_u  = norm_stats["utci"]["std"]
    p_norm = (p_all - mean_u) / std_u
    t_norm = (t_all - mean_u) / std_u
    metrics = compute_metrics(p_norm, t_norm, norm_stats)

    fig = plt.figure(figsize=(22, 9))
    fig.suptitle(
        f"Stage 04 — V2 Model Prediction Quality  ({n_test} test scenarios)",
        fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # A: hexbin scatter
    ax_a = fig.add_subplot(gs[0])
    lim_lo = min(t_all.min(), p_all.min()) - 1
    lim_hi = max(t_all.max(), p_all.max()) + 1
    hb = ax_a.hexbin(t_all, p_all, gridsize=60, cmap="Blues",
                     mincnt=1, extent=[lim_lo, lim_hi, lim_lo, lim_hi])
    ax_a.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=1.5, label="1:1 line")
    plt.colorbar(hb, ax=ax_a, label="Count")
    ax_a.set_xlabel("LBT Ground Truth UTCI (°C)")
    ax_a.set_ylabel("V2 Model Predicted UTCI (°C)")
    ax_a.set_title(f"Prediction vs Ground Truth\n"
                   f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f} °C  "
                   f"MAE={metrics['MAE']:.2f} °C")
    ax_a.legend(); ax_a.grid(True, alpha=0.3)

    # B: Residual distribution
    ax_b = fig.add_subplot(gs[1])
    ax_b.hist(res, bins=80, color="#2980b9", alpha=0.75, edgecolor="none", density=True)
    x_r = np.linspace(res.min(), res.max(), 200)
    from scipy.stats import norm as sp_norm
    mu_r, sig_r = res.mean(), res.std()
    ax_b.plot(x_r, sp_norm.pdf(x_r, mu_r, sig_r), "r-", lw=2, label="Normal fit")
    ax_b.axvline(0, c="k", ls="--", lw=1.5)
    ax_b.axvline( metrics["RMSE"], c="r", ls=":", lw=1.2,
                  label=f"±RMSE={metrics['RMSE']:.2f} °C")
    ax_b.axvline(-metrics["RMSE"], c="r", ls=":", lw=1.2)
    ax_b.set_xlabel("Residual  (V2 Pred − GT)  [°C]")
    ax_b.set_ylabel("Density")
    ax_b.set_title(f"Residual Distribution\nbias={mu_r:.3f} °C   σ={sig_r:.3f} °C")
    ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.3)

    # C: Per-hour R²
    ax_c = fig.add_subplot(gs[2])
    bar_colors = [cm.RdYlGn(r) for r in r2_hour]
    ax_c.bar(sim_hours, r2_hour, color=bar_colors, alpha=0.85, edgecolor="k", lw=0.5)
    ax_c.axhline(0.99, ls="--", c="green",  lw=1.2, alpha=0.8, label="R²=0.99")
    ax_c.axhline(0.90, ls="--", c="orange", lw=1.2, alpha=0.8, label="R²=0.90")
    for hr, r2 in zip(sim_hours, r2_hour):
        ax_c.text(hr, r2 + 0.002, f"{r2:.3f}", ha="center", va="bottom", fontsize=7)
    ax_c.set_xlabel("Hour of Day"); ax_c.set_ylabel("R²")
    ax_c.set_title("Per-Hour Prediction R²\n(V2 Model — all test scenarios)")
    ax_c.set_ylim(min(r2_hour) - 0.01, 1.01)
    ax_c.set_xticks(sim_hours); ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.3)

    # D: UTCI Thermal Stress Confusion Matrix
    ax_d = fig.add_subplot(gs[3])
    im = ax_d.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax_d, label="Row-normalised Accuracy")
    short_labels = ["<9°C", "9–26", "26–32", "32–38", "38–46", ">46°C"]
    ax_d.set_xticks(range(n_cls)); ax_d.set_xticklabels(short_labels, fontsize=8)
    ax_d.set_yticks(range(n_cls)); ax_d.set_yticklabels(short_labels, fontsize=8)
    ax_d.set_xlabel("Predicted Stress Class"); ax_d.set_ylabel("True Stress Class")
    overall_acc = metrics.get("category_accuracy", 0)
    ax_d.set_title(f"UTCI Thermal Stress Confusion Matrix\n"
                   f"Overall Accuracy = {overall_acc*100:.1f} %")
    for i in range(n_cls):
        for j in range(n_cls):
            v = conf_norm[i, j]
            ax_d.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if v > 0.6 else "black")

    fig.tight_layout()
    _save(fig, "fig4_prediction_quality.png")


# ════════════════════════════════════════════════════════════════
# Figure 5 — GT vs [REMOVED_ZH:6] (test [REMOVED_ZH:2])
# ════════════════════════════════════════════════════════════════

def fig5_spatial_comparison(model, dataset_test, epw, device, n_scenes=10):
    """
    Three-way spatial comparison at peak hour (14:00) for n_scenes test scenarios.

    4 columns per row:
      Col 0  LBT Simulation Ground Truth UTCI  (full CFD-based)
      Col 1  Sensor-Estimated UTCI             (pythermalcomfort formula from Ta/RH/va/Tmrt)
      Col 2  V2 ST-GNN Prediction UTCI
      Col 3  Absolute Error  |GT − V2 Pred|

    Row label shows scenario ID and per-scene RMSE / MAE.
    """
    print("\n[Fig 5] Spatial Comparison — LBT GT | Sensor-Est. | V2 Pred | |Error|")
    if model is None or dataset_test is None or epw is None:
        print("  [skip] model / dataset / epw not available")
        return
    if not H5_PATH.exists():
        print(f"  [skip] {H5_PATH} not found")
        return

    n_scenes  = min(n_scenes, len(dataset_test))
    indices   = list(range(n_scenes))
    preds, tgts, pts_list = _run_inference(model, dataset_test, epw, device, indices)

    sim_hours = dataset_test.sim_hours
    t_peak    = sim_hours.index(14) if 14 in sim_hours else len(sim_hours) // 2
    peak_lbl  = f"{sim_hours[t_peak]:02d}:00"

    # ── Load norm stats + test IDs ────────────────────────────────
    with h5py.File(H5_PATH, "r") as hf:
        norm_stats = {}
        for field in hf["normalization"].keys():
            grp = hf[f"normalization/{field}"]
            norm_stats[field] = {"mean": float(grp.attrs["mean"]),
                                  "std":  float(grp.attrs["std"])}
        test_ids = [str(i) for i in hf["splits/test_ids"][:]]

    def _dn(arr, key):
        return arr * norm_stats[key]["std"] + norm_stats[key]["mean"]

    # ── Load sensor-estimated UTCI for each scene ─────────────────
    sensor_utci_list = []
    scene_ids        = []
    with h5py.File(H5_PATH, "r") as hf:
        for idx in indices:
            sid = test_ids[idx]
            grp = hf[f"scenarios/{sid}"]
            ta_t   = _dn(grp["ta"][:],  "ta")[t_peak]
            rh_t   = _dn(grp["rh"][:],  "rh")[t_peak]
            va_t   = _dn(grp["va"][:],  "va")[t_peak]
            mrt_t  = _dn(grp["mrt"][:], "mrt")[t_peak]
            sensor_utci_list.append(_compute_sensor_utci(ta_t, rh_t, va_t, mrt_t))
            scene_ids.append(sid)

    # ── Shared UTCI colour scale ──────────────────────────────────
    all_utci = np.concatenate(
        [t[:, t_peak] for t in tgts] +
        [p[:, t_peak] for p in preds] +
        sensor_utci_list
    )
    v_lo, v_hi = np.percentile(all_utci, 2), np.percentile(all_utci, 98)
    utci_norm  = Normalize(vmin=v_lo, vmax=v_hi)

    all_err = np.concatenate([
        np.abs(t[:, t_peak] - p[:, t_peak]) for t, p in zip(tgts, preds)
    ])
    err_vmax = float(np.percentile(all_err, 98))

    COLS = 4
    CELL_W, CELL_H = 3.8, 3.6
    fig, axes = plt.subplots(n_scenes, COLS,
                              figsize=(CELL_W * COLS, CELL_H * n_scenes),
                              squeeze=False)
    fig.suptitle(
        f"Stage 04 — Three-way UTCI Comparison at Peak Hour {peak_lbl}\n"
        "LBT Simulation Ground Truth  |  Sensor-Estimated  |  V2 ST-GNN Prediction  |  |Error|",
        fontsize=12, fontweight="bold", y=1.002,
    )

    COL_TITLES = [
        f"LBT GT  (UTCI, °C)\n{peak_lbl}",
        f"Sensor-Estimated UTCI (°C)\n(pythermalcomfort / Bröde 2012)",
        f"V2 ST-GNN Prediction (°C)\n{peak_lbl}",
        "|GT − V2 Pred|  (°C)",
    ]

    for row, (pred_arr, tgt_arr, pts, sen_utci, sid) in enumerate(
            zip(preds, tgts, pts_list, sensor_utci_list, scene_ids)):

        px, py = pts[:, 0], pts[:, 1]
        ms = max(2, min(16, 2500 // max(len(px), 1)))
        rmse_s = float(np.sqrt(np.mean((tgt_arr[:, t_peak] - pred_arr[:, t_peak]) ** 2)))
        mae_s  = float(np.mean(np.abs(tgt_arr[:, t_peak]  - pred_arr[:, t_peak])))

        # Col 0: LBT GT
        sc0 = axes[row, 0].scatter(px, py, c=tgt_arr[:, t_peak],
                                    cmap=UTCI_CMAP, norm=utci_norm,
                                    s=ms, alpha=0.87, rasterized=True)
        plt.colorbar(sc0, ax=axes[row, 0], label="UTCI (°C)", shrink=0.85)

        # Col 1: Sensor-estimated
        sc1 = axes[row, 1].scatter(px, py, c=sen_utci,
                                    cmap=UTCI_CMAP, norm=utci_norm,
                                    s=ms, alpha=0.87, rasterized=True)
        plt.colorbar(sc1, ax=axes[row, 1], label="UTCI (°C)", shrink=0.85)

        # Col 2: V2 Prediction
        sc2 = axes[row, 2].scatter(px, py, c=pred_arr[:, t_peak],
                                    cmap=UTCI_CMAP, norm=utci_norm,
                                    s=ms, alpha=0.87, rasterized=True)
        plt.colorbar(sc2, ax=axes[row, 2], label="UTCI (°C)", shrink=0.85)

        # Col 3: |Error|
        err   = np.abs(tgt_arr[:, t_peak] - pred_arr[:, t_peak])
        sc3 = axes[row, 3].scatter(px, py, c=err,
                                    cmap="Reds", vmin=0, vmax=err_vmax,
                                    s=ms, alpha=0.87, rasterized=True)
        plt.colorbar(sc3, ax=axes[row, 3], label="|Error| (°C)", shrink=0.85)
        axes[row, 3].text(
            0.03, 0.97,
            f"RMSE={rmse_s:.2f} °C\nMAE ={mae_s:.2f} °C",
            transform=axes[row, 3].transAxes, va="top", ha="left",
            fontsize=7.5, bbox={"boxstyle": "round,pad=0.3",
                                 "facecolor": "white", "alpha": 0.80,
                                 "edgecolor": "#aaa"},
        )

        for col in range(COLS):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(COL_TITLES[col], fontsize=8.5, fontweight="bold", pad=4)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", fontsize=7)
            ax.set_ylabel("Y (m)" if col == 0 else "", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.12)
            if col == 0:
                ax.annotate(
                    f"#{row+1}  ID {sid}",
                    xy=(-0.30, 0.5), xycoords="axes fraction",
                    ha="right", va="center", fontsize=8, fontweight="bold",
                    rotation=90,
                )

    fig.tight_layout()
    _save(fig, "fig5_spatial_comparison.png")


# ════════════════════════════════════════════════════════════════
# Figure 6 — [REMOVED_ZH:7]
# ════════════════════════════════════════════════════════════════

def fig6_summary_dashboard(model, dataset_test, epw, device):
    """
    [REMOVED_ZH:7]：
    Panel A: Stage 01 UTCI [REMOVED_ZH:6]（CDF）
    Panel B: Stage 04 Train vs Val Loss（[REMOVED_ZH:2]50 epoch [REMOVED_ZH:2]）
    Panel C: Stage 04 Test [REMOVED_ZH:2] GT vs Pred [REMOVED_ZH:4]（[REMOVED_ZH:4]）
    Panel D: [REMOVED_ZH:1] UTCI Thermal Stress[REMOVED_ZH:8]
    """
    print("\n[Fig 6] Summary Dashboard...")

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        "PIN-ST-GNN  Summary Dashboard  ·  V2 Model + Sensing Integration",
        fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # ── Panel A: Test-set UTCI CDF ────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    if H5_PATH.exists():
        with h5py.File(H5_PATH, "r") as hf:
            mu  = float(hf["normalization/utci"].attrs["mean"])
            std = float(hf["normalization/utci"].attrs["std"])
            test_ids = [str(i) for i in hf["splits/test_ids"][:]]
            utci_vals = []
            for sid in test_ids[:20]:
                utci_n = hf[f"scenarios/{sid}/utci"][:].ravel()
                utci_vals.extend((utci_n * std + mu).tolist())
        u_arr    = np.array(utci_vals)
        u_sorted = np.sort(u_arr)
        cdf      = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
        ax_a.plot(u_sorted, cdf * 100, c="#2980b9", lw=2)
        for threshold, lbl, col in [(26, "Moderate", "green"),
                                     (32, "Strong",   "orange"),
                                     (38, "V.Strong",  "red")]:
            pct = float(np.searchsorted(u_sorted, threshold)) / len(u_sorted) * 100
            ax_a.axvline(threshold, ls="--", c=col, lw=1.2,
                         label=f"{lbl} ({100-pct:.0f}% above)")
        ax_a.set_xlabel("UTCI (°C)"); ax_a.set_ylabel("Cumulative (%)")
        ax_a.set_title("Test Scenario UTCI CDF\n(Stage 01 LBT Dataset)")
        ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.3)

    # ── Panel B: V2 training loss tail (last 50 epochs) ──────────
    ax_b = fig.add_subplot(gs[1])
    hist_src = HISTORY_JSON_V2 if HISTORY_JSON_V2.exists() else HISTORY_JSON
    if hist_src.exists():
        with open(hist_src, "r", encoding="utf-8") as f:
            hist = json.load(f)
        tl  = hist["train_loss"][-50:]
        vl  = hist["val_loss"][-50:]
        eps = list(range(len(hist["train_loss"]) - len(tl) + 1,
                         len(hist["train_loss"]) + 1))
        ax_b.plot(eps, tl, c="#2c7bb6", lw=1.8, label="Train")
        ax_b.plot(eps, vl, c="#d7191c", lw=1.8, label="Val")
        best_ep = int(np.argmin(hist["val_loss"])) + 1
        if best_ep >= eps[0]:
            ax_b.axvline(best_ep, ls=":", c="gray", lw=1.5,
                         label=f"Best epoch = {best_ep}")
        tag = "V2" if hist_src == HISTORY_JSON_V2 else "V1"
        ax_b.set_xlabel("Epoch"); ax_b.set_ylabel("Loss")
        ax_b.set_title(f"Loss Convergence — Last 50 Epochs  ({tag})\n(Stage 04 Training)")
        ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.3)

    # ── Panel C: UTCI diurnal sequence — 5 sampled air nodes ─────
    ax_c = fig.add_subplot(gs[2])
    if model is not None and dataset_test is not None and epw is not None:
        _p, _t, _ = _run_inference(model, dataset_test, epw, device, [0])
        pred0, tgt0 = _p[0], _t[0]   # (N_air, T)
        sim_h  = dataset_test.sim_hours
        n_air  = pred0.shape[0]
        chosen = np.linspace(0, n_air - 1, 5, dtype=int)
        for i, node_idx in enumerate(chosen):
            col = plt.cm.tab10(i / 10)
            ax_c.plot(sim_h, tgt0[node_idx],  "o-",  color=col, lw=1.5, ms=5,
                      label=f"GT #{node_idx}")
            ax_c.plot(sim_h, pred0[node_idx], "s--", color=col, lw=1.5, ms=4,
                      alpha=0.7, label=f"V2 Pred #{node_idx}")
        ax_c.set_xlabel("Hour of Day"); ax_c.set_ylabel("UTCI (°C)")
        ax_c.set_title("Diurnal UTCI Sequence — Sample Scenario\n"
                       "(5 Air Nodes: GT vs V2 Prediction)")
        ax_c.set_xticks(sim_h); ax_c.grid(True, alpha=0.3)
        handles, labels = ax_c.get_legend_handles_labels()
        ax_c.legend(handles[:10], labels[:10], fontsize=7, ncol=2)

    # ── Panel D: Per-class UTCI accuracy (V2 eval) ───────────────
    ax_d = fig.add_subplot(gs[3])
    eval_src = EVAL_JSON_V2 if EVAL_JSON_V2.exists() else EVAL_JSON
    if eval_src.exists():
        with open(eval_src, "r", encoding="utf-8") as f:
            eval_m = json.load(f)
        per_cls  = eval_m.get("per_category_accuracy", {})
        overall  = eval_m.get("category_accuracy", 0)
        labels_d = list(per_cls.keys())
        accs_d   = [v * 100 for v in per_cls.values()]
        colors_d = plt.cm.RdYlGn([a / 100 for a in accs_d])
        bars = ax_d.barh(labels_d, accs_d, color=colors_d, alpha=0.85,
                          edgecolor="k", lw=0.5)
        ax_d.axvline(overall * 100, ls="--", c="#333", lw=1.5,
                     label=f"Overall = {overall*100:.1f} %")
        for bar, acc in zip(bars, accs_d):
            ax_d.text(min(acc + 0.3, 101), bar.get_y() + bar.get_height() / 2,
                      f"{acc:.1f}%", va="center", fontsize=9)
        ax_d.set_xlim(80, 101); ax_d.set_xlabel("Accuracy (%)")
        tag_d = "V2" if eval_src == EVAL_JSON_V2 else "V1"
        ax_d.set_title(f"Per-class UTCI Thermal Stress Accuracy\n"
                       f"({tag_d} Model — ISO 15743 / Fiala 2012)")
        ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    _save(fig, "fig6_summary_dashboard.png")


# ════════════════════════════════════════════════════════════════
# Figure 7 — Multi-variable Environmental Comparison Array
# ════════════════════════════════════════════════════════════════

def fig7_multivariable_array(model, dataset_test, epw, device, n_scenes: int = 2):
    """
    Array grid at diurnal peak (14:00):
      Rows    = n_scenes test scenarios
      Columns = [Ta | Tmrt | va | RH | UTCI GT | UTCI Pred | |Error|]

    Bottom strip: diurnal mean profiles for all 5 variables.

    Ta / Tmrt / va / RH are simulation inputs (ground truth).
    UTCI columns show GT simulation vs ST-GNN prediction side-by-side.
    Designed for CAAD professionals evaluating urban thermal comfort.
    """
    print("\n[Fig 7] Multi-variable Environmental Array Comparison...")
    if model is None or dataset_test is None or epw is None:
        print("  [skip] model / dataset / epw not available")
        return
    if not H5_PATH.exists():
        print(f"  [skip] {H5_PATH} not found")
        return

    n_scenes = min(n_scenes, len(dataset_test))
    indices  = list(range(n_scenes))

    # ── Normalisation stats ───────────────────────────────────────
    with h5py.File(H5_PATH, "r") as hf:
        norm_stats = {}
        for field in hf["normalization"].keys():
            grp = hf[f"normalization/{field}"]
            norm_stats[field] = {"mean": float(grp.attrs["mean"]),
                                  "std":  float(grp.attrs["std"])}
        test_ids = [str(i) for i in hf["splits/test_ids"][:]]

    # ── Peak-hour index ───────────────────────────────────────────
    sim_hours = dataset_test.sim_hours
    t_peak    = sim_hours.index(14) if 14 in sim_hours else len(sim_hours) // 2
    peak_lbl  = f"{sim_hours[t_peak]:02d}:00"

    # ── UTCI inference ────────────────────────────────────────────
    preds, _, pts_list = _run_inference(model, dataset_test, epw, device, indices)

    def _dn(arr, key):
        return arr * norm_stats[key]["std"] + norm_stats[key]["mean"]

    # ── Per-scene data extraction ─────────────────────────────────
    scene_data = []
    with h5py.File(H5_PATH, "r") as hf:
        for i, idx in enumerate(indices):
            sid = test_ids[idx]
            grp = hf[f"scenarios/{sid}"]
            ta_all   = _dn(grp["ta"][:],   "ta")    # (T, N)
            mrt_all  = _dn(grp["mrt"][:],  "mrt")
            va_all   = _dn(grp["va"][:],   "va")
            rh_all   = _dn(grp["rh"][:],   "rh")
            utci_all = _dn(grp["utci"][:], "utci")
            scene_data.append({
                "sid":          sid,
                # spatial snapshots at peak hour
                "ta":           ta_all[t_peak],
                "mrt":          mrt_all[t_peak],
                "va":           va_all[t_peak],
                "rh":           rh_all[t_peak],
                "utci_gt":      utci_all[t_peak],
                "utci_pred":    preds[i][:, t_peak],
                "pts":          pts_list[i],
                # spatial-mean diurnal profiles
                "ta_ts":        ta_all.mean(axis=1),
                "mrt_ts":       mrt_all.mean(axis=1),
                "va_ts":        va_all.mean(axis=1),
                "rh_ts":        rh_all.mean(axis=1),
                "utci_gt_ts":   utci_all.mean(axis=1),
                "utci_pred_ts": preds[i].mean(axis=0),
            })

    # ── Variable configuration ────────────────────────────────────
    VAR_CFG = [
        # (key,         label,                         cmap,        vmin, vmax)
        ("ta",        r"$T_a$ (°C)",                  "RdYlBu_r",  None, None),
        ("mrt",       r"$T_{mrt}$ (°C)",               "RdYlBu_r",  None, None),
        ("va",        r"$v_a$ (m/s)",                  "Blues",     0.0,  None),
        ("rh",        r"$RH$ (%)",                     "BuGn",      0.0,  100.0),
        ("utci_gt",   "UTCI — Simulation (°C)",        UTCI_CMAP,   None, None),
        ("utci_pred", "UTCI — ST-GNN Pred (°C)",       UTCI_CMAP,   None, None),
    ]
    N_VAR = len(VAR_CFG)
    COLS  = N_VAR + 1   # last col = |error|

    vlims = {}
    for key, lbl, cmap, vlo, vhi in VAR_CFG:
        all_v = np.concatenate([d[key] for d in scene_data])
        vmin  = vlo if vlo is not None else float(np.percentile(all_v, 2))
        vmax  = vhi if vhi is not None else float(np.percentile(all_v, 98))
        vlims[key] = (vmin, vmax, cmap)

    err_vmax = float(np.percentile(
        np.concatenate([np.abs(d["utci_gt"] - d["utci_pred"]) for d in scene_data]), 98
    ))

    # ── Figure layout: spatial grid (top) + diurnal profiles (bottom) ──
    FIG_W = 3.2 * COLS
    FIG_H = 4.0 * n_scenes + 3.8
    fig   = plt.figure(figsize=(FIG_W, FIG_H))
    outer = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[n_scenes * 4.0, 3.8],
                               hspace=0.06)
    spatial_gs = gridspec.GridSpecFromSubplotSpec(
        n_scenes, COLS, subplot_spec=outer[0], hspace=0.30, wspace=0.26)

    fig.suptitle(
        f"Multi-variable Environmental Comparison Array  ·  Peak Hour {peak_lbl}\n"
        r"Columns: Simulation Inputs ($T_a$, $T_{mrt}$, $v_a$, $RH$)"
        "  |  UTCI Ground Truth  |  ST-GNN Prediction  |  Absolute Error",
        fontsize=12, fontweight="bold", y=1.003,
    )

    SCENE_COLORS = plt.cm.Set1(np.linspace(0, 0.7, n_scenes))

    for row, d in enumerate(scene_data):
        px, py = d["pts"][:, 0], d["pts"][:, 1]
        ms = max(2, min(18, 2500 // max(len(px), 1)))

        # --- variable columns ----------------------------------------
        for col, (key, lbl, cmap, *_) in enumerate(VAR_CFG):
            ax  = fig.add_subplot(spatial_gs[row, col])
            vmin, vmax, cmap_use = vlims[key]
            sc  = ax.scatter(px, py, c=d[key], cmap=cmap_use,
                             vmin=vmin, vmax=vmax, s=ms,
                             alpha=0.87, rasterized=True)
            cb  = plt.colorbar(sc, ax=ax, shrink=0.84, pad=0.01)
            cb.ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(lbl, fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("X (m)", fontsize=7)
            ax.set_ylabel("Y (m)" if col == 0 else "", fontsize=7)
            if col == 0:
                ax.annotate(
                    f"Scene {row + 1}  (ID {d['sid']})",
                    xy=(-0.35, 0.5), xycoords="axes fraction",
                    ha="right", va="center", fontsize=8, fontweight="bold",
                    rotation=90,
                )
            ax.tick_params(labelsize=6)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.12)

        # --- |error| column ------------------------------------------
        ax_e  = fig.add_subplot(spatial_gs[row, COLS - 1])
        err   = np.abs(d["utci_gt"] - d["utci_pred"])
        rmse_s = float(np.sqrt(np.mean((d["utci_gt"] - d["utci_pred"]) ** 2)))
        mae_s  = float(np.mean(err))
        sc_e  = ax_e.scatter(px, py, c=err, cmap="Reds",
                              vmin=0, vmax=err_vmax, s=ms,
                              alpha=0.87, rasterized=True)
        cb_e  = plt.colorbar(sc_e, ax=ax_e, label="|Error| (°C)",
                              shrink=0.84, pad=0.01)
        cb_e.ax.tick_params(labelsize=6)
        if row == 0:
            ax_e.set_title("|UTCI Error| (°C)", fontsize=9, fontweight="bold", pad=4)
        ax_e.set_xlabel("X (m)", fontsize=7)
        ax_e.tick_params(labelsize=6)
        ax_e.set_aspect("equal")
        ax_e.grid(True, alpha=0.12)
        ax_e.text(
            0.04, 0.97,
            f"RMSE = {rmse_s:.2f} °C\nMAE  = {mae_s:.2f} °C",
            transform=ax_e.transAxes, va="top", ha="left", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "alpha": 0.82, "edgecolor": "#aaa"},
        )

    # ── Diurnal mean profiles ─────────────────────────────────────
    diurnal_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], hspace=0.08, wspace=0.36)

    PROFILE_CFG = [
        ("ta_ts",                      r"$T_a$ (°C)",       False),
        ("mrt_ts",                     r"$T_{mrt}$ (°C)",    False),
        ("va_ts",                      r"$v_a$ (m/s)",       False),
        ("rh_ts",                      r"$RH$ (%)",          False),
        (("utci_gt_ts", "utci_pred_ts"), "UTCI (°C)",        True ),
    ]

    for c_idx, (key_cfg, ylabel, is_utci) in enumerate(PROFILE_CFG):
        ax_p = fig.add_subplot(diurnal_gs[0, c_idx])

        if is_utci:
            for s_idx, d in enumerate(scene_data):
                col = SCENE_COLORS[s_idx]
                ax_p.plot(sim_hours, d["utci_gt_ts"],   "o-",
                          color=col, lw=1.7, ms=4,
                          label=f"GT — Scene {s_idx + 1}")
                ax_p.plot(sim_hours, d["utci_pred_ts"], "s--",
                          color=col, lw=1.7, ms=4, alpha=0.65,
                          label=f"Pred — Scene {s_idx + 1}")
            ax_p.legend(fontsize=7, ncol=2, loc="upper left")
        else:
            for s_idx, d in enumerate(scene_data):
                ax_p.plot(sim_hours, d[key_cfg],
                          "o-", color=SCENE_COLORS[s_idx], lw=1.7, ms=4,
                          label=f"Sim — Scene {s_idx + 1}")
            ax_p.legend(fontsize=7, loc="upper left")

        ax_p.axvline(sim_hours[t_peak], ls=":", lw=1.2, c="gray", alpha=0.65)
        ax_p.set_xlabel("Hour of Day", fontsize=8)
        ax_p.set_ylabel(ylabel, fontsize=8)
        ax_p.set_title(f"Diurnal Profile · {ylabel}", fontsize=9, fontweight="bold")
        ax_p.set_xticks(sim_hours)
        ax_p.tick_params(labelsize=7)
        ax_p.grid(True, alpha=0.25)

    _save(fig, "fig7_multivariable_array.png")


# ════════════════════════════════════════════════════════════════
# Figure 8 — Sensing Integration Ablation Study
# ════════════════════════════════════════════════════════════════

def fig8_sensing_ablation():
    """
    Ablation study: model trained without vs with real-time sensor correction.

    Panel A  (top-left)  : Grouped bar chart — R², RMSE, MAE, Category Accuracy
    Panel B  (top-mid)   : Training loss overlay (log scale)
    Panel C  (top-right) : Validation R² convergence
    Panel D  (bottom)    : Per-class UTCI thermal stress accuracy comparison
    """
    print("\n[Fig 8] Sensing Integration Ablation Study...")

    # ── Load eval results ─────────────────────────────────────────
    VARIANT_CFG = [
        ("Without Sensing",    EVAL_JSON,    HISTORY_JSON,    "#2c7bb6"),
        ("With Sensing (v2)",  EVAL_JSON_V2, HISTORY_JSON_V2, "#d7191c"),
    ]

    variants  = []   # list of (label, metrics_dict, color)
    histories = []   # list of (label, hist_dict,    color)  or None entries

    for v_name, eval_path, hist_path, color in VARIANT_CFG:
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                variants.append((v_name, json.load(f), color))
            print(f"  Loaded: {eval_path}")
        else:
            print(f"  [info] not found: {eval_path}")

        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                histories.append((v_name, json.load(f), color))

    if not variants:
        print("  [skip] No eval results found for ablation study")
        return

    n_var = len(variants)

    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        "Sensing Integration Ablation Study\n"
        "Physics-Informed ST-GNN  ·  Without vs With Real-time Sensor Correction",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38)

    # ── Panel A: Grouped metric bars ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])

    METRIC_CFG = [
        ("R2",                "R²",                  None,  True),
        ("RMSE",              "RMSE (°C)",            None,  False),
        ("MAE",               "MAE (°C)",             None,  False),
        ("category_accuracy", "Category Acc.",        100.0, True),
    ]
    n_met = len(METRIC_CFG)
    x_pos = np.arange(n_met)
    width = 0.30

    for v_idx, (v_name, m_dict, color) in enumerate(variants):
        bar_vals = []
        for m_key, _, scale, _ in METRIC_CFG:
            raw = float(m_dict.get(m_key, 0.0))
            bar_vals.append(raw * scale if scale is not None else raw)
        offsets = (np.arange(n_var) - (n_var - 1) / 2.0) * width
        bars = ax_a.bar(
            x_pos + offsets[v_idx], bar_vals,
            width=width * 0.92, label=v_name,
            color=color, alpha=0.82, edgecolor="k", lw=0.5,
        )
        for bar, val in zip(bars, bar_vals):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bar_vals) * 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
            )

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([m[1] for m in METRIC_CFG], fontsize=10)
    ax_a.set_ylabel("Metric Value", fontsize=9)
    ax_a.set_title(
        "Key Performance Metrics Comparison\n"
        "R² & Category Acc ↑  |  RMSE & MAE ↓  (Category Acc scaled ×100)",
        fontsize=9,
    )
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3, axis="y")

    # ── Panel B: Train / Val loss ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    if histories:
        for h_name, hist, color in histories:
            eps = list(range(1, len(hist["train_loss"]) + 1))
            ax_b.semilogy(eps, hist["train_loss"], color=color,
                          lw=1.7, label=f"Train · {h_name}")
            ax_b.semilogy(eps, hist["val_loss"],   color=color,
                          lw=1.7, ls="--", alpha=0.7,
                          label=f"Val · {h_name}")
        best_annots = []
        for h_name, hist, color in histories:
            best_ep = int(np.argmin(hist["val_loss"])) + 1
            best_vl = float(np.min(hist["val_loss"]))
            best_annots.append((best_ep, best_vl, color))
            ax_b.scatter([best_ep], [best_vl], color=color, s=55, zorder=6)
        ax_b.set_xlabel("Epoch"); ax_b.set_ylabel("Loss (log scale)")
        ax_b.set_title("Training Loss Convergence")
        ax_b.legend(fontsize=7); ax_b.grid(True, alpha=0.3)
    else:
        ax_b.text(0.5, 0.5, "Training history\nnot available",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=11, color="#888")
        ax_b.set_title("Training Loss Convergence"); ax_b.axis("off")

    # ── Panel C: Val R² ───────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 3])
    if histories:
        for h_name, hist, color in histories:
            eps    = list(range(1, len(hist["train_loss"]) + 1))
            r2_h   = hist.get("val_r2", [0.0] * len(eps))
            ax_c.plot(eps, r2_h, color=color, lw=1.8, label=h_name)
            best_ep = int(np.argmax(r2_h)) + 1
            best_r2 = float(np.max(r2_h))
            ax_c.scatter([best_ep], [best_r2], color=color, s=60, zorder=6)
            ax_c.annotate(
                f"  R²={best_r2:.4f}",
                xy=(best_ep, best_r2), fontsize=7.5, color=color,
            )
        ax_c.axhline(0.90, ls="--", c="orange", lw=1.3, alpha=0.8,
                     label="Target R²=0.90")
        ax_c.set_xlabel("Epoch"); ax_c.set_ylabel("Val R²")
        ax_c.set_title("Validation R² Convergence")
        ax_c.set_ylim(bottom=max(0.0, min(
            min(hist.get("val_r2", [0]) for _, hist, _ in histories) - 0.05
        )))
        ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, "Training history\nnot available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=11, color="#888")
        ax_c.set_title("Validation R² Convergence"); ax_c.axis("off")

    # ── Panel D: Per-class UTCI accuracy ─────────────────────────
    ax_d = fig.add_subplot(gs[1, :])

    CLASS_LABELS_SHORT = [
        "Extreme Cold\n< 9 °C",
        "No Heat Stress\n9–26 °C",
        "Moderate\n26–32 °C",
        "Strong\n32–38 °C",
        "Very Strong\n38–46 °C",
        "Extreme Heat\n> 46 °C",
    ]
    n_cls  = len(CLASS_LABELS_SHORT)
    x_cls  = np.arange(n_cls)
    width2 = 0.30
    has_per = False

    for v_idx, (v_name, m_dict, color) in enumerate(variants):
        per_cls = m_dict.get("per_category_accuracy", {})
        if not per_cls:
            continue
        has_per = True
        cls_vals = [per_cls.get(lbl, 0.0) * 100
                    for lbl in list(per_cls.keys())[:n_cls]]
        offsets2 = (np.arange(n_var) - (n_var - 1) / 2.0) * width2
        bars2 = ax_d.bar(
            x_cls + offsets2[v_idx], cls_vals,
            width=width2 * 0.92, label=v_name,
            color=color, alpha=0.82, edgecolor="k", lw=0.5,
        )
        for bar, val in zip(bars2, cls_vals):
            ax_d.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    if has_per:
        ax_d.set_xticks(x_cls)
        ax_d.set_xticklabels(CLASS_LABELS_SHORT, fontsize=9)
        ax_d.set_ylabel("Classification Accuracy (%)", fontsize=9)
        ax_d.set_ylim(0, 115)
        ax_d.axhline(90, ls="--", c="#555", lw=1.3, alpha=0.7,
                     label="90 % target line")
        ax_d.set_title(
            "Per-class UTCI Thermal Stress Category Accuracy  "
            "(ISO 15743 / Fiala 2012 Stress Zones)",
            fontsize=10,
        )
        ax_d.legend(fontsize=9)
        ax_d.grid(True, alpha=0.3, axis="y")
    else:
        ax_d.text(
            0.5, 0.5,
            "Per-class accuracy data not available.\n"
            "Run  evaluate.py --out eval_results.json  to generate.",
            ha="center", va="center", transform=ax_d.transAxes,
            fontsize=12, color="#888",
        )
        ax_d.set_title("Per-class UTCI Thermal Stress Category Accuracy")
        ax_d.axis("off")

    _save(fig, "fig8_sensing_ablation.png")


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline Evaluation Visualization")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-scenes", type=int, default=10,
                        help="number of real-world scenarios to show (Fig 5 & 7, default 10)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="skip figures requiring model inference (faster)")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'='*60}")
    print(f"  PIN-ST-GNN Evaluation Visualization  (V2 + Sensing)")
    print(f"  Device : {device}")
    print(f"  Scenes : {args.n_scenes}")
    print(f"  Output : {OUT_DIR}")
    print(f"  V2 ckpt: {'found' if CKPT_PATH_V2.exists() else 'NOT FOUND — will fallback to V1'}")
    print(f"{'='*60}")

    # ── Stage 01 / 02 / 04 static plots ──────────────────────────
    fig1_dataset_overview()
    fig2_graph_features()
    fig3_training_curves()

    # ── Inference-dependent plots (V2 model) ──────────────────────
    model        = None
    dataset_test = None
    epw          = None

    if not args.skip_inference:
        if not _IMPORTS_OK:
            print("\n[warn] Pipeline imports failed — skipping inference figures.")
        elif not H5_PATH.exists():
            print(f"\n[warn] {H5_PATH} not found — skipping inference figures.")
        else:
            print("\n  Loading V2 model and test dataset...")
            epw   = _load_epw()
            model = _load_model(device, prefer_v2=True)   # V2 preferred

            if model is not None and epw is not None:
                dataset_test = UTCIGraphDataset(
                    h5_path      = str(H5_PATH),
                    scenario_pkl = str(SCENARIOS_PKL),
                    epw_pkl      = str(EPW_PKL),
                    split        = "test",
                )
                print(f"  Test dataset: {len(dataset_test)} scenarios")

    fig4_prediction_quality(model, dataset_test, epw, device)
    fig5_spatial_comparison(model, dataset_test, epw, device,
                             n_scenes=args.n_scenes)           # default 10
    fig6_summary_dashboard(model, dataset_test, epw, device)
    fig7_multivariable_array(model, dataset_test, epw, device,
                              n_scenes=min(args.n_scenes, 10))
    fig8_sensing_ablation()

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        from scipy.stats import norm as _sp_norm_test  # noqa: F401
    except ImportError:
        print("[warn] scipy not found — residual curve will be skipped gracefully")

    main()
