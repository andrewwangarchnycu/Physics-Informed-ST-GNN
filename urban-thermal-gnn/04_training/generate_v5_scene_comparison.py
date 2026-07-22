"""
04_training/generate_v5_scene_comparison.py
════════════════════════════════════════════════════════════════
Scene-by-scene comparison for the V5-300 model: real OSM building
geometry (reused unmodified from V4) + real ETH canopy, simulated with
real Radiance + EnergyPlus (Honeybee `utci_comfort_map` recipe) instead
of V1-V4's custom physics approximation, with dynamic edges (shadow/
veg_et/convective) actually populated for the first time. For several
held-out real test scenes, draws four columns:

  (1) real OSM building geometry (height-shaded) + ETH canopy,
  (2) ground-truth UTCI field (real Radiance/EnergyPlus simulation),
  (3) PI-ST-GNN prediction (V5-300 checkpoint),
  (4) signed error (Pred - GT).

This is the direct V5 counterpart of generate_real_scene_comparison.py
(V4), same plotting logic, retargeted at the V5-300 checkpoint and the
real_simulations_v5 dataset.

Usage:
    python generate_v5_scene_comparison.py
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly, Circle, Rectangle
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "02_graph_construction"))
sys.path.insert(0, str(ROOT))

import __main__
from shared import HourlyClimate as _HC, EPWData as _ED
__main__.HourlyClimate = _HC; __main__.EPWData = _ED

from dataset import UTCIGraphDataset
from train import build_model, build_env_time_seq

H5   = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "ground_truth_v5.h5"
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"
EPW  = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "epw_data.pkl"
CKPT = HERE / "checkpoints_v5_300" / "best_model.pt"
HALF = 40.0
# Scenario building/tree lists include a wider neighbourhood-context
# radius beyond the 80x80m design site (they can still cast shadows into
# it); the geometry panel shows this wider context (dashed, faded) with
# the actual site boundary marked, rather than silently clipping it away
# at HALF and making the site look emptier than the physics it was
# simulated with.
CONTEXT_HALF = 65.0


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def _heatmap(ax, pos, vals, cmap, vmin, vmax, half, grid_n=220, mask_dist=6.0):
    """Continuous heatmap via linear interpolation of the sparse sensor
    samples onto a fine regular grid, instead of plotting the raw sample
    points as isolated squares with visible white gaps between them.
    Cells farther than `mask_dist` from any real sample (e.g. inside a
    building footprint, where no sensor exists) are left blank rather
    than extrapolated over, so the heatmap doesn't fabricate values."""
    gx = np.linspace(-half, half, grid_n)
    gy = np.linspace(-half, half, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_vals = griddata(pos, vals, (GX, GY), method="linear")

    tree = cKDTree(pos)
    dist, _ = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    far = dist.reshape(GX.shape) > mask_dist
    grid_vals[far] = np.nan

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    im = ax.imshow(grid_vals, origin="lower", extent=[-half, half, -half, half],
                    cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=2)
    return im


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    dim_air = int(state["air_encoder.net.0.weight"].shape[1])

    ds = UTCIGraphDataset(str(H5), str(SCEN), str(EPW), split="test", dim_air=dim_air)
    epw = pickle.load(open(EPW, "rb"))
    scen = {s["scenario_id"]: s for s in pickle.load(open(SCEN, "rb"))}

    cfg = {"model": {"out_timesteps": len(ds.sim_hours), "dim_air": dim_air}}
    model = build_model(cfg).to(device)
    model.load_state_dict(state); model.eval()

    env_seq, time_seq = build_env_time_seq(epw, ds.sim_hours)
    env_seq = env_seq.to(device); time_seq = time_seq.to(device)
    mean = ds.norm_stats["utci"]["mean"]; std = ds.norm_stats["utci"]["std"]

    # choose 4 test scenes with a healthy building count for a clear comparison
    order = sorted(range(len(ds)), key=lambda i: -len(scen[int(ds._ids[i])]["buildings"]))
    chosen = order[:4]
    t_show = ds.sim_hours.index(14) if 14 in ds.sim_hours else len(ds.sim_hours) // 2

    fig, axes = plt.subplots(len(chosen), 4, figsize=(15, 3.6 * len(chosen)))
    for r, i in enumerate(chosen):
        sid = int(ds._ids[i]); sc = scen[sid]
        data = ds.get(i)
        obj_feat = data["object"].x.to(device)
        air_feat = data["air"].x.to(device)
        target   = data["air"].y.to(device)
        pos      = data["air"].pos.numpy()
        static_e = {}
        for rel in ["semantic", "contiguity"]:
            key = ("object", rel, "object") if rel == "semantic" else ("air", rel, "air")
            if hasattr(data[key], "edge_index"):
                static_e[rel] = data[key].edge_index.to(device)
        dyn = getattr(data, "dynamic_edges", [{}] * air_feat.shape[1])
        with torch.no_grad():
            pred = model(obj_feat, air_feat, dyn, static_e, env_seq, time_seq)
        gt = target[:, t_show].cpu().numpy() * std + mean
        pr = pred[:, t_show].cpu().numpy() * std + mean
        err = pr - gt
        vmin, vmax = min(gt.min(), pr.min()), max(gt.max(), pr.max())

        # col 0: geometry -- site buildings solid, wider shadow-casting
        # context buildings faded+dashed, actual site boundary marked.
        ax = axes[r, 0]
        hs = [b["height"] for b in sc["buildings"]] or [1]
        hmax = max(max(hs), 12)

        def _in_site(fp):
            minx, miny, maxx, maxy = fp.bounds
            return -HALF <= minx and maxx <= HALF and -HALF <= miny and maxy <= HALF

        n_in_site = 0
        for b in sc["buildings"]:
            fp = b["footprint"]
            minx, miny, maxx, maxy = fp.bounds
            if max(abs(minx), abs(maxx), abs(miny), abs(maxy)) > CONTEXT_HALF:
                continue  # far outside even the context window, skip entirely
            in_site = _in_site(fp)
            n_in_site += int(in_site)
            colour = plt.cm.YlOrBr(0.25 + 0.7 * b["height"] / hmax)
            ax.add_patch(MplPoly(_poly_xy(fp), closed=True,
                                 facecolor=colour if in_site else (*colour[:3], 0.25),
                                 edgecolor="#5a3d00" if in_site else "#8a7350",
                                 lw=0.4 if in_site else 0.5,
                                 linestyle="-" if in_site else (0, (3, 2)),
                                 alpha=0.9 if in_site else 0.55,
                                 zorder=2 if in_site else 1))
        for t in sc["trees"]:
            tx, ty = t["pos"]
            if max(abs(tx), abs(ty)) > CONTEXT_HALF:
                continue
            in_site = -HALF <= tx <= HALF and -HALF <= ty <= HALF
            radius = t.get("radius", max(1.5, t["height"] * 0.35))
            ax.add_patch(Circle(t["pos"], radius,
                                 facecolor="#2e8b57" if in_site else "#2e8b57",
                                 edgecolor="#1c5c37", linewidth=0.7,
                                 alpha=0.85 if in_site else 0.35,
                                 zorder=4 if in_site else 1))
        # dashed outline marking the actual 80x80m design-site boundary
        ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, facecolor="none",
                               edgecolor="#1a1a1a", linewidth=1.1, linestyle=(0, (5, 3)), zorder=5))

        ax.set_xlim(-CONTEXT_HALF, CONTEXT_HALF); ax.set_ylim(-CONTEXT_HALF, CONTEXT_HALF); ax.set_aspect("equal")
        ax.set_ylabel(f"場景 #{sid}\n({sc['assigned_month']}月，基地內{n_in_site}棟／周邊脈絡{len(sc['buildings'])}棟)", fontsize=8.5)
        if r == 0:
            ax.set_title("真實 OSM 幾何\n（虛線框＝基地邊界，淡色＝周邊遮蔭脈絡）", fontsize=9.5)

        for c, (vals, title, cmap, vr) in enumerate([
                (gt, "真值 UTCI（Radiance／EnergyPlus）", "RdYlBu_r", (vmin, vmax)),
                (pr, "PI-ST-GNN 預測（V5-300）", "RdYlBu_r", (vmin, vmax)),
                (err, "誤差 (預測-真值)", "RdBu_r", None)], start=1):
            ax = axes[r, c]
            if vr is None:
                vm = max(abs(err.min()), abs(err.max()), 0.5)
                im = _heatmap(ax, pos, vals, cmap, -vm, vm, HALF)
            else:
                im = _heatmap(ax, pos, vals, cmap, vr[0], vr[1], HALF)
            for b in sc["buildings"]:
                ax.add_patch(MplPoly(_poly_xy(b["footprint"]), closed=True,
                                     facecolor="none", edgecolor="black", lw=0.5, alpha=0.7, zorder=3))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         label=("ΔT (°C)" if vr is None else "UTCI (°C)"))
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

    fig.suptitle(f"以真實 Radiance／EnergyPlus 引擎訓練之 PI-ST-GNN（V5-300）：未見過真實測試場景之預測比較（{ds.sim_hours[t_show]}:00）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = HERE / "figures" / "fig_v5_scene_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out} (scenes {[int(ds._ids[i]) for i in chosen]})")


if __name__ == "__main__":
    main()
