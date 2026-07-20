"""
04_training/generate_real_scene_comparison.py
════════════════════════════════════════════════════════════════
Scene-by-scene comparison for the V4 model trained on REAL geographic
environments. For several held-out real test scenes, draws four columns:

  (1) real OSM building geometry (height-shaded) + ETH canopy,
  (2) ground-truth UTCI field (physics simulation on real geometry+IoT),
  (3) PI-ST-GNN prediction (model trained on real scenes),
  (4) signed error (Pred - GT).

This shows the model reproduces the real-geometry-driven spatial heat-stress
structure (building shadows, vegetation cooling) on scenes it never saw
during training, at peak heat hour.

Usage:
    python generate_real_scene_comparison.py
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly

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

H5   = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "ground_truth_v4.h5"
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"
EPW  = ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "epw_data.pkl"
CKPT = HERE / "checkpoints_v4" / "best_model.pt"
HALF = 40.0


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device)
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

        # col 0: geometry
        ax = axes[r, 0]
        hs = [b["height"] for b in sc["buildings"]] or [1]
        hmax = max(max(hs), 12)
        for b in sc["buildings"]:
            ax.add_patch(MplPoly(_poly_xy(b["footprint"]), closed=True,
                                 facecolor=plt.cm.YlOrBr(0.25 + 0.7 * b["height"] / hmax),
                                 edgecolor="#5a3d00", lw=0.4, alpha=0.9))
        for t in sc["trees"][:40]:
            ax.scatter(*t["pos"], s=max(3, t["height"] * 1.4), c="#2e8b57", alpha=0.5, edgecolor="none")
        ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
        ax.set_ylabel(f"場景 #{sid}\n({sc['assigned_month']}月, {len(sc['buildings'])}棟)", fontsize=9)
        if r == 0:
            ax.set_title("真實 OSM 幾何", fontsize=10)

        for c, (vals, title, cmap, vr) in enumerate([
                (gt, "真值 UTCI（物理模擬）", "inferno", (vmin, vmax)),
                (pr, "PI-ST-GNN 預測", "inferno", (vmin, vmax)),
                (err, "誤差 (預測-真值)", "RdBu_r", None)], start=1):
            ax = axes[r, c]
            if vr is None:
                vm = max(abs(err.min()), abs(err.max()), 0.5)
                s = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap, vmin=-vm, vmax=vm, s=8, marker="s")
            else:
                s = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap, vmin=vr[0], vmax=vr[1], s=8, marker="s")
            fig.colorbar(s, ax=ax, fraction=0.046, pad=0.02,
                         label=("ΔT (°C)" if vr is None else "UTCI (°C)"))
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

    fig.suptitle(f"以真實地理環境訓練之 PI-ST-GNN：未見過真實測試場景之預測比較（{ds.sim_hours[t_show]}:00）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = HERE / "figures" / "fig_real_scene_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out} (scenes {[int(ds._ids[i]) for i in chosen]})")


if __name__ == "__main__":
    main()
