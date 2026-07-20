"""
04_training/generate_real_scene_graph_figure.py
════════════════════════════════════════════════════════════════
Figure: how one V4 real-geometry scene becomes a heterogeneous graph.

Draws, for a representative real site, the genuine GIS ingredients and the
graph built on them:
  (a) real OSM building footprints (shaded by height) + real ETH canopy
      trees (green, sized by canopy height) within the 80x80 m scene;
  (b) the air-node sensor grid coloured by simulated peak-hour UTCI, with
      the k-NN contiguity edges drawn — i.e. the actual graph the PI-ST-GNN
      consumes, grounded entirely in real geospatial data.

Usage:
    python generate_real_scene_graph_figure.py [--sid 0]
"""
from __future__ import annotations

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import LineCollection

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"
SIMDIR = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4"


def knn_edges(pts, k=8):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k + 1, len(pts)))
    segs = []
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            segs.append([pts[i], pts[j]])
    return segs


def pick_scene(scenarios, sid):
    by_id = {s["scenario_id"]: s for s in scenarios}
    if sid is not None and sid in by_id:
        return by_id[sid]
    # else pick a scene with a healthy mix of buildings + trees
    best = max(scenarios, key=lambda s: min(len(s["buildings"]), 8) + min(s.get("n_trees_used", 0), 20))
    return best


def main(sid):
    scenarios = pickle.load(open(SCEN, "rb"))
    sc = pick_scene(scenarios, sid)
    sid = sc["scenario_id"]
    npz = SIMDIR / f"sim_{sid:04d}.npz"
    d = np.load(npz) if npz.exists() else None

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.6))
    half = 40.0

    for ax in (axA, axB):
        ax.add_patch(MplPoly([(-half, -half), (half, -half), (half, half), (-half, half)],
                             closed=True, fill=False, edgecolor="#333", lw=1.8, ls="--"))
        ax.set_xlim(-half - 8, half + 8); ax.set_ylim(-half - 8, half + 8)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # ── Panel (a): real geometry ────────────────────────────────
    heights = [b["height"] for b in sc["buildings"]]
    hmax = max(heights) if heights else 1.0
    cmap = plt.cm.YlOrBr
    for b in sc["buildings"]:
        fp = b["footprint"]
        xy = list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)
        col = cmap(0.25 + 0.7 * b["height"] / hmax)
        tag = b.get("height_from_tag")
        axA.add_patch(MplPoly(xy, closed=True, facecolor=col,
                              edgecolor="#5a3d00" if tag else "#888",
                              lw=1.1 if tag else 0.6, alpha=0.9))
    for t in sc["trees"]:
        x, y = t["pos"]
        axA.scatter([x], [y], s=max(8, t["height"] * 4), c="#2e8b57",
                    alpha=0.55, edgecolor="#14532d", linewidth=0.3, zorder=4)
    axA.set_title(f"(a) 場景 #{sid} 真實幾何\n"
                  f"OSM 建物 {len(sc['buildings'])} 棟（色階=簷高，實線框=真實樓高標籤）"
                  f"＋ETH 樹冠 {len(sc['trees'])} 株", fontsize=10)
    # height colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, hmax))
    cb = fig.colorbar(sm, ax=axA, fraction=0.046, pad=0.02); cb.set_label("建物簷高 (m)", fontsize=9)

    # ── Panel (b): the graph ────────────────────────────────────
    if d is not None:
        pts = d["sensor_pts"]
        utci_peak = np.nanmax(d["utci"], axis=0)   # per-node peak UTCI
        segs = knn_edges(pts, k=8)
        axB.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.3, alpha=0.5, zorder=1))
        scat = axB.scatter(pts[:, 0], pts[:, 1], c=utci_peak, cmap="inferno",
                           s=14, zorder=3, edgecolor="none")
        cb2 = fig.colorbar(scat, ax=axB, fraction=0.046, pad=0.02)
        cb2.set_label("節點尖峰 UTCI (°C)", fontsize=9)
        # overlay building outlines faintly for context
        for b in sc["buildings"]:
            fp = b["footprint"]
            xy = list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)
            axB.add_patch(MplPoly(xy, closed=True, facecolor="#dddddd",
                                  edgecolor="#999", lw=0.4, alpha=0.5, zorder=2))
        axB.set_title(f"(b) 空氣節點異質圖：{pts.shape[0]} 個感測節點 + k-NN 鄰接邊\n"
                      f"節點色=模擬尖峰 UTCI（真實 IoT 氣溫驅動）", fontsize=10)
    else:
        axB.text(0, 0, "（尚無模擬輸出 npz）", ha="center", fontsize=12)

    fig.suptitle("以真實 GIS 圖資建構之異質圖（V4 真實場景）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / "figures" / "fig_real_scene_graph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[fig] saved {out} (scene #{sid}, {len(sc['buildings'])} buildings, {len(sc['trees'])} trees)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, default=None)
    args = ap.parse_args()
    main(args.sid)
