"""
04_training/generate_graph_topology_figure.py
════════════════════════════════════════════════════════════════
異質圖拓樸全域視角（重繪版，統一為圖 19 之風格與圖幅尺寸）。

沿用與 generate_graph_construction_figure.py 相同之真實資料來源
（ground_truth_v2.h5 + scenarios.pkl，場景 10），忠實依 dataset.py 之
實際定義繪製：
  * 建物節點（object nodes）：僅 scenario["buildings"]，N_obj 與
    dataset.py 之 _extract_object_features 完全一致；喬木僅作為空氣
    節點特徵（tree_h_norm）之來源，本身非圖節點，故圖中以純視覺化
    喬木冠層圓形疊繪，不納入語義邊連接對象。
  * (object,semantic,object)：N_obj 個建物節點間之全連接邊。
  * (air,contiguity,air)：KNN(K=8) 毗鄰邊，經 cKDTree 實際計算。

Panel (a): 全域 80×80 公尺場景拓樸總覽。
Panel (b): 場域左下 40×40 公尺象限放大，標示邊緣類型與 9 維空氣節點
           特徵定義。

Usage:
    python generate_graph_topology_figure.py [--sid 10]
"""
from __future__ import annotations

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
H5_PATH = ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "ground_truth_v2.h5"
SCEN_PATH = ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "scenarios.pkl"
KNN_K = 8


def main(sid: int):
    scen = pickle.load(open(SCEN_PATH, "rb"))
    smap = {int(s.get("scenario_id", i)): s for i, s in enumerate(scen)}
    with h5py.File(H5_PATH, "r") as hf:
        grp = hf[f"scenarios/{sid}"]
        sensor_pts = grp["sensor_pts"][()]  # (N,2)
    scenario = smap[sid]
    buildings = scenario["buildings"]
    trees = scenario.get("trees", [])
    N_air = sensor_pts.shape[0]
    N_obj = len(buildings)
    centroids = np.array([b["centroid"] for b in buildings])

    # 真實 KNN(K=8) 毗鄰邊
    k = min(KNN_K, max(N_air - 1, 1))
    tree_idx = cKDTree(sensor_pts)
    _, knn_idx = tree_idx.query(sensor_pts, k=k + 1)
    contiguity_edges = [(i, int(j)) for i, nbrs in enumerate(knn_idx) for j in nbrs[1:]]
    semantic_edges = [(i, j) for i in range(N_obj) for j in range(i + 1, N_obj)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.6))
    half = 40.0

    for ax in (axA, axB):
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # ── Panel (a)：全域拓樸總覽 ──────────────────────────────────
    axA.add_patch(MplPoly([(0, 0), (80, 0), (80, 80), (0, 80)], closed=True,
                          fill=False, edgecolor="#333", lw=1.8, ls="--"))
    heights = [b["height"] for b in buildings]
    hmax = max(heights) if heights else 1.0
    cmap = plt.cm.YlOrBr
    for b in buildings:
        fp = b["footprint"]
        xy = list(fp.exterior.coords)
        col = cmap(0.25 + 0.7 * b["height"] / hmax)
        axA.add_patch(MplPoly(xy, closed=True, facecolor=col, edgecolor="#5a3d00", lw=0.9, zorder=3))
    for t in trees:
        x, y = t["pos"]
        axA.scatter([x], [y], s=max(30, t["height"] * 20), c="#2e8b57",
                    alpha=0.45, edgecolor="#14532d", linewidth=0.4, zorder=2)

    # KNN 毗鄰邊：僅示意顯示部分（全量繪製於此密度下不可讀）
    rng = np.random.default_rng(42)
    show_n = min(3000, len(contiguity_edges))
    show_idx = rng.choice(len(contiguity_edges), size=show_n, replace=False)
    segs = [[sensor_pts[i], sensor_pts[j]] for k_ in show_idx for i, j in [contiguity_edges[k_]]]
    axA.add_collection(LineCollection(segs, colors="#d9822b", linewidths=0.25, alpha=0.35, zorder=1))

    axA.scatter(sensor_pts[:, 0], sensor_pts[:, 1], c="#f0a04b", s=2.5, alpha=0.6, zorder=1, edgecolor="none")
    for i, j in semantic_edges:
        axA.plot([centroids[i, 0], centroids[j, 0]], [centroids[i, 1], centroids[j, 1]],
                 color="#7b3fa0", lw=1.3, ls="--", alpha=0.85, zorder=4)
    axA.scatter(centroids[:, 0], centroids[:, 1], s=70, facecolor="white",
               edgecolor="#7b3fa0", linewidth=1.6, zorder=5)

    axA.set_xlim(-4, 84); axA.set_ylim(-4, 84)
    axA.set_title(f"(a) 場景 #{sid} 完整異質圖拓樸\n"
                  f"空氣節點 N_air={N_air}，毗鄰邊 KNN K={k}"
                  f"（示意顯示 {show_n:,}／{len(contiguity_edges):,} 條）", fontsize=10)
    legend_a = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f0a04b",
              markersize=7, label="空氣節點（Air Node）"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=cmap(0.6),
              markeredgecolor="#5a3d00", markersize=10, label="建物節點（Object Node）"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2e8b57",
              alpha=0.5, markersize=10, label="喬木冠層（僅視覺化，非圖節點）"),
        Line2D([0], [0], color="#d9822b", lw=1.2, alpha=0.6, label="毗鄰邊（Contiguity Edge，KNN）"),
        Line2D([0], [0], color="#7b3fa0", lw=1.4, ls="--", label="語義邊（Semantic Edge，全連接）"),
    ]
    axA.legend(handles=legend_a, loc="lower left", fontsize=7.3, framealpha=0.9)

    # ── Panel (b)：左下 40×40 公尺象限放大 ──────────────────────
    in_win = (sensor_pts[:, 0] <= half) & (sensor_pts[:, 1] <= half)
    win_ids = set(np.where(in_win)[0].tolist())
    win_edges = [(i, j) for (i, j) in contiguity_edges if i in win_ids and j in win_ids]
    segs_b = [[sensor_pts[i], sensor_pts[j]] for i, j in win_edges]
    axB.add_collection(LineCollection(segs_b, colors="#bbbbbb", linewidths=0.4, alpha=0.55, zorder=1))
    axB.scatter(sensor_pts[in_win, 0], sensor_pts[in_win, 1], c="#f0a04b", s=8,
               zorder=3, edgecolor="none")

    for b in buildings:
        cx, cy = b["centroid"]
        if cx <= half + 10 and cy <= half + 10:
            fp = b["footprint"]
            xy = list(fp.exterior.coords)
            col = cmap(0.25 + 0.7 * b["height"] / hmax)
            axB.add_patch(MplPoly(xy, closed=True, facecolor=col, edgecolor="#5a3d00", lw=0.9, zorder=4))
    for i, j in semantic_edges:
        ci, cj = centroids[i], centroids[j]
        if min(ci[0], cj[0]) <= half + 10 and min(ci[1], cj[1]) <= half + 10:
            axB.plot([ci[0], cj[0]], [ci[1], cj[1]], color="#7b3fa0", lw=1.3,
                     ls="--", alpha=0.85, zorder=5)
    for c in centroids:
        if c[0] <= half + 10 and c[1] <= half + 10:
            axB.scatter(*c, s=70, facecolor="white", edgecolor="#7b3fa0", linewidth=1.6, zorder=6)

    axB.set_xlim(0, half); axB.set_ylim(0, half)
    axB.set_title(f"(b) 局部放大：邊緣類型與節點特徵\n（場域左下 {half:.0f}×{half:.0f} 公尺象限）", fontsize=10)
    legend_b = [
        Line2D([0], [0], color="#bbbbbb", lw=1.4, alpha=0.7, label="毗鄰邊（Contiguity Edge，KNN）"),
        Line2D([0], [0], color="#7b3fa0", lw=1.4, ls="--", label="語義邊（Semantic Edge，全連接）"),
    ]
    axB.legend(handles=legend_b, loc="upper left", fontsize=7.5, framealpha=0.9)

    feature_text = (
        "空氣節點特徵（9 維）\n"
        "－－－－－－－－－－－－－－－\n"
        "[0] 氣溫正規化（Ta norm）\n"
        "[1] 平均輻射溫度正規化（MRT norm）\n"
        "[2] 風速正規化（va norm）\n"
        "[3] 相對濕度正規化（RH norm）\n"
        "[4] 天空可見率（SVF）\n"
        "[5] 陰影旗標（Shadow）\n"
        "[6] 建物高度正規化（Bh/50）\n"
        "[7] 樹木高度正規化（Th/12）\n"
        "[8] 地表溫度正規化（Ts norm，V2）"
    )
    axB.text(1.03, 0.98, feature_text, transform=axB.transAxes,
             fontsize=8.2, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f4ef", edgecolor="gray"))

    fig.suptitle("異質圖拓樸全域視角（真實訓練資料場景 10）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    out = HERE / "figures" / "fig_geo5_graph_topology.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[fig] saved {out} (scene #{sid}, N_air={N_air}, N_obj={N_obj}, trees={len(trees)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, default=10)
    args = ap.parse_args()
    main(args.sid)

