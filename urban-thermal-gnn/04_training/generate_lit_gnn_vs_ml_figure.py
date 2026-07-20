"""
generate_lit_gnn_vs_ml_figure.py
================================
Original schematic diagram (not copied from any source paper) for
Thesis_GIA chapter 2, subsec:gnn_paradigm: contrasts the Euclidean
data representation used by traditional ML (voxel/pixel-grid CNN, or
flat-feature-vector random forest) against the heterogeneous graph
representation used by PI-ST-GNN (building / air / tree nodes connected
by shading, convective, and evapotranspiration edges).

Produces:
  figures/fig_lit_gnn_vs_ml.pdf
  figures/fig_lit_gnn_vs_ml.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_VOXEL  = "#8a8a8a"
C_EMPTY  = "#e0ded9"
C_BLDG   = "#4a4a4a"
C_AIR    = "#3f7fa6"
C_TREE   = "#7a9e6a"
C_SHADE  = "#c9704f"
C_CONV   = "#3f7fa6"
C_ET     = "#7a9e6a"
C_SEM    = "#8a6fa3"

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

# ── (a) Euclidean: regularised voxel/pixel grid, mostly empty ──
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("(a) 歐幾里得表徵\n（規則體素／像素網格 CNN、扁平特徵向量隨機森林）", fontsize=10, fontweight="bold")

n = 9
cell = 8.0 / n
x0, y0 = 0.7, 0.4
occupied = {(2, 2), (2, 3), (3, 2), (3, 3), (6, 5), (6, 6), (7, 5)}
for i in range(n):
    for j in range(n):
        cx, cy = x0 + i * cell, y0 + j * cell
        color = C_BLDG if (i, j) in occupied else C_EMPTY
        r = Rectangle((cx, cy), cell * 0.94, cell * 0.94, facecolor=color,
                       edgecolor="#aaaaaa", linewidth=0.5)
        ax.add_patch(r)

ax.text(x0 + n * cell / 2, y0 - 0.35,
        "規則網格：多數體素為「空節點」\n須極度細化才能捕捉建築縮退等細部幾何", ha="center", fontsize=8.4)

legend_a = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BLDG, markersize=11, label="建築體素／像素"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_EMPTY, markersize=11, label="空體素（無意義計算）"),
]
ax.legend(handles=legend_a, loc="lower center", bbox_to_anchor=(0.5, -0.2),
          ncol=1, fontsize=8.3, frameon=False)

# ── (b) Non-Euclidean: heterogeneous graph with typed edges ──
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("(b) 非歐幾里得表徵\n（異質圖：建物／空氣／喬木節點 + 因果類型邊）", fontsize=10, fontweight="bold")

bldg_nodes = [(1.8, 6.5), (2.0, 3.5)]
air_nodes  = [(5.0, 7.2), (6.5, 6.0), (7.8, 4.8), (5.5, 3.8), (7.0, 2.5), (4.2, 5.0)]
tree_nodes = [(2.6, 1.5)]

for (x, y) in bldg_nodes:
    ax.add_patch(Circle((x, y), 0.32, facecolor=C_BLDG, edgecolor="black", linewidth=1.0, zorder=4))
for (x, y) in air_nodes:
    ax.add_patch(Circle((x, y), 0.22, facecolor=C_AIR, edgecolor="black", linewidth=0.8, zorder=4))
for (x, y) in tree_nodes:
    ax.add_patch(Circle((x, y), 0.26, facecolor=C_TREE, edgecolor="black", linewidth=1.0, zorder=4))

# shading edges (building -> nearby air nodes)
shade_pairs = [(0, 0), (0, 5), (1, 3), (1, 5)]
for bi, ai in shade_pairs:
    a = FancyArrowPatch(bldg_nodes[bi], air_nodes[ai], arrowstyle="-|>", mutation_scale=9,
                         linewidth=1.1, color=C_SHADE, alpha=0.85)
    ax.add_patch(a)

# convective edges (air -> air, along a wind direction)
conv_pairs = [(0, 1), (1, 2), (5, 3), (3, 4)]
for i, j in conv_pairs:
    a = FancyArrowPatch(air_nodes[i], air_nodes[j], arrowstyle="-|>", mutation_scale=9,
                         linewidth=1.1, color=C_CONV, linestyle="--", alpha=0.85)
    ax.add_patch(a)

# evapotranspiration edges (tree -> nearby air nodes)
et_pairs = [3, 5]
for ai in et_pairs:
    a = FancyArrowPatch(tree_nodes[0], air_nodes[ai], arrowstyle="-|>", mutation_scale=9,
                         linewidth=1.1, color=C_ET, linestyle=":", alpha=0.9)
    ax.add_patch(a)

# semantic edge (building -> building, fully connected)
a = FancyArrowPatch(bldg_nodes[0], bldg_nodes[1], arrowstyle="<|-|>", mutation_scale=9,
                     linewidth=1.0, color=C_SEM, linestyle="-.", alpha=0.8)
ax.add_patch(a)

ax.text(1.9, 7.2, "建物節點", ha="center", fontsize=8, color=C_BLDG, fontweight="bold")
ax.text(2.6, 0.9, "喬木節點", ha="center", fontsize=8, color=C_TREE, fontweight="bold")
ax.text(5.0, 7.9, "空氣節點", ha="center", fontsize=8, color=C_AIR, fontweight="bold")

legend_b = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BLDG, markersize=10, label="建物節點"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AIR, markersize=9, label="空氣節點"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TREE, markersize=10, label="喬木節點"),
    Line2D([0], [0], color=C_SHADE, lw=1.6, label="遮蔽邊 shadow"),
    Line2D([0], [0], color=C_CONV, lw=1.6, linestyle="--", label="對流邊 convective"),
    Line2D([0], [0], color=C_ET, lw=1.6, linestyle=":", label="蒸散邊 veg_et"),
    Line2D([0], [0], color=C_SEM, lw=1.4, linestyle="-.", label="語義邊 semantic"),
]
ax.legend(handles=legend_b, loc="lower center", bbox_to_anchor=(0.5, -0.24),
          ncol=3, fontsize=7.8, frameon=False)

fig.suptitle("異質圖神經網路與傳統歐幾里得機器學習於都市微氣候建模之表徵對比", fontsize=12, fontweight="bold", y=1.03)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_lit_gnn_vs_ml.pdf"
out_png = FIG_DIR / "fig_lit_gnn_vs_ml.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
