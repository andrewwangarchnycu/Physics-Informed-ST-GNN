"""
generate_street_crosssection_figure.py
================================
Street/urban cross-section schematic requested by the thesis committee for
Thesis_GIA chapter 3 (sec:framework / sec:graph_construction): a side-view
diagram showing solar angle, building shadow, tree shade, and pedestrian
UTCI thermal-stress geometry together with the multi-height air-node
sampling used by the heterogeneous graph.

Produces:
  figures/fig_street_crosssection.pdf
  figures/fig_street_crosssection.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Wedge
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

C_BLDG   = "#4a4a4a"
C_TREE   = "#5f8a4f"
C_SUN    = "#d99a2b"
C_SHADOW = "#2b3a55"
C_AIR    = "#3f7fa6"
C_HOT    = "#b3401f"
C_COOL   = "#5f8a4f"
C_GROUND = "#8a8a8a"

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(-2, 32)
ax.set_ylim(-1.5, 16)
ax.axis("off")

# ── ground ──
ax.add_patch(Rectangle((-2, -1.5), 34, 1.5, facecolor=C_GROUND, edgecolor="none", alpha=0.35))
ax.plot([-2, 32], [0, 0], color="#555555", linewidth=1.2)

# ── two buildings framing a street canyon ──
b1 = Rectangle((0, 0), 6, 14, facecolor=C_BLDG, edgecolor="black", linewidth=1.1)
b2 = Rectangle((24, 0), 6, 9, facecolor=C_BLDG, edgecolor="black", linewidth=1.1)
ax.add_patch(b1); ax.add_patch(b2)
ax.text(3, 14.5, "建築 A（高層）", ha="center", fontsize=9, fontweight="bold")
ax.text(27, 9.5, "建築 B（低層）", ha="center", fontsize=9, fontweight="bold")

# ── street tree with canopy ──
tree_x, tree_trunk_h, canopy_r = 14.0, 2.2, 2.3
ax.plot([tree_x, tree_x], [0, tree_trunk_h], color="#6b4a2f", linewidth=4, solid_capstyle="round")
canopy_cy = tree_trunk_h + canopy_r * 0.75
ax.add_patch(Circle((tree_x, canopy_cy), canopy_r, facecolor=C_TREE, edgecolor="#3f5c33", alpha=0.85, linewidth=1.0))
ax.text(tree_x, canopy_cy + canopy_r + 0.6, "喬木冠層\n（樹蔭）", ha="center", fontsize=8.5, color="#3f5c33", fontweight="bold")

# ── sun and incoming solar rays at a low afternoon angle ──
sun_x, sun_y = 2.5, 15.3
ax.add_patch(Circle((sun_x, sun_y), 0.55, facecolor=C_SUN, edgecolor="#a87418", linewidth=1.0, zorder=6))
ax.text(sun_x, sun_y + 0.9, "太陽", ha="center", fontsize=8.5, color="#a87418", fontweight="bold")

sun_angle_deg = 35  # elevation angle above horizon, illustrative
ray_dx, ray_dy = np.cos(np.radians(180 - sun_angle_deg)), -np.sin(np.radians(sun_angle_deg))
for start_x in [9, 15, 21, 29]:
    x0, y0 = start_x, 16.0
    length = 20
    x1, y1 = x0 + ray_dx * length, y0 + ray_dy * length
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                  linewidth=1.1, color=C_SUN, alpha=0.75, linestyle="--"))
ax.annotate("", xy=(6, 8), xytext=(6, 14),
            arrowprops=dict(arrowstyle="-", color="none"))
ax.text(6.6, 12.3, r"太陽仰角 $\theta_{sun}$", fontsize=8.5, color="#a87418", fontweight="bold", rotation=-35)

# ── building A cast shadow on the ground (dark band to the right of A) ──
shadow_end = 6 + 14.0 / np.tan(np.radians(sun_angle_deg))
shadow_end = min(shadow_end, 24)
ax.add_patch(Rectangle((6, 0), shadow_end - 6, 0.35, facecolor=C_SHADOW, edgecolor="none", alpha=0.75, zorder=2))
ax.annotate("建築 A 陰影範圍", xy=((6 + shadow_end) / 2, 0.35), xytext=((6 + shadow_end) / 2, 1.6),
            ha="center", fontsize=8.3, color=C_SHADOW, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_SHADOW, lw=1.0))

# tree shade footprint on the ground
ax.add_patch(Rectangle((tree_x - canopy_r * 0.9, 0), canopy_r * 1.8, 0.35, facecolor=C_TREE, alpha=0.7, edgecolor="none", zorder=2))

# ── pedestrian silhouette at street level, with UTCI thermal-stress marker ──
ped_x = 19.0
ax.add_patch(Circle((ped_x, 1.55), 0.35, facecolor="#d9b48f", edgecolor="black", linewidth=0.8, zorder=6))
ax.add_patch(Rectangle((ped_x - 0.28, 0.15), 0.56, 1.25, facecolor="#5b7fa6", edgecolor="black", linewidth=0.8, zorder=6))
ax.text(ped_x, -0.9, "行人", ha="center", fontsize=8.5)
ax.annotate("高熱壓力\nUTCI > 38°C\n（曝曬於街谷中段）", xy=(ped_x, 2.0), xytext=(ped_x + 3.0, 4.5),
            fontsize=8.3, color=C_HOT, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-|>", color=C_HOT, lw=1.2))

ped2_x = 13.3
ax.add_patch(Circle((ped2_x, 1.55), 0.35, facecolor="#d9b48f", edgecolor="black", linewidth=0.8, zorder=6))
ax.add_patch(Rectangle((ped2_x - 0.28, 0.15), 0.56, 1.25, facecolor="#5b7fa6", edgecolor="black", linewidth=0.8, zorder=6))
ax.annotate("低熱壓力\nUTCI < 32°C\n（樹蔭遮蔽）", xy=(ped2_x, 2.0), xytext=(ped2_x - 4.6, 5.4),
            fontsize=8.3, color=C_COOL, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-|>", color=C_COOL, lw=1.2))

# ── multi-height air-node sampling column (heterogeneous graph air nodes) ──
air_x = 19.0
air_heights = [0.5, 3.0, 6.0, 9.0, 12.0]
for h in air_heights:
    ax.add_patch(Circle((air_x + 4.5, h), 0.28, facecolor=C_AIR, edgecolor="black", linewidth=0.8, zorder=5))
ax.plot([air_x + 4.5] * 2, [0.5, 12.0], color=C_AIR, linewidth=1.0, linestyle=":", zorder=4, alpha=0.7)
ax.text(air_x + 4.5, 13.0, "空氣節點\n（逐高度取樣）", ha="center", fontsize=8.3, color=C_AIR, fontweight="bold")

# ── street canyon aspect ratio annotation H/W ──
ax.annotate("", xy=(0, -0.9), xytext=(24, -0.9),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1))
ax.text(12, -1.3, r"街谷寬度 $W$", ha="center", fontsize=8.5)
ax.annotate("", xy=(-0.9, 0), xytext=(-0.9, 14),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1))
ax.text(-1.5, 7, r"$H$", ha="center", fontsize=8.5, rotation=90)

legend_elems = [
    Line2D([0], [0], color=C_SUN, lw=1.6, linestyle="--", label="太陽短波輻射入射角"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_SHADOW, alpha=0.75, markersize=11, label="建築陰影範圍"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TREE, markersize=11, label="喬木樹蔭範圍"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_AIR, markeredgecolor="black", markersize=9, label="空氣節點（多高度取樣）"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HOT, markersize=9, label="高熱壓力行人位置"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_COOL, markersize=9, label="低熱壓力行人位置"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, fontsize=8.2, frameon=False)

fig.suptitle("都市街道剖面示意：太陽角度、建築陰影、喬木樹蔭與行人熱壓力之幾何關係", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_street_crosssection.pdf"
out_png = FIG_DIR / "fig_street_crosssection.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
