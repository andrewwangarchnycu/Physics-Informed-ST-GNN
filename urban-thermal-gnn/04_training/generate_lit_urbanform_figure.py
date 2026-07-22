"""
generate_lit_urbanform_figure.py
================================
Original schematic diagram (not copied from any source paper) illustrating
the classic urban-canyon geometry / thermal / aerodynamic mechanisms
reviewed in Thesis_GIA chapter 2, subsec:urban_form_microclimate:
  - Sky View Factor (SVF) shortwave shading at the canyon opening
  - nocturnal longwave "radiation trapping" inside a narrow canyon
  - aspect-ratio (H/W) driven downwash on the windward face and a
    wake-recirculation vortex on the leeward face

Produces:
  figures/fig_lit_urbanform.pdf
  figures/fig_lit_urbanform.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle
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
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# House style (matches fig_overview_pistgnn_white.png): grayscale
# throughout, black-outlined geometry, simple solid/dashed/dotted lines.
# Colour is reserved only for sunlight/shortwave (the sole exception the
# committee asked to keep) -- every other mechanism (longwave, wind,
# wake vortex) is differentiated by line style/shade of gray, not hue.
C_BLDG   = "#5a5a5a"
C_SUN    = "#c9832f"
C_LW     = "#4d4d4d"
C_WIND   = "#1a1a1a"
C_VORTEX = "#8f8f8f"
TEXT_MAIN = "#000000"

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# ── (a) SVF shortwave shading + nocturnal longwave radiation trapping ──
ax = axes[0]
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 8)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.3); spine.set_color("black")
ax.set_title("(a) 天空視角因子（SVF）短波遮擋與\n夜間長波輻射多重反射", fontsize=10.5, fontweight="bold")

ground = Rectangle((-1, -0.5), 12, 0.5, facecolor="#8a8a8a", edgecolor="none")
ax.add_patch(ground)
b1 = Rectangle((0.5, 0), 2.2, 6.5, facecolor=C_BLDG, edgecolor="black", linewidth=1.0)
b2 = Rectangle((7.3, 0), 2.2, 5.0, facecolor=C_BLDG, edgecolor="black", linewidth=1.0)
ax.add_patch(b1); ax.add_patch(b2)
ax.text(1.6, 6.9, "建築 A", ha="center", fontsize=8.5)
ax.text(8.4, 5.4, "建築 B", ha="center", fontsize=8.5)

# canyon observer point
obs_x, obs_y = 5.0, 0.05
ax.plot(obs_x, obs_y, "o", color="black", markersize=5, zorder=5)
ax.text(obs_x, -0.35, "觀測點", ha="center", fontsize=8)

# SVF exposed solid angle (wedge between building tops as seen from observer)
theta1 = np.degrees(np.arctan2(6.5 - obs_y, 0.5 + 2.2 - obs_x))
theta2 = np.degrees(np.arctan2(5.0 - obs_y, 7.3 - obs_x))
wedge = Wedge((obs_x, obs_y), 6.5, theta1, theta2, facecolor=C_SUN, alpha=0.28, edgecolor=C_SUN, linewidth=1.2)
ax.add_patch(wedge)
ax.text(obs_x, 3.4, "SVF\n(暴露立體角)", ha="center", fontsize=8.3, color=TEXT_MAIN, fontweight="bold")

# incoming shortwave
arrow1 = FancyArrowPatch((3.6, 7.6), (obs_x + 0.3, obs_y + 0.4), arrowstyle="-|>",
                          mutation_scale=13, linewidth=1.4, color=C_SUN)
ax.add_patch(arrow1)
ax.text(3.0, 7.6, "太陽短波輻射", fontsize=8, color=TEXT_MAIN, fontweight="bold")

# nocturnal longwave bouncing between the two facades
for k, (x0, y0, x1, y1) in enumerate([
    (2.7, 2.0, 7.3, 2.6), (7.3, 2.6, 2.7, 3.2),
    (2.7, 3.2, 7.3, 3.8), (7.3, 3.8, 2.7, 4.4),
]):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                         linewidth=1.0, color=C_LW, linestyle="--", alpha=0.85)
    ax.add_patch(a)
ax.text(5.0, 5.1, "長波輻射多重反射\n（夜間熱遲滯）", ha="center", fontsize=8.3, color=TEXT_MAIN, fontweight="bold")
ax.annotate("", xy=(5.0, 4.7), xytext=(5.0, 4.95),
            arrowprops=dict(arrowstyle="-", color=C_LW, lw=0))

ax.text(5.0, -0.15, r"街谷高寬比 $H/W$", ha="center", fontsize=8.3, style="italic")

# ── (b) H/W aspect-ratio driven windward downwash + leeward wake vortex ──
ax = axes[1]
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 8)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.3); spine.set_color("black")
ax.set_title("(b) 街谷高寬比 $H/W$ 主導之\n迎風下沖流與背風渦流帶", fontsize=10.5, fontweight="bold")

ground = Rectangle((-1, -0.5), 12, 0.5, facecolor="#8a8a8a", edgecolor="none")
ax.add_patch(ground)
b1 = Rectangle((0.5, 0), 2.0, 7.0, facecolor=C_BLDG, edgecolor="black", linewidth=1.0)
b2 = Rectangle((7.5, 0), 2.0, 4.0, facecolor=C_BLDG, edgecolor="black", linewidth=1.0)
ax.add_patch(b1); ax.add_patch(b2)
ax.text(1.5, 7.4, "迎風建築（高）", ha="center", fontsize=8.5)
ax.text(8.5, 4.4, "背風建築（低）", ha="center", fontsize=8.5)

# incoming wind (horizontal arrows from the left)
for y in [5.2, 6.0, 6.8]:
    a = FancyArrowPatch((-0.8, y), (0.4, y), arrowstyle="-|>", mutation_scale=11,
                         linewidth=1.3, color=C_WIND)
    ax.add_patch(a)
ax.text(-0.8, 7.3, "來流風", fontsize=8.3, color=TEXT_MAIN, fontweight="bold")

# downwash along the windward facade
downwash = FancyArrowPatch((2.55, 6.6), (2.9, 1.0), arrowstyle="-|>", mutation_scale=12,
                            linewidth=1.6, color=C_WIND, connectionstyle="arc3,rad=0.25")
ax.add_patch(downwash)
ax.text(3.6, 3.6, "迎風面\n下沖流", ha="center", fontsize=8.3, color=TEXT_MAIN, fontweight="bold")

# leeward wake-recirculation vortex (circular arrows)
vx, vy, vr = 5.6, 1.6, 1.1
theta = np.linspace(20, 340, 100)
vxs = vx + vr * np.cos(np.radians(theta))
vys = vy + vr * 0.6 * np.sin(np.radians(theta))
ax.plot(vxs, vys, color=C_VORTEX, linewidth=1.8, linestyle=(0, (5, 1.5)))
a = FancyArrowPatch((vxs[-2], vys[-2]), (vxs[-1], vys[-1]), arrowstyle="-|>",
                     mutation_scale=12, linewidth=1.8, color=C_VORTEX)
ax.add_patch(a)
ax.text(vx, vy - 1.0, "背風渦流帶\n(Wake Recirculation)", ha="center", fontsize=8.3,
        color=TEXT_MAIN, fontweight="bold")

ax.text(5.0, -0.15, r"街谷高寬比 $H/W$", ha="center", fontsize=8.3, style="italic")

legend_elems = [
    Line2D([0], [0], color=C_SUN, lw=2, label="短波輻射 / SVF 暴露立體角"),
    Line2D([0], [0], color=C_LW, lw=1.6, linestyle="--", label="長波輻射多重反射"),
    Line2D([0], [0], color=C_WIND, lw=2, label="風場（來流 / 下沖流）"),
    Line2D([0], [0], color=C_VORTEX, lw=2, label="背風渦流帶"),
]
fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=4, fontsize=8.5, frameon=False)

fig.suptitle("都市街谷幾何、熱島熱遲滯與空氣動力學機制示意", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_lit_urbanform.pdf"
out_png = FIG_DIR / "fig_lit_urbanform.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
