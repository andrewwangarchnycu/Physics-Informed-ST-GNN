"""
generate_lit_utci_figure.py
================================
Original schematic diagram (not copied from any source paper) for
Thesis_GIA chapter 2, subsec:thermal_indices:
  (a) the four microclimate variables (Ta, Tmrt, va, RH) feeding the
      Fiala multi-node human thermal-physiology model that synthesises UTCI
  (b) the six-tier UTCI thermal-stress classification, annotated with the
      Taiwan-context empirical range this thesis repeatedly cites
      (>=38 degC common in summer noon; su2023outdoor canyon H/W findings)

House style: matches fig_overview_pistgnn_white.png -- white boxes with
black borders inside a light-gray zone for panel (a); panel (b) keeps its
severity color scale (a legitimate ordinal classification convention, akin
to the reference style's own coloured relation tags) but muted and framed
to match.

Produces:
  figures/fig_lit_utci_factors.pdf
  figures/fig_lit_utci_factors.png
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, zone, box as hbox, arrow, ARROW_COLOR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1.05, 1.0]})

# ── (a) four variables -> Fiala multi-node model -> UTCI ──
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.3); spine.set_color("black")
ax.set_title("(a) 四項微氣候變數經 Fiala 多節點\n人體熱生理模型合成 UTCI", fontsize=10.5, fontweight="bold")

zone(ax, 0.15, 6.55, 9.7, 2.15, "微氣候輸入", "Microclimate Inputs", fontsize=10)
hbox(ax, 0.4, 6.75, 2.1, 1.2, "空氣溫度\n$T_a$", 9.5)
hbox(ax, 2.7, 6.75, 2.1, 1.2, "平均輻射溫度\n$T_{mrt}$", 9.5)
hbox(ax, 5.1, 6.75, 2.1, 1.2, "風速\n$v_a$", 9.5)
hbox(ax, 7.5, 6.75, 2.1, 1.2, "相對濕度\nRH", 9.5)

hbox(ax, 1.2, 4.3, 7.6, 1.6,
     "Fiala 多節點人體熱傳遞與溫度調節模型\n"
     "Multi-node Thermoregulation Model\n"
     "12 軀幹部位 × 187 組織節點；血管擴張/收縮、排汗、顫抖", 8.9)

for x0 in [1.35, 3.75, 6.15, 8.55]:
    arrow(ax, (x0, 6.75), (5.0, 5.9), color=ARROW_COLOR, lw=1.2, mutation_scale=11)

hbox(ax, 2.5, 1.6, 5.0, 1.4, "通用熱氣候指數\nUniversal Thermal Climate Index (UTCI)\n六等熱壓力分級輸出", 9.3)
arrow(ax, (5.0, 4.3), (5.0, 3.0), color=ARROW_COLOR, lw=1.5, mutation_scale=13)

# ── (b) six-tier UTCI thermal-stress classification with Taiwan context ──
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(-0.9, 9)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.3); spine.set_color("black")
ax.set_title("(b) UTCI 六等熱壓力分級與\n台灣夏季在地實測脈絡", fontsize=10.5, fontweight="bold")

# muted, monochrome-compatible severity ramp (still ordinal/informative,
# but desaturated relative to the earlier saturated red-orange-green-blue set)
tiers = [
    ("極端熱壓力", "> 46", "#5c1414"),
    ("很強熱壓力", "38 – 46", "#8a3a1f"),
    ("強烈熱壓力", "32 – 38", "#a86a35"),
    ("中度熱壓力", "26 – 32", "#b89a55"),
    ("無熱壓力（舒適）", "9 – 26", "#6f8a63"),
    ("輕度至中度冷壓力", "< 9", "#4a6a8a"),
]

def tierbox(x, y, w, h, text, color, fontsize=8.6):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.06",
                        linewidth=1.0, edgecolor="black", facecolor=color, alpha=0.88)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color="white", fontweight="bold", linespacing=1.3)

y0 = 8.1
bar_h = 1.15
for name, rng, color in tiers:
    tierbox(0.3, y0 - bar_h, 6.0, bar_h - 0.12, f"{name}\nUTCI {rng} °C", color)
    y0 -= bar_h

ax.annotate("台灣夏至正午街谷常態\nUTCI > 38 °C\n(Su, 2023)",
            xy=(6.5, 5.7), xytext=(6.5, 5.7), fontsize=8.5, color="black",
            fontweight="bold", ha="left", va="center")
brace = FancyArrowPatch((6.35, 6.9), (6.35, 4.5), arrowstyle="-", mutation_scale=1,
                         linewidth=1.6, color="black", connectionstyle="arc3,rad=0.15")
ax.add_patch(brace)

ax.text(3.3, -0.15,
        "高密度街廓因風速增強與遮蔭效果，\n"
        "正午熱舒適表現可能優於低密度\n"
        "開闊街廓（非單調之密度—熱舒適關係）",
        ha="center", fontsize=8.2, color="#222222", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ececec", edgecolor="black", linewidth=0.9))

fig.suptitle("UTCI 之生理學合成基礎與台灣在地熱壓力分級對照", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_lit_utci_factors.pdf"
out_png = FIG_DIR / "fig_lit_utci_factors.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
