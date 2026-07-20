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

Produces:
  figures/fig_lit_utci_factors.pdf
  figures/fig_lit_utci_factors.png
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

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

C_TA   = "#c9704f"
C_TMRT = "#d99a2b"
C_VA   = "#3f7fa6"
C_RH   = "#7a9e6a"
C_CORE = "#5b4a7a"
C_OUT  = "#4a4a4a"

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0]})

# ── (a) four variables -> Fiala multi-node model -> UTCI ──
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("(a) 四項微氣候變數經 Fiala 多節點\n人體熱生理模型合成 UTCI", fontsize=10.5, fontweight="bold")

def box(x, y, w, h, text, color, fontsize=9, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                        linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, linespacing=1.3)

box(0.3, 7.0, 2.1, 1.2, r"空氣溫度" "\n" r"$T_a$", C_TA)
box(2.7, 7.0, 2.1, 1.2, r"平均輻射溫度" "\n" r"$T_{mrt}$", C_TMRT)
box(5.1, 7.0, 2.1, 1.2, r"風速" "\n" r"$v_a$", C_VA)
box(7.5, 7.0, 2.1, 1.2, r"相對濕度" "\n" r"RH", C_RH)

box(1.2, 4.3, 7.6, 1.6,
    "Fiala 多節點人體熱傳遞與溫度調節模型\n"
    "12 軀幹部位 × 187 組織節點\n"
    "血管擴張/收縮、排汗、顫抖等主動生理調適",
    C_CORE, fontsize=9.3)

for x0 in [1.35, 3.75, 6.15, 8.55]:
    a = FancyArrowPatch((x0, 7.0), (5.0, 5.9), arrowstyle="-|>", mutation_scale=11,
                         linewidth=1.2, color="#666666")
    ax.add_patch(a)

box(2.5, 1.6, 5.0, 1.4, "通用熱氣候指數\nUniversal Thermal Climate Index (UTCI)\n六等熱壓力分級輸出", C_OUT, fontsize=9.3)
a = FancyArrowPatch((5.0, 4.3), (5.0, 3.0), arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#666666")
ax.add_patch(a)

legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TA, markersize=11, label=r"$T_a$ 空氣溫度"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_TMRT, markersize=11, label=r"$T_{mrt}$ 平均輻射溫度"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_VA, markersize=11, label=r"$v_a$ 風速"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_RH, markersize=11, label="RH 相對濕度"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=8.3, frameon=False)

# ── (b) six-tier UTCI thermal-stress classification with Taiwan context ──
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("(b) UTCI 六等熱壓力分級與\n台灣夏季在地實測脈絡", fontsize=10.5, fontweight="bold")

tiers = [
    ("極端熱壓力", "> 46", "#7a1f1f"),
    ("很強熱壓力", "38 – 46", "#b3401f"),
    ("強烈熱壓力", "32 – 38", "#d9702b"),
    ("中度熱壓力", "26 – 32", "#e0a83f"),
    ("無熱壓力（舒適）", "9 – 26", "#7a9e6a"),
    ("輕度至中度冷壓力", "< 9", "#4f7fa6"),
]
y0 = 8.1
bar_h = 1.15
for name, rng, color in tiers:
    box(0.3, y0 - bar_h, 6.0, bar_h - 0.12, f"{name}\nUTCI {rng} °C", color, fontsize=8.6)
    y0 -= bar_h

ax.annotate("台灣夏至正午街谷常態\nUTCI > 38 °C\n(Su, 2023)",
            xy=(6.5, 5.7), xytext=(6.5, 5.7), fontsize=8.5, color="#7a1f1f",
            fontweight="bold", ha="left", va="center")
brace = FancyArrowPatch((6.35, 6.9), (6.35, 4.5), arrowstyle="-", mutation_scale=1,
                         linewidth=1.6, color="#7a1f1f", connectionstyle="arc3,rad=0.15")
ax.add_patch(brace)

ax.text(3.3, 1.4,
        "高密度街廓因風速增強與遮蔭效果，\n"
        "正午熱舒適表現可能優於低密度\n"
        "開闊街廓（非單調之密度—熱舒適關係）",
        ha="center", fontsize=8.2, color="#333333", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0ede8", edgecolor="#999999"))

fig.suptitle("UTCI 之生理學合成基礎與台灣在地熱壓力分級對照", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_lit_utci_factors.pdf"
out_png = FIG_DIR / "fig_lit_utci_factors.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
