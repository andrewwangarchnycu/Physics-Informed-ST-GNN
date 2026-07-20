"""
generate_arch_overview_figure.py
===================================
Redraw of fig_arch_clean.png (referenced as fig:arch_pipeline in Thesis_GIA
Ch3 subsec:model_overview) as a clean matplotlib diagram, replacing the old
draw.io-exported PNG whose baked-in title still read "PIN-ST-GNN" -- stale
from before the whole-thesis PIN-ST-GNN -> PI-ST-GNN rename. No drawio CLI
is available in this environment to re-export the .xml source, so this is
regenerated directly as a compact box-and-arrow summary of the same content:
input (object + air node features) -> RGCN+LSTM -> output UTCI field, with
the physics-informed loss constraint annotated.

Produces:
  figures/fig_arch_clean.png/pdf
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.6,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_IN   = "#5b7fa6"   # input -- blue
C_MID  = "#8a6fa3"   # RGCN+LSTM core -- purple
C_OUT  = "#7a9e6a"   # output -- green
C_LOSS = "#c9704f"   # physics loss -- orange

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.6)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=9.3, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.10",
                        linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, linespacing=1.35)

def arrow(p1, p2, color="#444444", lw=1.5):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                         linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# --- inputs ---
box(0.3, 3.2, 2.7, 1.3, "物件節點\nObject Nodes\n$(N_{\\mathrm{obj}}, 7)$", C_IN)
box(0.3, 1.1, 2.7, 1.3, "空氣節點\nAir Nodes\n$(N_{\\mathrm{air}}, T{=}11, 9)$", C_IN)

# --- core ---
box(4.05, 2.15, 3.0, 1.3, "RGCN 空間建模\n+\nLSTM 時序融合", C_MID)

# --- output ---
box(8.0, 2.15, 2.7, 1.3, "UTCI 熱壓力場\n$(N_{\\mathrm{air}}, T{=}11)$", C_OUT)

arrow((3.0, 3.85), (4.05, 3.1))
arrow((3.0, 1.75), (4.05, 2.55))
arrow((7.05, 2.8), (8.0, 2.8))

# --- physics loss annotation ---
box(4.05, 0.25, 3.0, 1.15, "物理引導損失函數\n$\\mathcal{L}_{\\mathrm{rad}}, \\mathcal{L}_{\\mathrm{temp}}, \\mathcal{L}_{\\mathrm{wind}}$", C_LOSS, fontsize=8.8)
arrow((5.55, 1.4), (5.55, 2.15), color=C_LOSS)

ax.text(5.55, 5.15, "PI-ST-GNN 整合運算框架示意圖", ha="center", fontsize=13, fontweight="bold")

legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_IN, markersize=12, label="輸入（建物幾何特徵 + 多時步氣象特徵）"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MID, markersize=12, label="核心（RGCN + LSTM）"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OUT, markersize=12, label="輸出（全場域 UTCI 熱壓力場）"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_LOSS, markersize=12, label="物理資訊損失函數（訓練期約束）"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=8.3, frameon=False)

fig.tight_layout()

out_pdf = FIG_DIR / "fig_arch_clean.pdf"
out_png = FIG_DIR / "fig_arch_clean.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
