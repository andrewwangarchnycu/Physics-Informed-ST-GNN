"""
generate_arch_overview_figure.py
===================================
Redraw of fig_arch_clean.png (referenced as fig:arch_pipeline in Thesis_GIA
Ch3 subsec:model_overview): a compact box-and-arrow summary of the model's
computational core -- input (object + air node features) -> RGCN+LSTM ->
output UTCI field, with the physics-informed loss constraint annotated.

House style: matches fig_overview_pistgnn_white.png -- white background,
white component boxes with black borders, black arrows, bold bilingual
labels; colour reserved for small stage-reference tags only.

Produces:
  figures/fig_arch_clean.png/pdf
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, zone, box, arrow, tag, ACCENTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.6)

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.0)
ax.axis("off")

# --- inputs ---
box(ax, 0.3, 3.35, 2.7, 1.3, "物件節點\nObject Nodes\n$(N_{\\mathrm{obj}}, 7)$", 9.3)
box(ax, 0.3, 1.25, 2.7, 1.3, "空氣節點\nAir Nodes\n$(N_{\\mathrm{air}}, T{=}11, 9)$", 9.3)

# --- core ---
box(ax, 4.05, 2.3, 3.0, 1.3, "RGCN 空間建模\n+\nLSTM 時序融合", 9.3)

# --- output ---
box(ax, 8.0, 2.3, 2.7, 1.3, "UTCI 熱壓力場\n$(N_{\\mathrm{air}}, T{=}11)$", 9.3)

arrow(ax, (3.0, 4.0), (4.05, 3.25))
arrow(ax, (3.0, 1.9), (4.05, 2.7))
arrow(ax, (7.05, 2.95), (8.0, 2.95))

# --- physics loss annotation ---
box(ax, 4.05, 0.3, 3.0, 1.15, "物理引導損失函數\n$\\mathcal{L}_{\\mathrm{rad}}, \\mathcal{L}_{\\mathrm{temp}}, \\mathcal{L}_{\\mathrm{wind}}$", 8.8)
arrow(ax, (5.55, 1.45), (5.55, 2.3))

tag(ax, 0.75, 4.85, "輸入", ACCENTS[2])
tag(ax, 4.5, 3.8, "核心", ACCENTS[3])
tag(ax, 8.45, 3.8, "輸出", ACCENTS[1])
tag(ax, 4.5, 1.65, "物理約束", ACCENTS[0])

ax.text(5.55, 5.55, "PI-ST-GNN 整合運算框架示意圖", ha="center", fontsize=13, fontweight="bold", color="#000000")

fig.tight_layout()

out_pdf = FIG_DIR / "fig_arch_clean.pdf"
out_png = FIG_DIR / "fig_arch_clean.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
