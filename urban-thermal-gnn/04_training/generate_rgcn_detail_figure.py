"""
generate_rgcn_detail_figure.py
================================
RGCN message-passing mechanism diagram for Thesis_GIA chapter 3
(sec:model_overview / fig:rgcn_detail): one relational-GCN block
aggregating messages from a target node's neighbours across the five
relation types (shadow, veg_et, convective, semantic, contiguity) plus
a self-loop, followed by residual + LayerNorm.

House style: matches fig_overview_pistgnn_white.png -- white background,
white node/formula boxes with black borders, black arrows, bold bilingual
labels; colour reserved for small relation-type tags only.

Produces:
  figures/fig_rgcn_detail.pdf
  figures/fig_rgcn_detail.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_diagram_style import apply_rcparams, box, tag, ACCENTS, TEXT_MAIN, BOX_EDGE, BOX_FILL

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(9.5)

fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.4)
ax.axis("off")

ax.text(7.0, 7.05, "RGCN 訊息聚合機制示意圖　One Block, 5 Relation Types",
        ha="center", fontsize=13, fontweight="bold", color=TEXT_MAIN)

# ── central target node ─────────────────────────────────────────────────
cx, cy = 4.4, 3.5
ax.add_patch(Circle((cx, cy), 0.6, facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.6, zorder=4))
ax.text(cx, cy, r"$h_i$", ha="center", va="center", fontsize=12, color=TEXT_MAIN, fontweight="bold", zorder=5)
ax.text(cx, cy - 1.05, "目標節點\ntarget node", ha="center", fontsize=8.2, color=TEXT_MAIN)

# ── 5 relation-type source nodes arranged in a fan ──────────────────────
angles = [150, 120, 90, 60, 30]
rels   = ["shadow", "veg_et", "convective", "semantic", "contiguity"]
tag_colors = [ACCENTS[0], ACCENTS[1], ACCENTS[2], ACCENTS[3], ACCENTS[4]]
r = 2.2

for angle_deg, rel, col in zip(angles, rels, tag_colors):
    angle = np.radians(angle_deg)
    sx = cx + r * np.cos(angle)
    sy = cy + r * np.sin(angle)

    ax.add_patch(Circle((sx, sy), 0.4, facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.3, zorder=3))
    ax.text(sx, sy, r"$h_j$", ha="center", va="center", fontsize=9.5, color=TEXT_MAIN, zorder=4)

    dx, dy = (cx - sx), (cy - sy)
    norm_d = np.sqrt(dx**2 + dy**2)
    ex, ey = sx + dx / norm_d * 0.42, sy + dy / norm_d * 0.42
    tx, ty = cx - dx / norm_d * 0.65, cy - dy / norm_d * 0.65

    ax.add_patch(FancyArrowPatch((ex, ey), (tx, ty), arrowstyle="-|>", mutation_scale=13,
                                  linewidth=1.5, color="#1a1a1a", zorder=2))

    mx, my = (ex + tx) / 2, (ey + ty) / 2
    tag(ax, mx, my, f"W_{rel[:3]}", col, fontsize=7.0)

    label_x = cx + (r + 0.78) * np.cos(angle)
    label_y = cy + (r + 0.78) * np.sin(angle)
    ax.text(label_x, label_y, rel, ha="center", va="center", fontsize=8.3,
            color=TEXT_MAIN, fontweight="bold")

# ── self-loop (bulges to the lower-right, clear of the fan and the label) ─
ax.add_patch(Arc((cx, cy), 1.5, 1.5, angle=0, theta1=280, theta2=60,
                  color="#1a1a1a", linewidth=1.4, linestyle="dashed"))
ax.text(cx + 1.25, cy - 0.35, r"$W_{self}$", fontsize=8.5, color=TEXT_MAIN, fontweight="bold")

# ── formula panel ────────────────────────────────────────────────────────
formula_txt = (
    r"$h_i^{(l+1)} = \sigma\left(\sum_{r}\sum_{j \in \mathcal{N}_r(i)}"
    r"\frac{1}{|\mathcal{N}_r(i)|} W_r h_j^{(l)}\right.$"
    "\n"
    r"$\qquad\qquad\quad \left. + \, W_{self} h_i^{(l)}\right)$"
    "\n\n"
    r"$r \in \{$shadow, veg_et, convective," "\n"
    r"$\qquad\;$ semantic, contiguity$\}$"
    "\n\n"
    "殘差 + 正規化　Norm + Residual:\n"
    r"$h_i^{(l+1)} \leftarrow \mathrm{LayerNorm}(h_i^{(l+1)} + h_i^{(l)})$"
)
box(ax, 7.9, 1.6, 5.6, 2.9, formula_txt, fontsize=9.5)

fig.tight_layout()

out_pdf = FIG_DIR / "fig_rgcn_detail.pdf"
out_png = FIG_DIR / "fig_rgcn_detail.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
