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

Layout notes:
- The left (graph) and right (formula box) panels' vertical extents are
  computed from the same top/bottom anchors so the two panels' top and
  bottom edges line up exactly, rather than being independently eyeballed.
- The main aggregation formula is rendered as a SINGLE line (one ax.text
  call, not split across two lines as in earlier revisions) -- the box is
  sized wide enough to fit it without wrapping.
- Explanatory Chinese/English labels (subtitle, relation-type labels,
  tags, target-node caption, the r-in-set line, and the
  residual/normalization line) are sized ~1.5x larger than the prior
  revision for legibility; the formula lines themselves are enlarged more
  modestly since they are already large relative to body text.

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
from thesis_diagram_style import apply_rcparams, box, tag, ACCENTS, TEXT_MAIN, TEXT_SUB, BOX_EDGE, BOX_FILL

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
apply_rcparams(11.5)

fig, ax = plt.subplots(figsize=(19, 10.5))
ax.set_xlim(0, 19)
ax.set_ylim(0, 10.6)
ax.axis("off")

ax.text(9.5, 10.3, "RGCN 訊息聚合機制示意圖",
        ha="center", fontsize=18, fontweight="bold", color=TEXT_MAIN)
ax.text(9.5, 9.85, "One Block · 5 Relation Types",
        ha="center", fontsize=19, color=TEXT_SUB)

# ── central target node ─────────────────────────────────────────────────
cx, cy = 5.2, 4.5
node_r = 0.85
ax.add_patch(Circle((cx, cy), node_r, facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.9, zorder=4))
ax.text(cx, cy, r"$h_i$", ha="center", va="center", fontsize=17, color=TEXT_MAIN, fontweight="bold", zorder=5)

target_label_y = cy - node_r - 0.65
ax.text(cx, target_label_y, "目標節點\ntarget node", ha="center", va="top",
        fontsize=17, color=TEXT_MAIN, linespacing=1.4)
# bottom extent of the two-line label (used to align the formula box below)
LEFT_BOTTOM = target_label_y - 0.95

# ── 5 relation-type source nodes arranged in a fan ──────────────────────
angles = [150, 120, 90, 60, 30]
rels   = ["shadow", "veg_et", "convective", "semantic", "contiguity"]
tag_colors = [ACCENTS[0], ACCENTS[1], ACCENTS[2], ACCENTS[3], ACCENTS[4]]
r = 2.7
sat_r = 0.62
label_pad = 1.3

for angle_deg, rel, col in zip(angles, rels, tag_colors):
    angle = np.radians(angle_deg)
    sx = cx + r * np.cos(angle)
    sy = cy + r * np.sin(angle)

    ax.add_patch(Circle((sx, sy), sat_r, facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=1.6, zorder=3))
    ax.text(sx, sy, r"$h_j$", ha="center", va="center", fontsize=13.5, color=TEXT_MAIN, zorder=4)

    dx, dy = (cx - sx), (cy - sy)
    norm_d = np.sqrt(dx**2 + dy**2)
    ex, ey = sx + dx / norm_d * sat_r, sy + dy / norm_d * sat_r
    tx, ty = cx - dx / norm_d * (node_r + 0.05), cy - dy / norm_d * (node_r + 0.05)

    ax.add_patch(FancyArrowPatch((ex, ey), (tx, ty), arrowstyle="-|>", mutation_scale=16,
                                  linewidth=1.8, color="#1a1a1a", zorder=2))

    mx, my = (ex + tx) / 2, (ey + ty) / 2
    tag(ax, mx, my, f"W_{rel[:3]}", col, fontsize=14)

    label_x = cx + (r + label_pad) * np.cos(angle)
    label_y = cy + (r + label_pad) * np.sin(angle)
    ax.text(label_x, label_y, rel, ha="center", va="center", fontsize=16.5,
            color=TEXT_MAIN, fontweight="bold")

# top extent of the fan labels (used to align the formula box above)
LEFT_TOP = cy + r + label_pad + 0.40

# ── self-loop (bulges straight down, clear of the fan above and the
#    "target node" label further below) ──────────────────────────────────
ax.add_patch(Arc((cx, cy), node_r * 2.3, node_r * 2.3, angle=0, theta1=200, theta2=340,
                  color="#1a1a1a", linewidth=1.7, linestyle="dashed"))
ax.text(cx + node_r * 2.2, cy - node_r - 0.05, r"$W_{self}$", fontsize=16.5, color=TEXT_MAIN,
        fontweight="bold", ha="left")

# ── formula panel — top/bottom explicitly matched to the left diagram ──
px, pw = 9.9, 8.7
py, ph = LEFT_BOTTOM, LEFT_TOP - LEFT_BOTTOM
box(ax, px, py, pw, ph, "", fontsize=11.5)
xc = px + pw / 2

# Main aggregation formula: ONE line, not split across two ax.text calls.
ax.text(xc, py + ph - 1.05,
        r"$h_i^{(l+1)} = \sigma\left(\sum_{r}\sum_{j \in \mathcal{N}_r(i)}"
        r"\frac{1}{|\mathcal{N}_r(i)|} W_r h_j^{(l)} + W_{self} h_i^{(l)}\right)$",
        ha="center", va="center", fontsize=15.5, color=TEXT_MAIN)

ax.text(xc, py + ph - 2.35,
        r"$r \in \{\mathrm{shadow, veg\_et, convective, semantic, contiguity}\}$",
        ha="center", va="center", fontsize=16.5, color=TEXT_MAIN)

ax.text(xc, py + ph - 3.65, "殘差＋正規化　Norm + Residual",
        ha="center", va="center", fontsize=19, color=TEXT_MAIN, fontweight="bold")
ax.text(xc, py + ph - 4.75,
        r"$h_i^{(l+1)} \leftarrow \mathrm{LayerNorm}(h_i^{(l+1)} + h_i^{(l)})$",
        ha="center", va="center", fontsize=15.5, color=TEXT_MAIN)

fig.tight_layout()

out_pdf = FIG_DIR / "fig_rgcn_detail.pdf"
out_png = FIG_DIR / "fig_rgcn_detail.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"Saved: {out_pdf}\n       {out_png}")
print(f"LEFT_TOP={LEFT_TOP:.2f}  LEFT_BOTTOM={LEFT_BOTTOM:.2f}  box top={py+ph:.2f}  box bottom={py:.2f}  box right={px+pw:.2f}")
