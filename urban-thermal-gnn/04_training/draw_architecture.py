"""
04_training/draw_architecture.py
Professional architecture diagram for PIN-ST-GNN.
Generates two figures:
  fig_arch.png  — Full model pipeline (Input → RGCN → LSTM → Output)
  fig_graph.png — Heterogeneous graph structure (node/edge types)

Usage:
    python draw_architecture.py
    python draw_architecture.py --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ── Colour palette (IEEE-friendly, print-safe) ─────────────────────────
C = {
    "obj":    "#2563EB",   # blue   — object nodes
    "air":    "#16A34A",   # green  — air nodes
    "sem":    "#9333EA",   # purple — semantic edges
    "knn":    "#EA580C",   # orange — contiguity edges
    "dyn":    "#DC2626",   # red    — dynamic edges
    "mlp":    "#0891B2",   # cyan   — MLP blocks
    "rgcn":   "#7C3AED",   # violet — RGCN block
    "lstm":   "#B45309",   # amber  — LSTM block
    "loss":   "#BE185D",   # pink   — loss terms
    "phys":   "#065F46",   # emerald— physics loss
    "bg":     "#F8FAFC",
    "text":   "#0F172A",
    "arrow":  "#475569",
    "border": "#CBD5E1",
}


def rounded_box(ax, x, y, w, h, label, sub=None,
                fc="#DBEAFE", ec="#2563EB", lw=1.5,
                fontsize=9, bold=False, alpha=0.92):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=fc, edgecolor=ec,
                          linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.06 if sub else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=C["text"], zorder=4)
    if sub:
        ax.text(x, y - 0.13, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#475569", zorder=4,
                style="italic")


def arrow(ax, x0, y0, x1, y1, label="", color="#475569",
          arrowstyle="-|>", lw=1.4, fontsize=7.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=12))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.04, my, label, fontsize=fontsize,
                color=color, va="center", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Full Model Architecture Pipeline
# ══════════════════════════════════════════════════════════════════════════
def draw_architecture(out_path: Path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(8, 6.6, "PIN-ST-GNN: Physics-Informed Spatio-Temporal GNN Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C["text"])

    # ── Row 1: Input blocks ───────────────────────────────────────────────
    #  Building nodes  |  Air sensor nodes  |  EPW / time context
    BW, BH = 2.2, 1.0

    # Object node input
    rounded_box(ax, 2, 5.2, BW, BH,
                "Object Nodes", r"$N_{obj}$ × 7",
                fc="#DBEAFE", ec=C["obj"], bold=True)

    # Air node input
    rounded_box(ax, 5.5, 5.2, BW, BH,
                "Air Nodes", r"$N_{air}$ × T × $d_{air}$=9",
                fc="#DCFCE7", ec=C["air"], bold=True)

    # Env / time context
    rounded_box(ax, 9.5, 5.2, BW, BH,
                "Env + Time Context", r"$(T, 7)$ + $(T, 2)$",
                fc="#FEF9C3", ec="#CA8A04", bold=True)

    # ── Row 2: Encoders ───────────────────────────────────────────────────
    rounded_box(ax, 2, 3.9, BW, 0.75,
                "Obj Encoder (MLP)", r"$7 \to 128$",
                fc="#EFF6FF", ec=C["obj"])

    rounded_box(ax, 5.5, 3.9, BW, 0.75,
                "Air Encoder (MLP)", r"$9 \to 128$ per $t$",
                fc="#F0FDF4", ec=C["air"])

    rounded_box(ax, 9.5, 3.9, BW, 0.75,
                "GlobalContextMLP",  r"$(T, 9) \to (T, 192)$",
                fc="#FEFCE8", ec="#CA8A04")

    # arrows input → encoder
    for xi in [2, 5.5, 9.5]:
        arrow(ax, xi, 4.7, xi, 4.27, color=C["arrow"])

    # ── Concatenation note ────────────────────────────────────────────────
    ax.text(3.75, 3.3, r"$h_{all}$ = cat[$h_{obj}$, $h_{air,t}$]  " +
            r"$(N_{obj}+N_{air},\;128)$",
            ha="center", fontsize=8, color="#6B7280",
            bbox=dict(fc="white", ec=C["border"], pad=3, boxstyle="round"))
    arrow(ax, 2, 3.52, 3.2, 3.42, color=C["obj"])
    arrow(ax, 5.5, 3.52, 4.3, 3.42, color=C["air"])

    # ── Row 3: RGCN ───────────────────────────────────────────────────────
    rounded_box(ax, 3.75, 2.45, 3.6, 0.85,
                "RGCN Encoder", "3 layers × 5 relation types\nResidual + LayerNorm",
                fc="#EDE9FE", ec=C["rgcn"], bold=True, fontsize=9)

    arrow(ax, 3.75, 3.05, 3.75, 2.88, color=C["arrow"])

    # 5 relation types legend
    rel_labels = ["shadow", "veg_et", "convective", "semantic", "contiguity"]
    rel_colors = [C["dyn"], C["dyn"], C["dyn"], C["sem"], C["knn"]]
    for i, (lbl, col) in enumerate(zip(rel_labels, rel_colors)):
        xi = 1.5 + i * 0.95
        ax.plot([xi, xi], [1.9, 2.03], color=col, lw=1.8)
        ax.text(xi, 1.8, lbl, ha="center", va="top", fontsize=6.5,
                color=col, rotation=35)
    ax.text(3.75, 1.65, "Edge relation types", ha="center",
            fontsize=7, color="#6B7280")

    # ── Fusion MLP ────────────────────────────────────────────────────────
    rounded_box(ax, 3.75, 0.85 + 3.0 - 1.5,   # y = 2.35
                2.2, 0.75,
                "Fusion MLP", r"$(d+ctx) \to 256$",
                fc="#E0F2FE", ec=C["mlp"])

    # Context arrow to fusion
    arrow(ax, 9.5, 3.52, 5.0, 2.35, color="#CA8A04", lw=1.2)
    arrow(ax, 3.75, 2.02, 3.75, 1.73, color=C["arrow"])

    # ── LSTM ──────────────────────────────────────────────────────────────
    rounded_box(ax, 3.75, 1.15, 2.2, 0.75,
                "LSTM", r"$(N_{air},\;T,\;256) \to (N_{air},\;T,\;256)$",
                fc="#FEF3C7", ec=C["lstm"], bold=True)

    # h0 initialisation note
    ax.text(5.8, 1.15, r"$h_0 = h_0$-proj(RGCN$_{t=0}$)",
            fontsize=7, color=C["lstm"], va="center")

    arrow(ax, 3.75, 1.52, 3.75, 1.52, color=C["arrow"])  # placeholder
    arrow(ax, 3.75, 1.73, 3.75, 1.52, color=C["arrow"])

    # ── Output MLP ────────────────────────────────────────────────────────
    rounded_box(ax, 3.75, 0.45, 2.2, 0.65,
                "Output MLP", r"$256 \to T_{pred}=11$",
                fc="#FDF4FF", ec="#A855F7")

    arrow(ax, 3.75, 0.77, 3.75, 0.77, color=C["arrow"])
    arrow(ax, 3.75, 0.78, 3.75, 0.62, color=C["arrow"])

    # Final output label
    ax.text(6.0, 0.45, r"$\hat{U}$  $(N_{air},\;11)$  UTCI prediction",
            fontsize=9, color=C["text"], va="center", fontweight="bold")

    # ── Physics Loss panel ────────────────────────────────────────────────
    px, py, pw, ph = 10.5, 3.0, 4.8, 2.8
    loss_bg = FancyBboxPatch((px, py - ph/2), pw, ph,
                              boxstyle="round,pad=0.08",
                              facecolor="#F0FDF4", edgecolor=C["phys"],
                              linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(loss_bg)
    ax.text(px + pw/2, py + ph/2 - 0.18,
            "Loss Functions", ha="center", fontsize=10,
            fontweight="bold", color=C["phys"])

    loss_entries = [
        (r"$\mathcal{L}_{data}$  MSE(pred, target)",          C["text"]),
        (r"$\mathcal{L}_{rad}$   Sun/shade UTCI constraint",  "#D97706"),
        (r"$\mathcal{L}_{temp}$  |$\Delta$UTCI| $\leq$ 5°C/hr", "#7C3AED"),
        (r"$\mathcal{L}_{wind}$  Leeward obstruction penalty","#0891B2"),
        (r"$\mathcal{L}_{total}$ = $\mathcal{L}_{data}$ + "
         r"$\lambda_1\mathcal{L}_{rad}$ + $\lambda_2\mathcal{L}_{temp}$ + "
         r"$\lambda_3\mathcal{L}_{wind}$", C["loss"]),
    ]
    for i, (txt, col) in enumerate(loss_entries):
        ax.text(px + 0.2, py + ph/2 - 0.55 - i * 0.48,
                txt, fontsize=7.8, color=col, va="center")

    # arrow from output to loss
    arrow(ax, 4.85, 0.45, 10.3, 1.8, color=C["arrow"], lw=1.2)

    # ── Parameter annotation ──────────────────────────────────────────────
    params_txt = ("Trainable params: ~1.2 M\n"
                  "hidden=128, RGCN×3, LSTM-256\n"
                  "RTX 5070 Ti  61.5±24.4 ms/scenario")
    ax.text(11.5, 0.5, params_txt, ha="center", va="center",
            fontsize=7.5, color="#6B7280",
            bbox=dict(fc="white", ec=C["border"], pad=4,
                      boxstyle="round", alpha=0.9))

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path / "fig_arch.png", dpi=200,
                facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path / 'fig_arch.png'}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Heterogeneous Graph Structure
# ══════════════════════════════════════════════════════════════════════════
def draw_graph(out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(C["bg"])

    # ── Left panel: node/edge schema ─────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 10)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    ax.set_title("Heterogeneous Graph Schema", fontsize=12,
                 fontweight="bold", color=C["text"])

    # Object nodes (top)
    obj_pos = [(2.5, 8), (5.5, 8), (8.5, 8)]
    for i, (x, y) in enumerate(obj_pos):
        circ = plt.Circle((x, y), 0.55, fc="#DBEAFE", ec=C["obj"],
                           lw=2.0, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, f"B{i+1}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=C["obj"])

    ax.text(5.5, 9.3, "Object Nodes (Buildings)  — 7D static features",
            ha="center", fontsize=9, color=C["obj"], fontweight="bold")

    # Semantic edges (object ↔ object, fully connected)
    for i, (x0, y0) in enumerate(obj_pos):
        for j, (x1, y1) in enumerate(obj_pos):
            if i < j:
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="<->",
                                            color=C["sem"], lw=1.5,
                                            connectionstyle="arc3,rad=0.2"))
    ax.text(5.5, 7.05, "semantic (fully-connected)", ha="center",
            fontsize=7.5, color=C["sem"], style="italic")

    # Air nodes (grid, 5×5 sample)
    rng = np.random.default_rng(42)
    air_xs = rng.uniform(1.5, 9.5, 16)
    air_ys = rng.uniform(2.0, 6.5, 16)

    # KNN contiguity edges
    from scipy.spatial import cKDTree
    pts = np.column_stack([air_xs, air_ys])
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=4)
    drawn = set()
    for i, nbrs in enumerate(idx):
        for j in nbrs[1:]:
            key = (min(i, j), max(i, j))
            if key not in drawn:
                ax.plot([air_xs[i], air_xs[j]], [air_ys[i], air_ys[j]],
                        color=C["knn"], lw=0.8, alpha=0.55, zorder=1)
                drawn.add(key)

    for i, (x, y) in enumerate(zip(air_xs, air_ys)):
        circ = plt.Circle((x, y), 0.32, fc="#DCFCE7", ec=C["air"],
                           lw=1.5, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, "a", ha="center", va="center",
                fontsize=6.5, color=C["air"])

    ax.text(5.5, 1.35, "Air Nodes (Sensor Points)  — 9D×T features",
            ha="center", fontsize=9, color=C["air"], fontweight="bold")
    ax.text(5.5, 0.75, "contiguity edges: KNN K=8",
            ha="center", fontsize=7.5, color=C["knn"], style="italic")

    # Dashed arrows: building → nearby air nodes (dynamic edges)
    for bx, by in obj_pos:
        dists = np.sqrt((air_xs - bx)**2 + (air_ys - by)**2)
        close = np.argsort(dists)[:3]
        for ci in close:
            ax.annotate("", xy=(air_xs[ci], air_ys[ci]),
                        xytext=(bx, by - 0.55),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=C["dyn"], lw=0.9,
                                        linestyle="dashed",
                                        mutation_scale=8, alpha=0.65))
    ax.text(9.5, 4.5, "dynamic\nedges\n(shadow,\nveg_et,\nconvective)",
            ha="center", fontsize=7, color=C["dyn"],
            bbox=dict(fc="white", ec=C["dyn"], pad=3, boxstyle="round"))

    # Legend
    legend_items = [
        mpatches.Patch(fc="#DBEAFE", ec=C["obj"], label="Object node (Building)"),
        mpatches.Patch(fc="#DCFCE7", ec=C["air"], label="Air node (Sensor pt.)"),
        mpatches.Patch(fc=C["sem"],  ec=C["sem"],  label="Semantic edge (obj↔obj)"),
        mpatches.Patch(fc=C["knn"],  ec=C["knn"],  label="Contiguity edge (KNN-8)"),
        mpatches.Patch(fc=C["dyn"],  ec=C["dyn"],  label="Dynamic edge (per step)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.95, edgecolor=C["border"])

    # ── Right panel: feature dimension breakdown ──────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.5, 11)
    ax2.axis("off")
    ax2.set_facecolor(C["bg"])
    ax2.set_title("Feature Dimensions & Edge Types", fontsize=12,
                  fontweight="bold", color=C["text"])

    # Object node features table
    ax2.text(5, 10.5, "Object Node Features  (N_obj, 7)",
             ha="center", fontsize=10, fontweight="bold", color=C["obj"])
    obj_feats = [
        ("0", "height",        "height / 50.0",      "static"),
        ("1", "floors",        "floors / 12.0",       "static"),
        ("2", "footprint_area","area / 2000.0",       "static"),
        ("3", "centroid_x",    "cx / 80.0",           "static"),
        ("4", "centroid_y",    "cy / 80.0",           "static"),
        ("5", "GFA",           "gfa / 20000.0",       "static"),
        ("6", "shape_type",    "0=rect / 1=L",        "static"),
    ]
    col_x = [0.4, 1.0, 3.8, 7.0]
    for row, (idx, name, norm, typ) in enumerate(obj_feats):
        y = 9.7 - row * 0.4
        ax2.text(col_x[0], y, idx,  fontsize=7.5, color="#6B7280", va="center")
        ax2.text(col_x[1], y, name, fontsize=7.5, color=C["text"],  va="center")
        ax2.text(col_x[2], y, norm, fontsize=7.0, color="#475569",  va="center")
        ax2.text(col_x[3], y, typ,  fontsize=7.0, color="#7C3AED",  va="center")

    ax2.axhline(6.7, color=C["border"], lw=0.8, xmin=0.02, xmax=0.98)

    # Air node features table
    ax2.text(5, 6.4, "Air Node Features  (N_air, T=11, dim_air=9)",
             ha="center", fontsize=10, fontweight="bold", color=C["air"])
    air_feats = [
        ("0", "Ta",       "z-score", "dynamic"),
        ("1", "MRT",      "z-score", "dynamic"),
        ("2", "Va",       "z-score", "dynamic"),
        ("3", "RH",       "z-score", "dynamic"),
        ("4", "SVF",      "[0,1]",   "STATIC"),
        ("5", "shadow",   "0/1",     "dynamic"),
        ("6", "Bh",       "/ 50.0",  "STATIC"),
        ("7", "Th",       "/ 12.0",  "STATIC"),
        ("8", "Ts (V2)",  "z-score", "dynamic"),
    ]
    for row, (idx, name, norm, typ) in enumerate(air_feats):
        y = 5.8 - row * 0.46
        col = C["phys"] if "STATIC" in typ else C["air"]
        ax2.text(col_x[0], y, idx,  fontsize=7.5, color="#6B7280", va="center")
        ax2.text(col_x[1], y, name, fontsize=7.5, color=C["text"],  va="center")
        ax2.text(col_x[2], y, norm, fontsize=7.0, color="#475569",  va="center")
        ax2.text(col_x[3], y, typ,  fontsize=7.0, color=col,        va="center")

    ax2.axhline(1.4, color=C["border"], lw=0.8, xmin=0.02, xmax=0.98)

    # Edge types summary
    ax2.text(5, 1.1, "Edge Types", ha="center", fontsize=10,
             fontweight="bold", color=C["text"])
    edge_rows = [
        ("semantic",    "object→object",  "fully-connected",   C["sem"]),
        ("contiguity",  "air→air",        "KNN K=8, +N_obj offset", C["knn"]),
        ("shadow",      "object→air",     "per-step dynamic",  C["dyn"]),
        ("veg_et",      "object→air",     "per-step dynamic",  C["dyn"]),
        ("convective",  "object→air",     "per-step dynamic",  C["dyn"]),
    ]
    for row, (rel, nodes, desc, col) in enumerate(edge_rows):
        y = 0.6 - row * 0.38
        ax2.text(0.3, y, rel,   fontsize=7.5, color=col,      va="center", fontweight="bold")
        ax2.text(2.5, y, nodes, fontsize=7.0, color=C["text"],va="center")
        ax2.text(5.5, y, desc,  fontsize=7.0, color="#475569",va="center")

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path / "fig_graph.png", dpi=200,
                facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path / 'fig_graph.png'}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: RGCN Message-Passing Diagram
# ══════════════════════════════════════════════════════════════════════════
def draw_rgcn_detail(out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    ax.text(6.5, 5.7, "RGCN Message Passing — One Block (5 Relation Types)",
            ha="center", fontsize=12, fontweight="bold", color=C["text"])

    # Central target node
    circ = plt.Circle((6.5, 3.0), 0.6, fc="#EDE9FE", ec=C["rgcn"],
                       lw=2.5, zorder=4)
    ax.add_patch(circ)
    ax.text(6.5, 3.0, r"$h_i$", ha="center", va="center",
            fontsize=11, color=C["rgcn"], fontweight="bold")
    ax.text(6.5, 2.2, "target node", ha="center", fontsize=8, color="#6B7280")

    # Source nodes (5 relation types, arranged in a fan)
    angles = [150, 120, 90, 60, 30]
    rels   = ["shadow", "veg_et", "convective", "semantic", "contiguity"]
    colors = [C["dyn"], C["dyn"], C["dyn"], C["sem"], C["knn"]]
    r = 2.8

    for angle_deg, rel, col in zip(angles, rels, colors):
        angle = np.radians(angle_deg)
        sx = 6.5 + r * np.cos(angle)
        sy = 3.0 + r * np.sin(angle)

        circ_s = plt.Circle((sx, sy), 0.4, fc="white", ec=col,
                             lw=1.8, zorder=3)
        ax.add_patch(circ_s)
        ax.text(sx, sy, r"$h_j$", ha="center", va="center",
                fontsize=9, color=col)

        # Arrow with relation label
        dx = (6.5 - sx)
        dy = (3.0 - sy)
        norm_d = np.sqrt(dx**2 + dy**2)
        ex = sx + dx / norm_d * 0.4
        ey = sy + dy / norm_d * 0.4
        tx = 6.5 - dx / norm_d * 0.6
        ty = 3.0 - dy / norm_d * 0.6

        ax.annotate("", xy=(tx, ty), xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.5, mutation_scale=10))

        # Label midpoint
        mx = (ex + tx) / 2
        my = (ey + ty) / 2
        ax.text(mx, my, f"W_{rel[:3]}", ha="center", va="center",
                fontsize=7, color=col,
                bbox=dict(fc="white", ec=col, pad=1.5, boxstyle="round",
                          alpha=0.85))

    # Self-loop
    self_loop = mpatches.Arc((6.5, 3.0), 1.4, 1.4, angle=0,
                              theta1=200, theta2=340,
                              color="#6B7280", lw=1.5, linestyle="dashed")
    ax.add_patch(self_loop)
    ax.text(7.6, 2.1, r"$W_{self}$", fontsize=8, color="#6B7280")

    # Formula panel
    formula_txt = (
        r"$h_i^{(l+1)} = \sigma\!\left(\sum_{r} \sum_{j \in \mathcal{N}_r(i)}"
        r"\frac{1}{|\mathcal{N}_r(i)|} W_r h_j^{(l)} + W_{self} h_i^{(l)}\right)$"
        "\n\n"
        r"$r \in \{$shadow, veg_et, convective, semantic, contiguity$\}$"
        "\n"
        r"Norm + Residual:  $h_i^{(l+1)} \leftarrow$ LayerNorm$(h_i^{(l+1)} + h_i^{(l)})$"
    )
    ax.text(10.5, 3.0, formula_txt, ha="center", va="center",
            fontsize=8.5, color=C["text"],
            bbox=dict(fc="#F5F3FF", ec=C["rgcn"], pad=10,
                      boxstyle="round", alpha=0.95))

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path / "fig_rgcn_detail.png", dpi=200,
                facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path / 'fig_rgcn_detail.png'}")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Draw PIN-ST-GNN architecture diagrams")
    ap.add_argument("--out", default="figures", help="Output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    draw_architecture(out)
    draw_graph(out)
    draw_rgcn_detail(out)
    print("\nDone. Three diagrams generated:")
    print(f"  {out}/fig_arch.png       — Full pipeline diagram")
    print(f"  {out}/fig_graph.png      — Heterogeneous graph schema + feature table")
    print(f"  {out}/fig_rgcn_detail.png — RGCN message-passing diagram")


if __name__ == "__main__":
    main()
