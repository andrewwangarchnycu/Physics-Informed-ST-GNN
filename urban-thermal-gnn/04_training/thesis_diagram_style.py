"""
thesis_diagram_style.py
=========================
Shared "house style" helpers for schematic/diagram figures in Thesis_GIA,
matching fig_overview_pistgnn_white.png (the reference style requested by
the user): white page background, light-gray rounded "zone" containers with
black borders, white rounded component boxes with black borders, bold
bilingual (Chinese + English) black text, black/dark-gray arrows, and only
small, sparing colour accents for category-distinguishing tags/icons rather
than saturated full-box fills.

Import and reuse across generate_*.py diagram scripts instead of each
script hand-rolling its own colour palette / box style.
"""
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── palette ──────────────────────────────────────────────────────────────
ZONE_FILL   = "#ececec"
ZONE_EDGE   = "#000000"
BOX_FILL    = "#ffffff"
BOX_EDGE    = "#000000"
TEXT_MAIN   = "#000000"
TEXT_SUB    = "#3a3a3a"
ARROW_COLOR = "#1a1a1a"
GRID_COLOR  = "#bfbfbf"

# small, sparing accent set for category tags / legend dots only -- never
# used as a full box/zone fill
ACCENTS = ["#c9704f", "#5b8c5a", "#5b7fa6", "#8a6fa3", "#c9a23f"]

ZONE_LW = 1.6
BOX_LW  = 1.2


def apply_rcparams(base_fontsize=9.5):
    mpl.rcParams.update({
        "font.family": "Microsoft JhengHei",
        "font.size": base_fontsize,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def zone(ax, x, y, w, h, title_zh, title_en=None, fontsize=12.5):
    """Large light-gray rounded container representing a pipeline stage."""
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=ZONE_LW, edgecolor=ZONE_EDGE,
                        facecolor=ZONE_FILL, zorder=1)
    ax.add_patch(b)
    label = title_zh if not title_en else f"{title_zh}  {title_en}"
    ax.text(x + 0.18, y + h - 0.22, label, ha="left", va="top",
            fontsize=fontsize, fontweight="bold", color=TEXT_MAIN, zorder=2)
    return b


def box(ax, x, y, w, h, text, fontsize=9.3, zorder=3, align="center"):
    """White component box with black border and bold bilingual text."""
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=BOX_LW, edgecolor=BOX_EDGE,
                        facecolor=BOX_FILL, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=TEXT_MAIN, linespacing=1.35, zorder=zorder + 1,
            fontweight="bold" if align == "center" else "normal")
    return b


def arrow(ax, p1, p2, color=ARROW_COLOR, lw=1.6, style="-|>", mutation_scale=14, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mutation_scale,
                         linewidth=lw, color=color, linestyle=ls,
                         shrinkA=2, shrinkB=2, zorder=5)
    ax.add_patch(a)
    return a


def tag(ax, x, y, text, color, fontsize=7.5):
    """Small pastel category tag/pill, echoing the relation-type tags in
    fig_overview_pistgnn_white.png -- the only place saturated colour is
    allowed under this house style."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                       edgecolor="none"), zorder=6)
