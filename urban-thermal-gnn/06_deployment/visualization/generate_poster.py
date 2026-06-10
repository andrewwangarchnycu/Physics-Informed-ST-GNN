"""
generate_poster.py
Generates a high-definition, publication-ready research poster for:
"Physics-Informed Spatio-Temporal GNN for Urban Thermal Comfort (UTCI) Prediction"

Output: PI_ST_GNN_Research_Poster.png (300 DPI) and .pdf in the same directory.
Run:  python generate_poster.py
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Ellipse
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
PAL = {
    'bg_fig':    '#0D1117',
    'bg_panel':  '#161B22',
    'bg_card':   '#1C2128',
    'bg_card2':  '#21262D',
    'border':    '#30363D',
    'hdr_blue':  '#0D47A1',
    'hdr_teal':  '#004D40',
    'hdr_purple':'#4A148C',
    'hdr_orange':'#BF360C',
    'hdr_mid':   '#1A237E',
    'blue':      '#1565C0',
    'blue_lt':   '#90CAF9',
    'green':     '#2E7D32',
    'green_lt':  '#A5D6A7',
    'orange':    '#E65100',
    'orange_lt': '#FFCC80',
    'purple':    '#6A1B9A',
    'purple_lt': '#CE93D8',
    'cyan':      '#00ACC1',
    'text':      '#E6EDF3',
    'text_dim':  '#8B949E',
    'accent':    '#58A6FF',
    'node_bldg': '#546E7A',
    'node_air1': '#1565C0',
    'node_air2': '#DC3C3C',
    'node_tree': '#2E7D32',
    'shadow_e':  '#FDD835',
    'veget_e':   '#66BB6A',
    'conv_e':    '#26C6DA',
    'sem_e':     '#5C6BC0',
    'cont_e':    '#78909C',
}

REL_COLORS  = [PAL['shadow_e'], PAL['veget_e'], PAL['conv_e'], PAL['sem_e'], PAL['cont_e']]
REL_NAMES   = ['shadow', 'veg_et', 'convective', 'semantic', 'contiguity']
REL_LABELS  = ['Shadow', 'Veg_ET\n(ET)', 'Convective', 'Semantic', 'Contiguity']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'text.color': PAL['text'],
    'axes.facecolor': PAL['bg_panel'],
    'axes.edgecolor': PAL['border'],
    'figure.facecolor': PAL['bg_fig'],
    'axes.labelcolor': PAL['text'],
    'xtick.color': PAL['text_dim'],
    'ytick.color': PAL['text_dim'],
    'grid.color': '#21262D',
    'grid.alpha': 0.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'mathtext.fontset': 'dejavusans',
})


def make_utci_cmap():
    stops = [(0.00, '#4682B4'), (0.25, '#90EE90'),
             (0.55, '#FFFF66'), (0.80, '#FFA500'), (1.00, '#DC3C3C')]
    return LinearSegmentedColormap.from_list('utci', [(v, c) for v, c in stops])


UTCI_CMAP = make_utci_cmap()
UTCI_NORM = Normalize(vmin=18, vmax=38)


# ---------------------------------------------------------------------------
# FIGURE SETUP
# ---------------------------------------------------------------------------
def setup_figure():
    fig = plt.figure(figsize=(46.8, 33.1), facecolor=PAL['bg_fig'])
    master = GridSpec(3, 1, figure=fig,
                      height_ratios=[1.55, 0.26, 1.17],
                      hspace=0.035, left=0.01, right=0.99,
                      top=0.955, bottom=0.018)

    top_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=master[0],
                                     width_ratios=[3.0, 3.0, 4.0], wspace=0.025)
    mid_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=master[1], wspace=0.025)
    bot_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=master[2],
                                     width_ratios=[3.0, 4.0, 3.0], wspace=0.025)

    axes = {
        'tc1': fig.add_subplot(top_gs[0]),
        'tc2': fig.add_subplot(top_gs[1]),
        'tc3': fig.add_subplot(top_gs[2]),
        'mid': [fig.add_subplot(mid_gs[i]) for i in range(4)],
        'bc1': fig.add_subplot(bot_gs[0]),
        'bc2': fig.add_subplot(bot_gs[1]),
        'bc3': fig.add_subplot(bot_gs[2]),
    }
    for key, ax in axes.items():
        if key == 'mid':
            for a in ax:
                a.set_facecolor(PAL['bg_panel'])
                a.set_xlim(0, 1); a.set_ylim(0, 1)
                a.axis('off')
        else:
            ax.set_facecolor(PAL['bg_panel'])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')

    # Special: urban graph uses metre coords
    axes['tc2'].set_xlim(0, 80)
    axes['tc2'].set_ylim(0, 80)

    return fig, axes


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------
def _fbbox(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.02, zorder=3, alpha=1.0,
           clip=False):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad={radius}',
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        zorder=zorder, alpha=alpha, clip_on=clip)
    ax.add_patch(p)
    return p


def draw_section_header(ax, title, color, y_top=0.975, height=0.045):
    _fbbox(ax, 0.005, y_top - height, 0.990, height, fc=color, ec=color,
           radius=0.005, zorder=10)
    ax.text(0.5, y_top - height / 2, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=11,
            transform=ax.transData)


def draw_panel_card(ax, x, y, w, h, title, title_color, fontsize=9.5, zorder=5):
    _fbbox(ax, x, y, w, h, fc=PAL['bg_card'], ec=PAL['border'], lw=1.0,
           radius=0.008, zorder=zorder)
    # colored left accent bar
    ax.add_patch(Rectangle((x, y + h - 0.038), 0.006, 0.038,
                             facecolor=title_color, zorder=zorder + 1))
    ax.text(x + 0.016, y + h - 0.019, title, ha='left', va='center',
            fontsize=fontsize, fontweight='bold', color=title_color,
            zorder=zorder + 2)


def _arrow(ax, x0, y0, x1, y1, color=None, lw=2.0, style='simple',
           rad=0.0, zorder=8, alpha=0.9, hw=6, hl=6, tw=2):
    if color is None:
        color = PAL['accent']
    arrowstyle = f'Simple,tail_width={tw},head_width={hw},head_length={hl}'
    conn = f'arc3,rad={rad}' if rad != 0 else 'arc3,rad=0'
    p = FancyArrowPatch((x0, y0), (x1, y1),
                         arrowstyle=arrowstyle,
                         connectionstyle=conn,
                         color=color, linewidth=lw,
                         zorder=zorder, alpha=alpha)
    ax.add_patch(p)


# ---------------------------------------------------------------------------
# POSTER TITLE
# ---------------------------------------------------------------------------
def draw_poster_title(fig):
    fig.text(0.5, 0.982,
             'Physics-Informed Spatio-Temporal GNN for Urban Thermal Comfort (UTCI) Prediction',
             ha='center', va='top', fontsize=30, fontweight='bold',
             color=PAL['text'],
             path_effects=[pe.withStroke(linewidth=5, foreground=PAL['bg_fig'])])
    fig.text(0.5, 0.966,
             'PIN-ST-GNN  ·  Heterogeneous Urban Graph  ·  RGCN-LSTM  ·  NSGA-II Optimization  ·  NYCU',
             ha='center', va='top', fontsize=15, color=PAL['accent'])
    # horizontal rule
    line = Line2D([0.01, 0.99], [0.960, 0.960],
                  transform=fig.transFigure, color=PAL['border'], lw=1.0)
    fig.add_artist(line)


# ---------------------------------------------------------------------------
# TOP COLUMN 1 — DATA INPUTS
# ---------------------------------------------------------------------------
def _icon_building(ax, cx, cy, s, color):
    for i, (bw, bh) in enumerate([(s, s*0.5), (s*0.7, s*0.35)]):
        ax.add_patch(Rectangle((cx - bw/2, cy + i * s*0.35 * 0.9), bw, bh,
                                 facecolor=color, edgecolor='white', lw=0.5,
                                 alpha=0.85, zorder=12))
    # windows
    for xi in [-0.3, 0, 0.3]:
        ax.add_patch(Rectangle((cx + xi*s - s*0.08, cy + s*0.08), s*0.12, s*0.12,
                                 facecolor='white', alpha=0.5, zorder=13))


def _icon_tree(ax, cx, cy, s, color):
    tri = Polygon([[cx, cy + s], [cx - s*0.5, cy], [cx + s*0.5, cy]],
                   closed=True, facecolor=color, edgecolor='white', lw=0.5,
                   alpha=0.85, zorder=12)
    ax.add_patch(tri)
    ax.add_patch(Rectangle((cx - s*0.1, cy - s*0.35), s*0.2, s*0.35,
                             facecolor='#795548', zorder=12))


def _icon_sensor(ax, cx, cy, s, color):
    ax.add_patch(Circle((cx, cy + s*0.15), s*0.28, facecolor=color,
                          edgecolor='white', lw=0.5, alpha=0.85, zorder=12))
    ax.plot([cx, cx], [cy - s*0.2, cy], color='white', lw=1.5, zorder=13)
    ax.plot([cx - s*0.2, cx + s*0.2], [cy - s*0.2, cy - s*0.2],
            color='white', lw=1.5, zorder=13)


def _icon_cloud(ax, cx, cy, s, color):
    for dx, dy, r in [(-s*0.2, 0, s*0.22), (s*0.15, 0, s*0.20),
                       (0, s*0.15, s*0.20)]:
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor=color,
                              edgecolor='white', lw=0.4, alpha=0.8, zorder=12))


def _icon_gear(ax, cx, cy, s, color):
    ax.add_patch(Circle((cx, cy), s*0.3, facecolor=color,
                          edgecolor='white', lw=0.5, alpha=0.85, zorder=12))
    ax.add_patch(Circle((cx, cy), s*0.13, facecolor=PAL['bg_card'],
                          zorder=13))
    # teeth
    for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
        tx = cx + np.cos(ang) * s*0.32
        ty = cy + np.sin(ang) * s*0.32
        ax.add_patch(Rectangle((tx - s*0.05, ty - s*0.05), s*0.10, s*0.10,
                                 facecolor=color, alpha=0.85, zorder=12))


def _icon_chart(ax, cx, cy, s, color):
    heights = [0.5, 0.8, 0.65]
    for i, bh in enumerate(heights):
        ax.add_patch(Rectangle((cx - s*0.4 + i*s*0.28, cy), s*0.22, s*bh,
                                 facecolor=color, edgecolor='white', lw=0.4,
                                 alpha=0.85, zorder=12))


ICON_FN = {
    'building': _icon_building,
    'tree':     _icon_tree,
    'sensor':   _icon_sensor,
    'cloud':    _icon_cloud,
    'gear':     _icon_gear,
    'chart':    _icon_chart,
}


def draw_data_icon(ax, cx, cy, icon_type, color, s=0.028):
    ICON_FN[icon_type](ax, cx, cy, s, color)


def draw_data_inputs_panel(ax):
    panels = [
        # (y_bot, y_top, title, color, items[(icon, label, sublabel)])
        (0.665, 0.966,
         'GIS & GEOMETRY DATA', PAL['blue'],
         [('building', 'OSM Building Footprints', 'Polygon geometry, height, floors'),
          ('tree',     'Tree Canopy Data',          'Meta HighRes GeoTIFF, 1 m res.'),
          ('gear',     'Urban Geometry Sampler',    '80×80 m site, FAR/BCR constraints')]),
        (0.335, 0.652,
         'ENVIRONMENTAL & IOT SENSOR DATA', PAL['green'],
         [('cloud',   'CWB Meteorological Data',   'Ta / RH / Ws / Wd / Rain (hourly)'),
          ('sensor',  'MoEnv IoT Time-series',     'Ta / RH minute→hourly, 5 km radius'),
          ('chart',   'EPW Weather Forcing',        'GHI / DHI / DNI, TMY annual')]),
        (0.012, 0.322,
         'SIMULATION & GROUND TRUTH', PAL['orange'],
         [('gear',   'LBT Batch Simulation Engine', 'UTCI / MRT / SVF per scenario'),
          ('chart',  'Sensing Calibration Pipeline', 'z₀ · albedo · ta_bias (diff. evol.)')]),
    ]
    for (yb, yt, title, color, items) in panels:
        _fbbox(ax, 0.010, yb, 0.980, yt - yb, fc=PAL['bg_card'],
               ec=color, lw=1.4, radius=0.005, zorder=4)
        # header bar
        ax.add_patch(Rectangle((0.010, yt - 0.046), 0.980, 0.046,
                                 facecolor=color, alpha=0.88, zorder=5))
        ax.text(0.50, yt - 0.023, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', zorder=6)
        # items
        row_h = (yt - yb - 0.052) / len(items)
        for idx, (icon, label, sub) in enumerate(items):
            iy = yb + (len(items) - 1 - idx) * row_h + row_h * 0.5
            draw_data_icon(ax, 0.085, iy - 0.008, icon, color, s=0.030)
            ax.text(0.160, iy + 0.006, label, ha='left', va='center',
                    fontsize=8.5, fontweight='bold', color=PAL['text'], zorder=7)
            ax.text(0.160, iy - 0.018, sub, ha='left', va='center',
                    fontsize=7.5, color=PAL['text_dim'], zorder=7)
    draw_section_header(ax, 'MULTIMODAL DATA INPUTS & PREPROCESSING',
                        PAL['hdr_blue'])


# ---------------------------------------------------------------------------
# TOP COLUMN 2 — URBAN GRAPH
# ---------------------------------------------------------------------------
BUILDINGS = [
    (5,  50, 20, 22, 'B1'),
    (55, 52, 20, 22, 'B2'),
    (8,   8, 15, 18, 'B3'),
    (56,  8, 18, 20, 'B4'),
    (30, 30, 15, 16, 'B5'),
]
AIR_NODES = [
    (28, 68, 26),
    (42, 68, 23),
    (28, 46, 31),
    (42, 46, 29),
    (28, 24, 25),
    (42, 24, 22),
    (62, 35, 34),
    (15, 38, 21),
]
TREE_NODES = [(25, 75), (40, 75), (55, 75), (25, 15), (55, 15)]

EDGES = {
    'shadow':     [((28,68),(28,46)), ((42,68),(42,46))],
    'veg_et':     [((25,75),(28,68)), ((40,75),(42,68)), ((55,75),(62,35))],
    'convective': [((28,68),(42,68)), ((28,46),(42,46))],
    'semantic':   [((5+10, 50+11),(55+10, 52+11)),
                   ((8+7.5, 8+9),(30+7.5, 30+8))],
    'contiguity': [((28,68),(42,68)), ((28,46),(42,46)),
                   ((28,68),(28,46)), ((42,68),(42,46)),
                   ((28,24),(42,24)), ((28,46),(28,24)),
                   ((15,38),(28,46)), ((62,35),(42,46))],
}
EDGE_STYLE = {
    'contiguity': (PAL['cont_e'], 1.4, (4,3)),
    'semantic':   (PAL['sem_e'],  1.8, (6,3)),
    'convective': (PAL['conv_e'], 2.0, 'solid'),
    'veg_et':     (PAL['veget_e'],1.8, 'solid'),
    'shadow':     (PAL['shadow_e'],2.2,'solid'),
}


def draw_urban_graph_panel(ax):
    # background
    ax.set_facecolor(PAL['bg_card2'])
    for spine in ax.spines.values():
        spine.set_edgecolor(PAL['border'])
        spine.set_linewidth(1.2)

    # edges (draw first)
    for rel in ['contiguity','semantic','convective','veg_et','shadow']:
        color, lw, ls = EDGE_STYLE[rel]
        for (p0, p1) in EDGES[rel]:
            x0, y0 = p0; x1, y1 = p1
            ls_arg = ls if isinstance(ls, str) else (0, ls)
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                    linestyle=ls_arg, alpha=0.75, zorder=2)

    # buildings
    for (bx, by, bw, bh, lbl) in BUILDINGS:
        ax.add_patch(Rectangle((bx, by), bw, bh,
                                 facecolor=PAL['node_bldg'], edgecolor='#90A4AE',
                                 lw=1.0, alpha=0.90, zorder=3))
        ax.text(bx + bw/2, by + bh/2, lbl, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=4)

    # trees
    for (tx, ty) in TREE_NODES:
        ax.add_patch(Circle((tx, ty), 2.8, facecolor=PAL['node_tree'],
                              edgecolor='#A5D6A7', lw=0.8, alpha=0.85, zorder=4))
        ax.plot([tx-1.6, tx+1.6], [ty, ty], color='#A5D6A7', lw=1.0, zorder=5)
        ax.plot([tx, tx], [ty-1.6, ty+1.6], color='#A5D6A7', lw=1.0, zorder=5)

    # air sensor nodes
    for (nx, ny, utci) in AIR_NODES:
        c = UTCI_CMAP(UTCI_NORM(utci))
        ax.add_patch(Circle((nx, ny), 2.8, facecolor='white',
                              edgecolor='white', lw=0, zorder=5))
        ax.add_patch(Circle((nx, ny), 2.4, facecolor=c,
                              edgecolor='white', lw=0.8, zorder=6))
        ax.text(nx, ny, f'{utci}', ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=7)

    # node labels annotation
    ax.text(5, 76, 'Air Node\n(Ta/RH/UTCI)', fontsize=7.5, color=PAL['text'],
            va='top', zorder=8)
    ax.text(0.5, 74, 'Object Node\n(Buildings)', fontsize=7.5, color='#90A4AE',
            va='top', zorder=8)
    ax.text(41, 76, 'Tree Node\n(Canopy)', fontsize=7.5, color=PAL['green_lt'],
            va='top', zorder=8)

    # legend
    legend_x, legend_y = 0.5, 1.5
    lh = 3.2
    node_handles = [
        mpatches.Patch(fc=UTCI_CMAP(UTCI_NORM(28)), ec='white', lw=0.5,
                       label='Air Node (UTCI val)'),
        mpatches.Patch(fc=PAL['node_bldg'], ec='#90A4AE', lw=0.5,
                       label='Object Node (Building)'),
        mpatches.Patch(fc=PAL['node_tree'], ec='#A5D6A7', lw=0.5,
                       label='Tree Node (Canopy)'),
    ]
    edge_handles = [
        Line2D([0],[0], color=PAL['shadow_e'], lw=2, label='Shadow'),
        Line2D([0],[0], color=PAL['veget_e'],  lw=2, label='Veg_ET'),
        Line2D([0],[0], color=PAL['conv_e'],   lw=2, label='Convective'),
        Line2D([0],[0], color=PAL['sem_e'],    lw=2, linestyle=(0,(6,3)), label='Semantic'),
        Line2D([0],[0], color=PAL['cont_e'],   lw=2, linestyle=(0,(4,3)), label='Contiguity'),
    ]
    leg = ax.legend(handles=node_handles + edge_handles,
                    loc='lower left', bbox_to_anchor=(0.01, 0.01),
                    fontsize=7.5, framealpha=0.85,
                    facecolor=PAL['bg_card'], edgecolor=PAL['border'],
                    labelcolor=PAL['text'], ncol=2, handlelength=2.0,
                    handleheight=1.0, borderpad=0.5, labelspacing=0.4)

    # UTCI colorbar (inset)
    cax = ax.inset_axes([0.72, 0.01, 0.265, 0.14])
    sm = plt.cm.ScalarMappable(cmap=UTCI_CMAP, norm=UTCI_NORM)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('UTCI (°C)', fontsize=7, color=PAL['text'])
    cb.ax.tick_params(labelsize=6.5, colors=PAL['text_dim'])
    cb.ax.xaxis.set_tick_params(color=PAL['text_dim'])

    # scale bar
    ax.plot([60, 70], [3.5, 3.5], color=PAL['text_dim'], lw=2, zorder=8)
    ax.text(65, 5.0, '10 m', ha='center', va='bottom',
            fontsize=7, color=PAL['text_dim'], zorder=8)

    draw_section_header(ax, 'HETEROGENEOUS URBAN GRAPH MODEL', PAL['hdr_teal'])
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# TOP COLUMN 3 — METHODOLOGY A
# ---------------------------------------------------------------------------
def draw_tier1_relations(ax, yb, yt):
    _fbbox(ax, 0.012, yb, 0.976, yt - yb, fc=PAL['bg_card'], ec=PAL['sem_e'],
           lw=1.0, radius=0.006, zorder=4)
    ax.text(0.5, yt - 0.025, 'MULTI-RELATIONAL EDGES', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PAL['sem_e'], zorder=5)
    bw = 0.170; gap = 0.016; x0 = 0.020
    ymid = (yb + yt) / 2 - 0.01
    bh = 0.062
    for i, (color, lbl) in enumerate(zip(REL_COLORS, REL_LABELS)):
        bx = x0 + i * (bw + gap)
        _fbbox(ax, bx, ymid - bh/2, bw, bh, fc=color, ec='white',
               lw=0.6, radius=0.005, alpha=0.88, zorder=5)
        # swatch line
        ax.plot([bx + 0.015, bx + bw - 0.015], [ymid + 0.013, ymid + 0.013],
                color='white', lw=2, zorder=6)
        ax.text(bx + bw/2, ymid - 0.012, lbl, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#1C2128', zorder=6,
                linespacing=1.1)


def _mini_graph(ax, cx, cy, size, shadow_pair):
    s = size
    pts = [(cx - s, cy + s), (cx, cy + s),
           (cx - s, cy),     (cx, cy),
           (cx + s, cy - s), (cx + s, cy)]
    # contiguity edges
    for (i, j) in [(0,1),(0,2),(1,3),(2,3),(4,5)]:
        x0,y0 = pts[i]; x1,y1 = pts[j]
        ax.plot([x0,x1],[y0,y1], color=PAL['cont_e'], lw=1.0, alpha=0.6, zorder=3)
    # shadow edge
    i, j = shadow_pair
    x0,y0 = pts[i]; x1,y1 = pts[j]
    ax.plot([x0,x1],[y0,y1], color=PAL['shadow_e'], lw=2.2, alpha=0.9, zorder=4)
    # rectangle (building)
    ax.add_patch(Rectangle((cx - s*0.4, cy + s*0.5), s*0.7, s*0.7,
                             facecolor=PAL['node_bldg'], edgecolor='#90A4AE',
                             lw=0.7, alpha=0.85, zorder=4))
    # nodes
    for px, py in pts:
        c = UTCI_CMAP(UTCI_NORM(np.random.default_rng(
            int(abs(px*100+py*100))).uniform(22, 34)))
        ax.add_patch(Circle((px, py), s*0.13, facecolor=c,
                              edgecolor='white', lw=0.5, zorder=5))


def draw_tier2_temporal(ax, yb, yt):
    _fbbox(ax, 0.012, yb, 0.976, yt - yb, fc=PAL['bg_card'], ec=PAL['conv_e'],
           lw=1.0, radius=0.006, zorder=4)
    ax.text(0.5, yt - 0.025, 'DYNAMIC SPATIOTEMPORAL GRAPH', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PAL['conv_e'], zorder=5)
    ymid = (yb + yt) / 2 - 0.005
    cx_list = [0.20, 0.50, 0.80]
    labels  = ['T − 1\n(09:00)', 'T\n(12:00, Solar Noon)', 'T + 1\n(15:00)']
    shadow_pairs = [(0, 2), (1, 3), (4, 5)]
    for cx, lbl, sp in zip(cx_list, labels, shadow_pairs):
        _mini_graph(ax, cx, ymid, 0.065, sp)
        ax.text(cx, ymid - 0.095, lbl, ha='center', va='top',
                fontsize=7.5, color=PAL['text_dim'], zorder=6, linespacing=1.2)
    # arrows between snapshots
    for x_from, x_to in [(0.295, 0.380), (0.595, 0.680)]:
        _arrow(ax, x_from, ymid, x_to, ymid, color=PAL['accent'],
               lw=1.5, hw=5, hl=5, tw=1.5, zorder=7)


def draw_tier3_heatmap(ax, yb, yt):
    _fbbox(ax, 0.012, yb, 0.976, yt - yb, fc=PAL['bg_card'], ec=PAL['orange'],
           lw=1.0, radius=0.006, zorder=4)
    ax.text(0.5, yt - 0.024, 'SIMULATED GROUND TRUTH (LBT UTCI Heatmap)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PAL['orange_lt'], zorder=5)

    rng = np.random.default_rng(42)
    base = (np.linspace(22, 34, 20).reshape(1,20) +
            np.linspace(0, 3, 20).reshape(20,1))
    utci_map = np.clip(base + rng.normal(0, 1.2, (20, 20)), 18, 38)

    inner_ax = ax.inset_axes([0.08, yb + 0.005, 0.84, yt - yb - 0.060],
                              transform=ax.transData)
    inner_ax.set_axis_off()
    im_ax = ax.inset_axes([0.10, yb + 0.01, 0.76, yt - yb - 0.065],
                           transform=ax.transData)
    im = im_ax.imshow(utci_map, aspect='auto', cmap=UTCI_CMAP,
                       norm=UTCI_NORM, origin='lower', rasterized=True)
    im_ax.set_xlabel('Site Width (m)', fontsize=7, color=PAL['text_dim'],
                      labelpad=2)
    im_ax.set_ylabel('Site Depth (m)', fontsize=7, color=PAL['text_dim'],
                      labelpad=2)
    im_ax.set_xticks([0, 10, 19]); im_ax.set_xticklabels(['0','40','80'],
                                                            fontsize=6.5)
    im_ax.set_yticks([0, 10, 19]); im_ax.set_yticklabels(['0','40','80'],
                                                            fontsize=6.5)
    im_ax.tick_params(colors=PAL['text_dim'], length=2)
    for s in im_ax.spines.values():
        s.set_edgecolor(PAL['border'])

    cax = ax.inset_axes([0.875, yb + 0.02, 0.022, yt - yb - 0.070],
                         transform=ax.transData)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('UTCI (°C)', fontsize=6.5, color=PAL['text'])
    cb.ax.tick_params(labelsize=6, colors=PAL['text_dim'])


def draw_methodology_a_panel(ax):
    draw_tier1_relations(ax, yb=0.700, yt=0.958)
    _arrow(ax, 0.50, 0.697, 0.50, 0.662, color=PAL['accent'],
           lw=2, hw=7, hl=7, tw=2, zorder=9)
    draw_tier2_temporal(ax, yb=0.360, yt=0.660)
    _arrow(ax, 0.50, 0.357, 0.50, 0.322, color=PAL['accent'],
           lw=2, hw=7, hl=7, tw=2, zorder=9)
    draw_tier3_heatmap(ax, yb=0.012, yt=0.320)
    draw_section_header(ax, 'METHODOLOGY A: PHYSICS-INFORMED GRAPH CONSTRUCTION',
                        PAL['hdr_purple'])


# ---------------------------------------------------------------------------
# MIDDLE BAND — FORMULAS
# ---------------------------------------------------------------------------
FORMULAS = [
    ('R-GCN Propagation (Eq. 2)',
     r'$\mathbf{h}_i^{(l+1)} = \sigma\!\left(\mathrm{LN}\!\left('
     r'\sum_r\!\sum_{j\in\mathcal{N}_r(i)}'
     r'\dfrac{\mathbf{W}_r\mathbf{h}_j^{(l)}}{|\mathcal{N}_r|}+'
     r'\mathbf{W}_{\mathrm{self}}\mathbf{h}_i^{(l)}\right)\!\right)$'),
    ('Physics-Informed Loss',
     r'$\mathcal{L}=\mathcal{L}_{\mathrm{data}}+\lambda_1\mathcal{L}_{\mathrm{rad}}'
     r'+\lambda_2\mathcal{L}_{\mathrm{temp}}+\lambda_3\mathcal{L}_{\mathrm{wind}}$'
     '\n'
     r'$\lambda_1{=}0.1\quad\lambda_2{=}0.05\quad\lambda_3{=}0.05$'),
    ('Mean Radiant Temperature',
     r'$\mathrm{MRT}=T_a+\Delta T_{\mathrm{mrt}}$'
     '\n'
     r'$\Delta T_{\mathrm{mrt}}=(I_{\mathrm{dir}}+I_{\mathrm{diff}}+I_{\mathrm{ref}})/100$'),
    ('Sky View Factor',
     r'$\mathrm{SVF}=1-\overline{\sin\!\left(\theta_{\mathrm{elev}}\right)}$'
     '\n'
     r'$\theta_{\mathrm{elev}}$: elevation angle to building envelopes'),
]

FORMULA_COLORS = [PAL['blue'], PAL['orange'], PAL['green'], PAL['purple']]


def draw_formulas_band(mid_axes):
    for ax, (title, formula), color in zip(mid_axes, FORMULAS, FORMULA_COLORS):
        _fbbox(ax, 0.008, 0.05, 0.984, 0.90, fc=PAL['bg_card'], ec=color,
               lw=1.2, radius=0.04, zorder=3)
        ax.add_patch(Rectangle((0.008, 0.74), 0.984, 0.21,
                                 facecolor=color, alpha=0.80, zorder=4))
        ax.text(0.50, 0.845, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white', zorder=5)
        ax.text(0.50, 0.42, formula, ha='center', va='center',
                fontsize=10.5, color=PAL['text'], zorder=5, linespacing=1.6)


# ---------------------------------------------------------------------------
# BOTTOM COLUMN 1 — FEATURE ENCODING
# ---------------------------------------------------------------------------
def draw_mlp_block(ax, cx, yb, yt, in_dim, hid_dim, out_dim, label, color):
    bw = 0.30; x = cx - bw/2
    total_h = yt - yb
    lh = total_h / 3.8
    rects = [
        (in_dim,  f'Input  ×{in_dim}',  0.85),
        (hid_dim, f'Linear → LN → ReLU\n({hid_dim})', 0.85),
        (out_dim, f'Linear → ReLU\n({out_dim})',       0.85),
    ]
    y_cur = yb + 0.005
    prev_y = None
    for (dim, lbl, alpha) in rects:
        rh = lh * 0.82
        _fbbox(ax, x, y_cur, bw, rh, fc=color, ec='white',
               lw=0.6, radius=0.006, alpha=alpha, zorder=6)
        ax.text(cx, y_cur + rh/2, lbl, ha='center', va='center',
                fontsize=7.0, color='white', fontweight='bold', zorder=7,
                linespacing=1.2)
        if prev_y is not None:
            _arrow(ax, cx, prev_y, cx, y_cur, color='white',
                   lw=1, hw=4, hl=4, tw=1, zorder=8)
        prev_y = y_cur + rh
        y_cur += lh
    # label above
    ax.text(cx, yt + 0.005, label, ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color=color, zorder=7)


def draw_feature_encoding_panel(ax):
    # NODE ENCODERS card
    _fbbox(ax, 0.010, 0.540, 0.980, 0.415, fc=PAL['bg_card'],
           ec=PAL['blue'], lw=1.0, radius=0.005, zorder=4)
    ax.text(0.50, 0.935, 'NODE ENCODERS', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PAL['blue_lt'], zorder=5)

    draw_mlp_block(ax, cx=0.285, yb=0.560, yt=0.905,
                   in_dim=7, hid_dim=128, out_dim=128,
                   label='Object MLP\n(7 → 128)', color=PAL['blue'])
    draw_mlp_block(ax, cx=0.720, yb=0.560, yt=0.905,
                   in_dim=10, hid_dim=128, out_dim=128,
                   label='Air MLP\n(10 → 128)', color=PAL['green'])

    # merge arrow
    _arrow(ax, 0.285, 0.555, 0.500, 0.540, color=PAL['accent'],
           lw=1.5, hw=4, hl=4, tw=1, rad=0.3, zorder=9)
    _arrow(ax, 0.720, 0.555, 0.500, 0.540, color=PAL['accent'],
           lw=1.5, hw=4, hl=4, tw=1, rad=-0.3, zorder=9)
    ax.text(0.50, 0.528, '→ h_all  (N_obj+N_air, 128)',
            ha='center', va='center', fontsize=8, color=PAL['text_dim'], zorder=5)

    # GLOBAL CONTEXT card
    _fbbox(ax, 0.010, 0.238, 0.980, 0.283, fc=PAL['bg_card'],
           ec=PAL['purple'], lw=1.0, radius=0.005, zorder=4)
    ax.text(0.50, 0.498, 'GLOBAL CONTEXT ENCODER', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PAL['purple_lt'], zorder=5)

    # two branches
    for cx, label, in_d, color in [
        (0.27, 'Env MLP\n(7→64→128)', 7,  PAL['orange']),
        (0.73, 'Time MLP\n(2→64→64)',  2,  PAL['cyan']),
    ]:
        _fbbox(ax, cx - 0.16, 0.258, 0.32, 0.105, fc=color, ec='white',
               lw=0.6, radius=0.005, alpha=0.82, zorder=6)
        ax.text(cx, 0.311, label, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=7,
                linespacing=1.2)
        _arrow(ax, cx, 0.357, 0.50, 0.390, color='white',
               lw=1, hw=4, hl=4, tw=1,
               rad=0.25 if cx < 0.5 else -0.25, zorder=8)

    # concat box
    _fbbox(ax, 0.370, 0.380, 0.260, 0.060, fc=PAL['purple'], ec='white',
           lw=0.6, radius=0.005, alpha=0.85, zorder=6)
    ax.text(0.50, 0.410, '[cat] → ctx (192)', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=7)

    # dim summary table
    _fbbox(ax, 0.010, 0.010, 0.980, 0.205, fc=PAL['bg_card2'],
           ec=PAL['border'], lw=0.8, radius=0.005, zorder=4)
    rows = [
        'DIM_OBJECT = 7    DIM_AIR = 10    hidden_dim = 128',
        'ENV_DIM = 7       TIME_DIM = 2    ctx_dim = 192',
        'Timesteps T = 11  (08:00 – 18:00 hourly)',
    ]
    for i, row in enumerate(rows):
        ax.text(0.50, 0.178 - i * 0.060, row, ha='center', va='center',
                fontsize=8, color=PAL['text_dim'],
                fontfamily='monospace', zorder=5)

    draw_section_header(ax, 'MULTI-MODAL FEATURE ENCODING', PAL['hdr_orange'])


# ---------------------------------------------------------------------------
# BOTTOM COLUMN 2 — SPATIOTEMPORAL LEARNING
# ---------------------------------------------------------------------------
def draw_rgcn_block(ax, yb, yt, layer_num):
    bdr_colors = [PAL['blue_lt'], PAL['accent'], PAL['purple_lt']]
    color = bdr_colors[(layer_num - 1) % 3]
    _fbbox(ax, 0.015, yb, 0.970, yt - yb, fc=PAL['bg_card'],
           ec=color, lw=1.3, radius=0.006, zorder=5)
    ax.text(0.035, yt - 0.022,
            f'RGCN Layer {layer_num}   ·   in=128  out=128  (PReLU + LayerNorm)',
            ha='left', va='center', fontsize=8.5, fontweight='bold',
            color=color, zorder=6)

    ymid = (yb + yt) / 2 - 0.008
    rw, rh = 0.116, 0.058
    x0, gap = 0.048, 0.014
    for i, (rc, rl) in enumerate(zip(REL_COLORS, ['W_shadow','W_veg_et',
                                                    'W_conv','W_sem','W_cont'])):
        rx = x0 + i * (rw + gap)
        _fbbox(ax, rx, ymid - rh/2, rw, rh, fc=rc, ec='white',
               lw=0.5, radius=0.004, alpha=0.88, zorder=6)
        ax.text(rx + rw/2, ymid, rl, ha='center', va='center',
                fontsize=6.5, color='#1C2128', fontweight='bold', zorder=7)

    # W_self
    ws_x = x0 + 5 * (rw + gap) + 0.008
    _fbbox(ax, ws_x, ymid - rh/2, rw*0.7, rh, fc='#B0BEC5', ec='white',
           lw=0.5, radius=0.004, alpha=0.80, zorder=6)
    ax.text(ws_x + rw*0.35, ymid, 'W_self', ha='center', va='center',
            fontsize=6.5, color='#1C2128', fontweight='bold', zorder=7)

    # LN badge
    ln_x = 0.895
    _fbbox(ax, ln_x, ymid - 0.020, 0.070, 0.040, fc='#37474F',
           ec='#90CAF9', lw=0.8, radius=0.004, zorder=6)
    ax.text(ln_x + 0.035, ymid, 'LN', ha='center', va='center',
            fontsize=8, color='#90CAF9', fontweight='bold', zorder=7)

    # residual arrow
    p = FancyArrowPatch((0.010, yb + 0.025), (0.010, yt - 0.025),
                         arrowstyle='Simple,tail_width=1.5,head_width=5,head_length=4',
                         connectionstyle='arc3,rad=-0.55',
                         color=PAL['cont_e'], alpha=0.55, zorder=5)
    ax.add_patch(p)
    ax.text(0.001, (yb + yt)/2, 'res', ha='center', va='center',
            fontsize=6.5, color=PAL['cont_e'], rotation=90, zorder=6)


def draw_lstm_unrolled(ax, yb, yt):
    _fbbox(ax, 0.010, yb, 0.980, yt - yb, fc=PAL['bg_card'],
           ec=PAL['green'], lw=1.0, radius=0.006, zorder=4)
    ax.text(0.50, yt - 0.020, 'TEMPORAL LSTM  (RGCN h₀ warm-start)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PAL['green_lt'], zorder=5)

    hours = list(range(8, 19))  # 8 to 18 inclusive = 11 steps
    n = len(hours)
    x0, x1 = 0.030, 0.970
    cell_spacing = (x1 - x0) / n
    cell_w = cell_spacing * 0.68
    y_cell = (yb + yt) / 2 - 0.012
    cell_h = (yt - yb) * 0.28

    for i, hr in enumerate(hours):
        cx = x0 + i * cell_spacing + cell_w/2
        # context injection arrow
        _arrow(ax, cx, yt - 0.065, cx, y_cell + cell_h + 0.002,
               color=PAL['purple'], lw=1, hw=3, hl=3, tw=0.8, zorder=7)
        # cell box
        _fbbox(ax, cx - cell_w/2, y_cell, cell_w, cell_h,
               fc='#1B2B3A', ec=PAL['green'], lw=0.9, radius=0.004, zorder=5)
        ax.text(cx, y_cell + cell_h/2, f'h_{i}', ha='center', va='center',
                fontsize=7, color=PAL['green_lt'], fontweight='bold', zorder=6)
        # time label (every other)
        if i % 2 == 0:
            ax.text(cx, y_cell - 0.028, f'{hr}:00', ha='center', va='top',
                    fontsize=7, color=PAL['text_dim'], zorder=6)
        # horizontal arrow to next cell
        if i < n - 1:
            nx = x0 + (i+1) * cell_spacing - cell_w/2
            _arrow(ax, cx + cell_w/2, y_cell + cell_h/2,
                   nx, y_cell + cell_h/2,
                   color=PAL['green_lt'], lw=1, hw=3, hl=3, tw=0.8, zorder=7)

    # ctx label
    ax.text(0.50, yt - 0.038, '↓ Global Context (ctx, 192-dim) injected at each step',
            ha='center', va='center', fontsize=7.5, color=PAL['purple_lt'], zorder=6)

    # input/output
    ax.annotate('← x_t\n(RGCN out)', xy=(x0, y_cell + cell_h/2),
                 xytext=(x0 - 0.005, y_cell + cell_h/2),
                 fontsize=7, color=PAL['text_dim'], ha='right', va='center', zorder=7)
    ax.annotate('→ h_11\n(Output MLP)', xy=(x1, y_cell + cell_h/2),
                 xytext=(x1 + 0.005, y_cell + cell_h/2),
                 fontsize=7, color=PAL['text_dim'], ha='left', va='center', zorder=7)


def draw_spatiotemporal_panel(ax):
    # RGCN blocks
    rgcn_regions = [(0.640, 0.755), (0.758, 0.873), (0.876, 0.958)]
    for i, (yb, yt) in enumerate(rgcn_regions):
        draw_rgcn_block(ax, yb, yt, layer_num=i+1)
        if i < 2:
            _arrow(ax, 0.50, yb - 0.003, 0.50, rgcn_regions[i+1][1] + 0.003,
                   color=PAL['accent'], lw=1.5, hw=4, hl=4, tw=1, zorder=9)

    # section label
    ax.text(0.50, 0.618, 'RELATIONAL GCN BLOCKS  (3 layers, hidden_dim=128)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PAL['accent'], zorder=5)
    _arrow(ax, 0.50, 0.600, 0.50, 0.565, color=PAL['accent'],
           lw=2, hw=6, hl=6, tw=1.5, zorder=9)

    # LSTM
    draw_lstm_unrolled(ax, yb=0.015, yt=0.590)

    draw_section_header(ax, 'SPATIOTEMPORAL GRAPH LEARNING', PAL['hdr_orange'])


# ---------------------------------------------------------------------------
# BOTTOM COLUMN 3 — OUTPUT & OPTIMIZATION
# ---------------------------------------------------------------------------
UTCI_HOURLY = [22, 24, 27, 30, 33, 35, 34, 32, 29, 26, 23]


def draw_output_panel(ax):
    # UTCI PREDICTION HEAD
    _fbbox(ax, 0.010, 0.680, 0.980, 0.268, fc=PAL['bg_card'],
           ec=PAL['orange'], lw=1.0, radius=0.005, zorder=4)
    ax.text(0.50, 0.928, 'UTCI PREDICTION HEAD', ha='center', va='center',
            fontsize=9, fontweight='bold', color=PAL['orange_lt'], zorder=5)

    # MLP diagram
    layers = [(256,'LSTM\nout'), (256,'FC 256\nReLU'), (11,'OutputMLP\n(11 steps)')]
    lx = [0.15, 0.42, 0.695]
    ly = 0.808
    lh = 0.068; lw_box = 0.22
    for i, ((dim, lbl), x) in enumerate(zip(layers, lx)):
        _fbbox(ax, x - lw_box/2, ly - lh/2, lw_box, lh,
               fc=PAL['orange'], ec='white', lw=0.5, radius=0.004,
               alpha=0.85, zorder=6)
        ax.text(x, ly, lbl, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=7,
                linespacing=1.2)
        if i < len(lx)-1:
            _arrow(ax, x + lw_box/2, ly, lx[i+1] - lw_box/2, ly,
                   color='white', lw=1, hw=4, hl=4, tw=1, zorder=8)

    # 11 output squares
    sq_w = 0.062; sq_h = 0.048; sq_y = 0.695
    x_start = 0.50 - (11 * sq_w + 10 * 0.004) / 2
    hours = list(range(8, 19))
    for i, (hr, val) in enumerate(zip(hours, UTCI_HOURLY)):
        sx = x_start + i * (sq_w + 0.004)
        c = UTCI_CMAP(UTCI_NORM(val))
        _fbbox(ax, sx, sq_y, sq_w, sq_h, fc=c, ec='white', lw=0.4,
               radius=0.003, zorder=6)
        ax.text(sx + sq_w/2, sq_y + sq_h/2, f'{val}',
                ha='center', va='center', fontsize=5.8, color='white',
                fontweight='bold', zorder=7)
        if i % 2 == 0:
            ax.text(sx + sq_w/2, sq_y - 0.012, f'{hr}h',
                    ha='center', va='top', fontsize=5.8,
                    color=PAL['text_dim'], zorder=7)
    _arrow(ax, 0.695 + lw_box/2, ly, 0.50, sq_y + sq_h + 0.006,
           color=PAL['orange_lt'], lw=1, hw=4, hl=4, tw=1, rad=-0.2, zorder=8)

    # SPATIAL COMFORT MAP
    _fbbox(ax, 0.010, 0.385, 0.980, 0.278, fc=PAL['bg_card'],
           ec=PAL['cyan'], lw=1.0, radius=0.005, zorder=4)
    ax.text(0.50, 0.642, 'SPATIAL COMFORT MAP (UTCI Heatmap)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PAL['cyan'], zorder=5)

    rng = np.random.default_rng(42)
    base = (np.linspace(22, 34, 20).reshape(1,20) +
            np.linspace(0,3,20).reshape(20,1))
    utci_map = np.clip(base + rng.normal(0, 1.2, (20,20)), 18, 38)

    hm_ax = ax.inset_axes([0.08, 0.398, 0.72, 0.218], transform=ax.transData)
    im = hm_ax.imshow(utci_map, aspect='auto', cmap=UTCI_CMAP,
                       norm=UTCI_NORM, origin='lower', rasterized=True)
    hm_ax.set_axis_off()

    cb_ax = ax.inset_axes([0.815, 0.400, 0.025, 0.210], transform=ax.transData)
    cb = plt.colorbar(im, cax=cb_ax)
    cb.set_label('UTCI °C', fontsize=6.5, color=PAL['text'])
    cb.ax.tick_params(labelsize=6, colors=PAL['text_dim'])

    ax.text(0.50, 0.392, 'GIS-Projected UTCI Distribution  |  Site 80×80 m',
            ha='center', va='center', fontsize=7.5, color=PAL['text_dim'], zorder=5)

    # NSGA-II OPTIMIZATION
    _fbbox(ax, 0.010, 0.012, 0.980, 0.355, fc=PAL['bg_card'],
           ec=PAL['veget_e'], lw=1.0, radius=0.005, zorder=4)
    ax.text(0.50, 0.345, 'NSGA-II MULTI-OBJECTIVE OPTIMIZATION',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PAL['green_lt'], zorder=5)

    pf_ax = ax.inset_axes([0.075, 0.025, 0.870, 0.295], transform=ax.transData)
    rng2 = np.random.default_rng(7)
    u_pf  = np.sort(np.linspace(28.5, 35.2, 28) + rng2.normal(0, 0.25, 28))
    g_pf  = 0.60 - (u_pf - 28.5) / 30 + rng2.normal(0, 0.022, 28)
    u_dom = rng2.uniform(28, 37, 90)
    g_dom = rng2.uniform(0.04, 0.62, 90)

    pf_ax.scatter(u_dom, g_dom, c='#30363D', s=18, alpha=0.45, zorder=2,
                   label='Dominated solutions')
    sc = pf_ax.scatter(u_pf, g_pf, c=u_pf, cmap=UTCI_CMAP,
                        norm=Normalize(28.5, 35.2), s=45, zorder=4,
                        edgecolors='white', linewidths=0.5,
                        label='Pareto front')
    pf_ax.plot(u_pf, g_pf, '-', color=PAL['green_lt'], lw=1.2,
               alpha=0.7, zorder=3)
    # star on best (lowest UTCI)
    best = np.argmin(u_pf)
    pf_ax.scatter(u_pf[best], g_pf[best], marker='*', s=220,
                   c='#FFD700', edgecolors='white', lw=0.7, zorder=5)
    pf_ax.annotate('Optimal\ndesign', xy=(u_pf[best], g_pf[best]),
                    xytext=(u_pf[best]+1.2, g_pf[best]-0.06),
                    fontsize=7, color='#FFD700',
                    arrowprops=dict(arrowstyle='->', color='#FFD700', lw=0.8))

    pf_ax.set_xlabel('f₁: Mean UTCI (°C)  →  minimize', fontsize=7.5,
                      color=PAL['text'])
    pf_ax.set_ylabel('f₂: Green Coverage Ratio  →  maximize', fontsize=7.5,
                      color=PAL['text'])
    pf_ax.set_facecolor(PAL['bg_card2'])
    for sp in pf_ax.spines.values():
        sp.set_edgecolor(PAL['border'])
    pf_ax.tick_params(colors=PAL['text_dim'], labelsize=6.5)
    pf_ax.legend(fontsize=7, facecolor=PAL['bg_card'], edgecolor=PAL['border'],
                  labelcolor=PAL['text'], loc='upper right', markerscale=0.9,
                  handlelength=1.5)

    cb2 = plt.colorbar(sc, ax=pf_ax, orientation='vertical', pad=0.02)
    cb2.set_label('UTCI', fontsize=6.5, color=PAL['text'])
    cb2.ax.tick_params(labelsize=6, colors=PAL['text_dim'])

    draw_section_header(ax, 'THERMAL COMFORT OUTPUT & URBAN OPTIMIZATION',
                        PAL['hdr_orange'])


# ---------------------------------------------------------------------------
# INTER-SECTION FLOW ARROWS (drawn on figure canvas)
# ---------------------------------------------------------------------------
def draw_global_flow(fig, axes):
    """Thin accent lines connecting top→middle→bottom sections."""
    for xf in [0.012, 0.988]:
        line = Line2D([xf, xf], [0.018, 0.955],
                       transform=fig.transFigure,
                       color=PAL['border'], lw=0.8, alpha=0.5)
        fig.add_artist(line)


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
def save_poster(fig):
    out_dir = Path(__file__).parent
    png_path = out_dir / 'PI_ST_GNN_Research_Poster.png'
    pdf_path = out_dir / 'PI_ST_GNN_Research_Poster.pdf'

    print('Saving PNG (300 DPI) ...')
    fig.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), format='png')
    print(f'  PNG: {png_path}')

    print('Saving PDF ...')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  PDF: {pdf_path}')

    plt.close(fig)
    print('Done.')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print('Building PI-ST-GNN Research Poster ...')
    fig, axes = setup_figure()

    draw_poster_title(fig)

    draw_data_inputs_panel(axes['tc1'])
    draw_urban_graph_panel(axes['tc2'])
    draw_methodology_a_panel(axes['tc3'])

    draw_formulas_band(axes['mid'])

    draw_feature_encoding_panel(axes['bc1'])
    draw_spatiotemporal_panel(axes['bc2'])
    draw_output_panel(axes['bc3'])

    draw_global_flow(fig, axes)

    save_poster(fig)


if __name__ == '__main__':
    main()
