"""
generate_graph_construction_figure.py
=======================================
Academic-quality figure faithfully depicting the heterogeneous graph
construction implemented in 02_graph_construction/dataset.py
(UTCIGraphDataset.get()). Loads ONE real scenario from ground_truth_v2.h5
and scenarios.pkl -- every element drawn is a real value the dataset loader
actually produces, not a schematic mock-up.

Panel (a): spatial layout -- object nodes (buildings, from scenarios.pkl
           "buildings" footprints) + air nodes (sensor_pts from HDF5),
           coloured by a real per-node feature (building_height for object
           nodes underlying the air feature bh_nt; UTCI for air nodes).
Panel (b): edges actually constructed by dataset.py --
           (object,semantic,object) fully-connected + (air,contiguity,air)
           KNN(K=8) via cKDTree -- drawn as real edge_index pairs, not a
           schematic.
Panel (c): node/edge type schema + air-node feature vector layout, exactly
           matching the dim_air=9 (V2) feature order documented in
           dataset.py's module docstring.

Produces:
  figures/fig_graph_construction.pdf
  figures/fig_graph_construction.png
"""

import sys, pickle, warnings
from pathlib import Path
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
H5_PATH   = _ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "ground_truth_v2.h5"
SCEN_PATH = _ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "scenarios.pkl"
FIG_DIR   = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KNN_K = 8   # matches UTCIGraphDataset default knn_k

mpl.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

print("[1] Loading one real scenario ...")
with open(SCEN_PATH, "rb") as f:
    all_scenarios = pickle.load(f)
scenario_map = {int(s.get("scenario_id", i)): s for i, s in enumerate(all_scenarios)}

with h5py.File(H5_PATH, "r") as hf:
    scen_ids = sorted(int(k) for k in hf["scenarios"].keys())
    # Pick the first scenario id that also has real building geometry
    sid = None
    for cand in scen_ids:
        s = scenario_map.get(cand)
        if s is not None and s.get("buildings"):
            sid = cand
            break
    if sid is None:
        sid = scen_ids[0]
    print(f"    Using scenario_id={sid}  ({len(scen_ids)} scenarios in HDF5)")

    grp = hf[f"scenarios/{sid}"]
    sensor_pts  = grp["sensor_pts"][()]              # (N,2)
    utci        = grp["utci"][()]                    # (T,N)
    bldg_height = grp["building_height"][()]          # (N,)
    tree_height = grp["tree_height"][()]               # (N,)
    svf         = grp["svf"][()]                       # (N,)

scenario = scenario_map.get(sid, {})
buildings = scenario.get("buildings", [])
N_air = sensor_pts.shape[0]
N_obj = len(buildings)
print(f"    N_air={N_air}  N_obj={N_obj}  T={utci.shape[0]}")

# ── 2. Build the SAME edges dataset.py actually constructs ──────────────────
print("[2] Constructing real edge sets (mirrors UTCIGraphDataset.get) ...")
k = min(KNN_K, max(N_air - 1, 1))
tree = cKDTree(sensor_pts)
_, knn_idx = tree.query(sensor_pts, k=k + 1)
contiguity_edges = []
for i, nbrs in enumerate(knn_idx):
    for j in nbrs[1:]:
        contiguity_edges.append((i, int(j)))
print(f"    (air,contiguity,air): {len(contiguity_edges)} directed edges (KNN K={k})")

semantic_edges = [(i, j) for i in range(N_obj) for j in range(N_obj) if i != j]
print(f"    (object,semantic,object): {len(semantic_edges)} directed edges (fully-connected)")

# ── 3. Figure ─────────────────────────────────────────────────────────────
print("[3] Rendering figure ...")
fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.3, top=0.78, bottom=0.1)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

utci_mean = utci.mean(axis=0)  # (N,) time-averaged UTCI per air node, real data

# --- Panel A: spatial layout ---
for b in buildings:
    fp = b.get("footprint")
    if fp is not None and hasattr(fp, "exterior"):
        xs, ys = fp.exterior.xy
        axA.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                  facecolor="#c9463d", edgecolor="black",
                                  alpha=0.55, linewidth=0.8, zorder=3))
sc = axA.scatter(sensor_pts[:, 0], sensor_pts[:, 1], c=utci_mean, cmap="inferno",
                  s=14, zorder=4, edgecolors="none")
cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.04)
cb.set_label("mean UTCI (°C), real simulation output", fontsize=8)
axA.set_title(f"(a) object + air nodes\n(scenario {sid}: N_obj={N_obj}, N_air={N_air})", fontsize=9.5)
axA.set_xlabel("x (m)"); axA.set_ylabel("y (m)")
axA.set_aspect("equal")
legend_a = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#c9463d",
           markeredgecolor="black", alpha=0.55, markersize=9, label="object node (building)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
           markersize=6, label="air node (sensor point)"),
]
axA.legend(handles=legend_a, loc="upper right", fontsize=7, framealpha=0.9)

# --- Panel B: real edges ---
# N_air=6241 on an ~80x80m domain is a dense ~1m grid -- drawing all 49928
# KNN edges at full-domain scale saturates into an unreadable solid mass.
# Zoom into one representative local neighbourhood (15x15m window centred
# on a building, if any) so individual KNN(K=8) edges are actually visible;
# this is the SAME edge_index data, just windowed for legibility.
if N_obj >= 1:
    centroids = np.array([b.get("centroid", (0.0, 0.0)) for b in buildings])
    win_cx, win_cy = centroids[0]
else:
    win_cx, win_cy = sensor_pts[:, 0].mean(), sensor_pts[:, 1].mean()
win_half = 9.0

for b in buildings:
    fp = b.get("footprint")
    if fp is not None and hasattr(fp, "exterior"):
        xs, ys = fp.exterior.xy
        axB.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                                  facecolor="#c9463d", edgecolor="black",
                                  alpha=0.35, linewidth=0.6, zorder=2))

in_window = (np.abs(sensor_pts[:, 0] - win_cx) <= win_half) & \
            (np.abs(sensor_pts[:, 1] - win_cy) <= win_half)
window_ids = set(np.where(in_window)[0].tolist())
edges_in_window = [(i, j) for (i, j) in contiguity_edges
                    if i in window_ids and j in window_ids]
for (i, j) in edges_in_window:
    axB.plot([sensor_pts[i, 0], sensor_pts[j, 0]],
              [sensor_pts[i, 1], sensor_pts[j, 1]],
              color="steelblue", linewidth=0.6, alpha=0.55, zorder=1)

if N_obj >= 2:
    for (i, j) in semantic_edges:
        axB.plot([centroids[i, 0], centroids[j, 0]],
                  [centroids[i, 1], centroids[j, 1]],
                  color="#c9463d", linewidth=1.1, alpha=0.7, zorder=3)

axB.scatter(sensor_pts[in_window, 0], sensor_pts[in_window, 1],
            c="orange", s=26, zorder=4, edgecolors="black", linewidths=0.4)
axB.set_xlim(win_cx - win_half, win_cx + win_half)
axB.set_ylim(win_cy - win_half, win_cy + win_half)
axB.set_title(f"(b) KNN(K={k}) contiguity edges, local {win_half*2:.0f}m×{win_half*2:.0f}m window\n"
               f"({len(edges_in_window)} of {len(contiguity_edges)} total edges shown)",
               fontsize=9.5)
axB.set_xlabel("x (m)"); axB.set_ylabel("y (m)")
axB.set_aspect("equal")
legend_b = [
    Line2D([0], [0], color="steelblue", linewidth=1.2, alpha=0.7, label="(air,contiguity,air) — KNN"),
    Line2D([0], [0], color="#c9463d", linewidth=1.2, alpha=0.8, label="(object,semantic,object) — full"),
]
axB.legend(handles=legend_b, loc="upper right", fontsize=6.5, framealpha=0.9)

# --- Panel C: schema (node/edge types + feature layout, from dataset.py docstring) ---
# Keep the axes frame (spines) visible instead of axis("off") so panel C's
# box has the same top/bottom extent as panels A/B, rather than a tightly
# cropped text bbox floating at an unrelated height.
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)
axC.set_xticks([])
axC.set_yticks([])
for spine in axC.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
axC.set_title("(c) heterogeneous graph schema\n(UTCIGraphDataset.get, dim_air=9)", fontsize=9.5)

schema_text = (
    "Node types\n"
    "───────────────────────\n"
    "object  (N_obj, 7)\n"
    "  0 height/50   1 floors/12\n"
    "  2 footprint_area/2000\n"
    "  3–4 centroid_x,y /80\n"
    "  5 gfa/20000   6 is_L_shape\n\n"
    "air  (N_air, T, 9)\n"
    "  0 ta_norm     1 mrt_norm\n"
    "  2 va_norm     3 rh_norm\n"
    "  4 svf         5 in_shadow\n"
    "  6 bldg_h_norm 7 tree_h_norm\n"
    "  8 ts_norm  (V2 surface temp.)\n\n"
    "Edge types\n"
    "───────────────────────\n"
    "(object,semantic,object)\n"
    "  fully-connected, static\n\n"
    "(air,contiguity,air)\n"
    f"  KNN K={KNN_K}, static, cKDTree\n\n"
    "dynamic_edges[t]  (populated\n"
    "  downstream — shadow / convective /\n"
    "  contiguity edges per timestep,\n"
    "  see urbangraph.py _merge_edges)"
)
axC.text(0.02, 0.98, schema_text, transform=axC.transAxes,
          fontsize=7.8, family="monospace", va="top", ha="left",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f4ef", edgecolor="gray"))

fig.suptitle("Heterogeneous Graph Construction — 02_graph_construction/dataset.py",
             fontsize=12, fontweight="bold", y=0.98)

# --- Force all three panel frames to share identical top/bottom edges ---
# fig.colorbar(ax=axA) and the legends can each nudge their host axes'
# bounding box independently; explicitly re-syncing y0/y1 here guarantees
# the (a)/(b)/(c) frames -- and therefore their titles -- line up exactly,
# regardless of any such incidental layout shifts.
posA, posB, posC = axA.get_position(), axB.get_position(), axC.get_position()
y0 = max(posA.y0, posB.y0, posC.y0)   # highest bottom edge (common lower bound)
y1 = min(posA.y1, posB.y1, posC.y1)   # lowest top edge (common upper bound)
for ax, pos in ((axA, posA), (axB, posB), (axC, posC)):
    ax.set_position([pos.x0, y0, pos.width, y1 - y0])

out_pdf = FIG_DIR / "fig_graph_construction.pdf"
out_png = FIG_DIR / "fig_graph_construction.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"[4] Saved: {out_pdf}")
print(f"           {out_png}")
