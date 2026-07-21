"""
generate_lit_urbantheory_figure.py
=====================================
Redraw of fig_lit_urbantheory.png (Thesis_GIA Ch1 fig:lit_urbantheory).

Panel (a): the actual Lynch (1960) "Image of the City" Los Angeles diagram
           (Path/Edge/Node/District/Landmark), loaded from the real scanned
           plate the user placed at Thesis_GIA/img/ -- not redrawn/approximated.
Panel (b): REAL map centred on National Yang Ming Chiao Tung University
           (NYCU, 陽明交大, Guangfu campus, ~24.7867N 120.9968E) -- real
           CartoDB basemap + real IoT sensor positions (station_metadata.json)
           + real OSM building footprints (local PBF extract via
           sensing_integration/osm_pbf_extract.py, NOT synthetic geometry) +
           the heterogeneous graph representation (object nodes on real
           buildings, air-node KNN mesh) drawn directly at Hsinchu urban
           scale, replacing the earlier abstract "urban morphology hierarchy"
           schematic panel.

Produces:
  figures/fig_lit_urbantheory.png/pdf
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
import contextily as ctx
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
THESIS_IMG_DIR = _ROOT / "Thesis_GIA" / "img"
DATA_DIR = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "iot_data"
FIG_DIR = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_SCRIPT_DIR.parent / "sensing_integration"))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

LYNCH_IMG = THESIS_IMG_DIR / "fig1_Kevin-Lynch-The-Image-of-the-City_Path,Edge,Node,District,Landmark.png"
PBF_PATH = _SCRIPT_DIR.parent / "data" / "osm" / "taiwan-latest.osm.pbf"
STATION_JSON = DATA_DIR / "station_metadata.json"

NYCU_LAT, NYCU_LON = 24.7867, 120.9968   # NYCU Guangfu campus, publicly known coordinate
VIEW_RADIUS_M = 700.0                     # panel (b) visible half-width
GRAPH_RADIUS_M = 260.0                    # sub-area where the graph mesh is drawn
GRID_SPACING_M = 34.0                     # air-node spacing within the graph sub-area
KNN_K = 6

mpl.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9.3,
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

print("[1] Loading real OSM buildings near NYCU (local PBF, cached region) ...")
from osm_pbf_extract import RegionOSM
region = RegionOSM(str(PBF_PATH), bbox=(23.95, 120.55, 25.10, 121.30)).load()
buildings = region.buildings_local(NYCU_LAT, NYCU_LON, radius_m=VIEW_RADIUS_M + 60, min_area_m2=15.0)
print(f"    {len(buildings)} real building footprints within {VIEW_RADIUS_M:.0f} m of NYCU")

print("[2] Loading real IoT station positions ...")
with open(STATION_JSON, encoding="utf-8") as f:
    stations = json.load(f)
tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
ctr_x, ctr_y = tr.transform(NYCU_LON, NYCU_LAT)

# buildings_local() / the graph mesh below are built in true ground metres
# (local tangent-plane, via a simple equirectangular deg->m conversion), but
# the basemap is in Web Mercator (EPSG:3857), whose local metre scale is
# stretched by sec(lat) relative to true ground metres away from the equator.
# At NYCU's latitude (~24.8N) that is a ~10% error -- ~70m at this figure's
# 700m radius -- which is exactly why the building footprints did not line
# up with the basemap. Scale local-metre offsets by this factor before
# adding them to the Web-Mercator centre so both agree.
MERC_SCALE = 1.0 / np.cos(np.radians(NYCU_LAT))
print(f"    Web Mercator local-scale correction at lat={NYCU_LAT}: x{MERC_SCALE:.4f}")

sta_m = []
for s in stations:
    sx, sy = tr.transform(s["lon"], s["lat"])
    if abs(sx - ctr_x) < VIEW_RADIUS_M * MERC_SCALE and abs(sy - ctr_y) < VIEW_RADIUS_M * MERC_SCALE:
        sta_m.append((sx, sy))
sta_m = np.array(sta_m) if sta_m else np.empty((0, 2))
print(f"    {len(sta_m)} real IoT stations within view")

# ── Figure: two panels, harmonised style (monochrome + single accent) ─────
print("[3] Rendering ...")
FRAME_LW = 1.3
FRAME_COLOR = "#000000"
TITLE_FS = 11.0

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.9))

# --- Panel (a): the real Lynch (1960) plate, auto-cropped to content ------
# The scanned plate has a wide white margin that panel (b)'s full-bleed map
# does not; crop to the bounding box of non-near-white pixels so both panels
# fill their frame to a comparable degree.
lynch_img = plt.imread(LYNCH_IMG)
gray = lynch_img[..., :3].mean(axis=-1) if lynch_img.ndim == 3 else lynch_img
mask = gray < 0.92
rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
pad = 8
r0, r1 = max(0, rows[0] - pad), min(gray.shape[0], rows[-1] + pad)
c0, c1 = max(0, cols[0] - pad), min(gray.shape[1], cols[-1] + pad)
lynch_crop = lynch_img[r0:r1, c0:c1]

axA.imshow(lynch_crop)
axA.set_xticks([]); axA.set_yticks([])
for spine in axA.spines.values():
    spine.set_visible(True); spine.set_linewidth(FRAME_LW); spine.set_color(FRAME_COLOR)
axA.set_title("(a) Lynch (1960) 都市意象五元素\n（Path／Edge／Node／District／Landmark）",
              fontsize=TITLE_FS, fontweight="bold", pad=10)

# --- Panel (b): real NYCU map + IoT + buildings + graph ------------------
# Muted, near-monochrome palette (slate-gray buildings, warm-gray graph mesh,
# black object-node squares, dark maroon sensor markers, hollow star landmark)
# to match panel (a)'s black/white/gray halftone academic-plate aesthetic
# instead of a saturated GIS-software colour scheme.
C_BUILDING = "#7a8087"
C_MESH     = "#9c8a6b"
C_OBJNODE  = "#141414"
C_SENSOR   = "#7a3030"
C_BOUNDARY = "#222222"

axB.set_xlim(ctr_x - VIEW_RADIUS_M * MERC_SCALE, ctr_x + VIEW_RADIUS_M * MERC_SCALE)
axB.set_ylim(ctr_y - VIEW_RADIUS_M * MERC_SCALE, ctr_y + VIEW_RADIUS_M * MERC_SCALE)
axB.set_aspect("equal")

for b in buildings:
    fp = np.array(b["footprint"])  # local ground metres relative to (NYCU_LAT, NYCU_LON), x=east, y=north
    poly_xy = np.column_stack([ctr_x + fp[:, 0] * MERC_SCALE, ctr_y + fp[:, 1] * MERC_SCALE])
    axB.add_patch(MplPolygon(poly_xy, closed=True, facecolor=C_BUILDING,
                              edgecolor="#3a3d40", linewidth=0.3, alpha=0.65, zorder=3))

# Real IoT sensor positions
if len(sta_m):
    axB.scatter(sta_m[:, 0], sta_m[:, 1], s=30, c=C_SENSOR, marker="^",
                edgecolors="white", linewidths=0.6, zorder=6,
                label=f"真實 IoT 感測器（n={len(sta_m)}）")

# --- Graph representation drawn directly on real geometry (sub-area) ---
gx = np.arange(-GRAPH_RADIUS_M, GRAPH_RADIUS_M + 1, GRID_SPACING_M)
GX, GY = np.meshgrid(gx, gx)
air_pts = np.column_stack([GX.ravel(), GY.ravel()])
air_pts = air_pts[np.linalg.norm(air_pts, axis=1) <= GRAPH_RADIUS_M]
air_xy = np.column_stack([ctr_x + air_pts[:, 0] * MERC_SCALE, ctr_y + air_pts[:, 1] * MERC_SCALE])

from scipy.spatial import cKDTree
tree = cKDTree(air_pts)
k_eff = min(KNN_K + 1, len(air_pts))
_, nbrs = tree.query(air_pts, k=k_eff)
segs = []
for i, row in enumerate(nbrs):
    for j in row[1:]:
        segs.append([air_xy[i], air_xy[j]])
lc = LineCollection(segs, colors=C_MESH, linewidths=0.5, alpha=0.7, zorder=4)
axB.add_collection(lc)
axB.scatter(air_xy[:, 0], air_xy[:, 1], s=8, c=C_MESH, edgecolors="none", zorder=5,
            label=f"空氣節點 KNN 網格（$k$={KNN_K}, 間距 {GRID_SPACING_M:.0f}m）")

# Object nodes: real building centroids within the graph sub-area
obj_pts = []
for b in buildings:
    fp = np.array(b["footprint"])
    c = fp[:-1].mean(axis=0) if len(fp) > 1 else fp.mean(axis=0)
    if np.linalg.norm(c) <= GRAPH_RADIUS_M:
        obj_pts.append((ctr_x + c[0] * MERC_SCALE, ctr_y + c[1] * MERC_SCALE))
obj_pts = np.array(obj_pts) if obj_pts else np.empty((0, 2))
if len(obj_pts):
    axB.scatter(obj_pts[:, 0], obj_pts[:, 1], s=36, c=C_OBJNODE, marker="s",
                edgecolors="white", linewidths=0.6, zorder=7,
                label=f"物件節點（真實建物質心，n={len(obj_pts)}）")

# Graph sub-area boundary
theta = np.linspace(0, 2 * np.pi, 100)
axB.plot(ctr_x + GRAPH_RADIUS_M * MERC_SCALE * np.cos(theta),
          ctr_y + GRAPH_RADIUS_M * MERC_SCALE * np.sin(theta),
          color=C_BOUNDARY, linewidth=1.0, linestyle="--", zorder=8)

# Landmark icon echoes Lynch's hollow-star convention (panel a legend)
axB.scatter([ctr_x], [ctr_y], s=170, c="white", marker="*", edgecolors="black",
            linewidths=1.1, zorder=9, label="陽明交大（光復校區）")

try:
    ctx.add_basemap(axB, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                     zoom=17, attribution="")
except Exception as e:
    print(f"    [WARN] basemap fetch failed: {e}")

axB.set_xticks([]); axB.set_yticks([])
for spine in axB.spines.values():
    spine.set_visible(True); spine.set_linewidth(FRAME_LW); spine.set_color(FRAME_COLOR)
axB.set_title("(b) 異質圖表徵疊圖於真實都市紋理\n（新竹．陽明交大周邊，真實建物足跡＋IoT 感測器）",
              fontsize=TITLE_FS, fontweight="bold", pad=10)
axB.legend(loc="lower left", fontsize=7.4, framealpha=0.92, handlelength=1.5,
           edgecolor="#555555", facecolor="white")

fig.suptitle("都市意象理論與異質圖表徵之對照：從經典理論到真實都市紋理建模",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_lit_urbantheory.pdf"
out_png = FIG_DIR / "fig_lit_urbantheory.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"[4] Saved: {out_pdf}\n           {out_png}")
