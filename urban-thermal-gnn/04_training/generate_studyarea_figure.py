"""
generate_studyarea_figure.py
=============================
v4 study-area selection figure: identifies high-density IoT sensor clusters
within the existing simulation domain radius and visualizes them against a
real Hsinchu basemap, alongside the sensor-density surface used to pick the
GIS sampling regions for v4 training.

Design decision (per user confirmation 2026-07-17): v4 reuses the EXISTING
simulation anchor point (site_constraints.yaml: lat=24.80, lon=120.97) rather
than expanding to a new multi-county domain. This script's job is to justify
*where within that existing radius* the high-density sensor clusters sit,
so GIS building-footprint sampling (nlsc_building_vectorize.py) can be
targeted at those clusters rather than the full 15 km catchment.

Produces:
  figures/fig_studyarea_sensor_density.pdf
  figures/fig_studyarea_sensor_density.png

Panel (a): KDE sensor-density heatmap over the full 15 km IoT catchment,
           with the top-K density clusters outlined and ranked.
Panel (b): Close-up of the #1 density cluster with individual sensor points,
           for use as the GIS sampling reference in the thesis GIS section.

NOTE ON BUILDING-FOOTPRINT LAYER: this script does NOT yet overlay real
NLSC building polygons, because those require an interactive QGIS session
(nlsc_building_vectorize.py v4, cannot run headless -- needs `iface`).
Once that script is run for the panel-(b) extent and exported (e.g. to
GeoJSON via QGIS "Export > Save Features As"), point BUILDING_GEOJSON below
at the export and re-run this script to add the building-footprint overlay.
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde
from pathlib import Path
import contextily as ctx
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "iot_data"
FIG_DIR  = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# V4 update: use the full regional IoT roster (2,018 real stations across the
# expanded study region) and overlay the 300 selected real training sites,
# instead of the 806-station 15 km Hsinchu-only catchment. Same KDE-density +
# basemap visual style is preserved.
ROSTER_PKL = DATA_DIR / "all_devices_region.pkl"
SITES_JSON = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"

BUILDING_GEOJSON = None

CTR_LAT, CTR_LON = 24.78805, 120.99754   # NYCU Guangfu Campus anchor
# Expanded V4 study region (Hsinchu City/County, Taoyuan, Miaoli, Taichung)
REGION = dict(lat_min=23.95, lat_max=25.10, lon_min=120.55, lon_max=121.30)
TOP_K_CLUSTERS = 5
CLUSTER_RADIUS_M = 400.0              # cluster footprint radius for panel (b) close-up

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

print("[1] Loading regional IoT roster + selected V4 sites ...")
import pickle
with open(ROSTER_PKL, "rb") as f:
    roster = pickle.load(f)          # {deviceId: (lat, lon)}
rows = [(la, lo) for (la, lo) in roster.values()
        if REGION["lat_min"] <= la <= REGION["lat_max"]
        and REGION["lon_min"] <= lo <= REGION["lon_max"]]
meta_df = pd.DataFrame(rows, columns=["lat", "lon"])
print(f"    {len(meta_df)} real stations in expanded V4 region")

sites = json.loads(Path(SITES_JSON).read_text(encoding="utf-8"))["sites"]
site_df = pd.DataFrame([(s["lat"], s["lon"]) for s in sites], columns=["lat", "lon"])
print(f"    {len(site_df)} selected V4 training sites")

tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
meta_df["mx"], meta_df["my"] = tr.transform(meta_df["lon"].values, meta_df["lat"].values)
site_df["mx"], site_df["my"] = tr.transform(site_df["lon"].values, site_df["lat"].values)
ctr_x, ctr_y = tr.transform(CTR_LON, CTR_LAT)
x0, y0 = tr.transform(REGION["lon_min"], REGION["lat_min"])
x1, y1 = tr.transform(REGION["lon_max"], REGION["lat_max"])

# ── 2. KDE density surface over the region ───────────────────────────────────
print("[2] Computing sensor-density KDE surface ...")
xy = np.vstack([meta_df["mx"].values, meta_df["my"].values])
kde = gaussian_kde(xy, bw_method=0.12)

grid_n = 260
gx = np.linspace(x0, x1, grid_n)
gy = np.linspace(y0, y1, grid_n)
GX, GY = np.meshgrid(gx, gy)
density = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
density_masked = density

# ── 3. Identify top-K density clusters via local maxima ──────────────────────
print("[3] Ranking density clusters ...")
from scipy import ndimage as ndi
# Local maxima of the density surface (footprint = ~10 grid cells)
local_max = (density_masked == ndi.maximum_filter(
    np.nan_to_num(density_masked, nan=-1), size=12))
local_max &= ~np.isnan(density_masked)
peak_rows, peak_cols = np.where(local_max)
peak_vals = density_masked[peak_rows, peak_cols]
order = np.argsort(-peak_vals)[:TOP_K_CLUSTERS]

clusters = []
for rank, idx in enumerate(order, 1):
    r, c = peak_rows[idx], peak_cols[idx]
    cx, cy = GX[r, c], GY[r, c]
    n_within = int(((meta_df["mx"] - cx) ** 2 + (meta_df["my"] - cy) ** 2
                     <= CLUSTER_RADIUS_M ** 2).sum())
    clusters.append({"rank": rank, "mx": cx, "my": cy,
                      "density": float(peak_vals[idx]), "n_sensors": n_within})
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326",
                                     always_xy=True).transform(cx, cy)
    print(f"    #{rank}: ({lat:.5f}, {lon:.5f})  "
          f"n_sensors(r={CLUSTER_RADIUS_M:.0f}m)={n_within}  "
          f"density_score={peak_vals[idx]:.3e}")

top_cluster = clusters[0]

# ── 4. Figure: panel (a) full catchment density, panel (b) top-cluster close-up
print("[4] Rendering figure ...")
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.4))

# --- Panel (a): full expanded region ---
axA.set_xlim(x0, x1)
axA.set_ylim(y0, y1)
axA.set_aspect("equal")
im = axA.contourf(GX, GY, density_masked, levels=14, cmap="magma", alpha=0.5, zorder=2)
axA.scatter(meta_df["mx"], meta_df["my"], s=2, c="cyan", alpha=0.4,
            linewidths=0, zorder=3, label=f"IoT sensor (n={len(meta_df)})")
# overlay the 300 selected V4 training sites
axA.scatter(site_df["mx"], site_df["my"], s=7, c="lime", alpha=0.9,
            edgecolors="black", linewidths=0.2, zorder=4,
            label=f"selected V4 site (n={len(site_df)})")
axA.scatter(ctr_x, ctr_y, s=120, c="red", marker="*", edgecolors="white",
            linewidths=0.8, zorder=6, label="NYCU anchor")

try:
    ctx.add_basemap(axA, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                     zoom=10, attribution="")
except Exception as e:
    print(f"    [WARN] basemap fetch failed for panel (a): {e}")

axA.set_title("(a) Sensor-density surface — expanded V4 study region")
axA.set_xticks([]); axA.set_yticks([])
axA.legend(loc="lower left", framealpha=0.85, fontsize=7.5)

# --- Panel (b): top-cluster close-up ---
# Panel (b): NYCU anchor core close-up with selected V4 sites + sensors
bx, by = ctr_x, ctr_y
half = 6000.0
axB.set_xlim(bx - half, bx + half)
axB.set_ylim(by - half, by + half)
axB.set_aspect("equal")

near = meta_df[(meta_df["mx"] - bx) ** 2 + (meta_df["my"] - by) ** 2 <= half ** 2]
near_sites = site_df[(site_df["mx"] - bx) ** 2 + (site_df["my"] - by) ** 2 <= half ** 2]
axB.scatter(near["mx"], near["my"], s=12, c="cyan", edgecolors="none",
            alpha=0.6, zorder=3, label=f"IoT sensor (n={len(near)})")
axB.scatter(near_sites["mx"], near_sites["my"], s=32, c="lime", edgecolors="black",
            linewidths=0.4, zorder=4, label=f"selected V4 site (n={len(near_sites)})")
axB.scatter(bx, by, s=160, c="red", marker="*", edgecolors="white",
            linewidths=0.9, zorder=6, label="NYCU anchor")
axB.set_title("(b) NYCU anchor core — selected V4 sites x IoT sensors")

try:
    ctx.add_basemap(axB, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                     zoom=13, attribution="")
except Exception as e:
    print(f"    [WARN] basemap fetch failed for panel (b): {e}")

axB.set_xticks([]); axB.set_yticks([])
axB.legend(loc="lower left", framealpha=0.85, fontsize=7.5)

fig.suptitle("V4 Study Area: Expanded Regional IoT Sensor Density and 300 Selected Real Sites",
             fontsize=11.5, fontweight="bold", y=1.01)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_studyarea_sensor_density.pdf"
out_png = FIG_DIR / "fig_studyarea_sensor_density.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"[5] Saved: {out_pdf}")
print(f"           {out_png}")

# ── 6. Write cluster ranking table for thesis GIS section ────────────────────
out_json = FIG_DIR.parent / "studyarea_clusters.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "site_anchor": {"lat": CTR_LAT, "lon": CTR_LON},
        "region_bbox": REGION,
        "n_stations": int(len(meta_df)),
        "n_selected_sites": int(len(site_df)),
        "cluster_radius_m": CLUSTER_RADIUS_M,
        "clusters": clusters,
    }, f, ensure_ascii=False, indent=2, default=float)
print(f"[6] Cluster table: {out_json}")
