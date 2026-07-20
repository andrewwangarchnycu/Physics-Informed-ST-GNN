"""
generate_site_context_figure.py
=================================
Redraw of fig_geo1_site_context.png using REAL basemaps (contextily /
CartoDB), matching the visual style of fig_studyarea_sensor_density.py --
replaces the earlier version's hand-drawn Taiwan outline polygon with an
actual tile basemap at national scale, and the earlier schematic sensor
markers with real station_metadata.json positions at regional scale.

Panel (a): Taiwan national-scale real basemap, Hsinchu marked relative to
           four other well-known major cities (Taipei/Taichung/Tainan/
           Kaohsiung -- standard, publicly known city-centre coordinates).
Panel (b): Hsinchu regional-scale real basemap with real IoT sensor
           positions (station_metadata.json) within ~5 km of the site
           anchor, and the site anchor itself marked.
Panel (c): 80x80 m study site -- kept as a labelled SCHEMATIC of the
           procedural scenario structure (this is synthetic training
           geometry, not real-world geography, so it cannot honestly be
           drawn on a real basemap; the panel title says so explicitly).

Note: no CWA weather-station marker is plotted in this version -- the
exact station coordinate could not be verified from an authoritative
source in this session, and this script does not plot unverified
coordinates. Re-add it if/when a verified coordinate is available (e.g.
from CWA's CODiS station metadata export).

Produces:
  figures/fig_geo1_site_context.pdf
  figures/fig_geo1_site_context.png
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import contextily as ctx
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "iot_data"
FIG_DIR  = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROSTER_PKL = DATA_DIR / "all_devices_region.pkl"
SITES_JSON = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"
SCEN_PKL   = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"

CTR_LAT, CTR_LON = 24.78805, 120.99754   # NYCU Guangfu Campus anchor
# Expanded V4 study region
REGION = dict(lat_min=23.95, lat_max=25.10, lon_min=120.55, lon_max=121.30)
CITIES = {
    "Taipei":   (25.0330, 121.5654),
    "Taichung": (24.1477, 120.6736),
    "Tainan":   (22.9997, 120.2270),
    "Kaohsiung": (22.6273, 120.3014),
}

mpl.rcParams.update({
    "font.family":      "Microsoft JhengHei",
    "axes.unicode_minus": False,
    "font.size":        9,
    "axes.titlesize":   10.2,
    "axes.titleweight": "bold",
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

print("[1] Loading regional IoT roster + selected V4 sites ...")
import pickle
with open(ROSTER_PKL, "rb") as f:
    roster = pickle.load(f)
rows = [(la, lo) for (la, lo) in roster.values()
        if REGION["lat_min"] <= la <= REGION["lat_max"]
        and REGION["lon_min"] <= lo <= REGION["lon_max"]]
meta_df = pd.DataFrame(rows, columns=["lat", "lon"])
meta_df["mx"], meta_df["my"] = tr.transform(meta_df["lon"].values, meta_df["lat"].values)
sites = json.loads(Path(SITES_JSON).read_text(encoding="utf-8"))["sites"]
site_df = pd.DataFrame([(s["lat"], s["lon"]) for s in sites], columns=["lat", "lon"])
site_df["mx"], site_df["my"] = tr.transform(site_df["lon"].values, site_df["lat"].values)
ctr_x, ctr_y = tr.transform(CTR_LON, CTR_LAT)
print(f"    {len(meta_df)} regional sensors, {len(site_df)} selected V4 sites")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
axA, axB, axC = axes

# ── Panel (a): Taiwan national scale, real basemap ───────────────────────────
print("[2] Rendering panel (a): Taiwan national scale ...")
# Bounding box covering Taiwan main island with margin
lon_min, lon_max = 119.8, 122.1
lat_min, lat_max = 21.8, 25.4
x0, y0 = tr.transform(lon_min, lat_min)
x1, y1 = tr.transform(lon_max, lat_max)
axA.set_xlim(x0, x1)
axA.set_ylim(y0, y1)
axA.set_aspect("equal")

for name, (lat, lon) in CITIES.items():
    cx, cy = tr.transform(lon, lat)
    axA.scatter(cx, cy, s=22, c="#444444", zorder=4, edgecolors="white", linewidths=0.5)
    axA.annotate(name, (cx, cy), fontsize=7.5, color="#333333",
                 xytext=(4, 2), textcoords="offset points", zorder=5)

# V4 expanded region box
rx0, ry0 = tr.transform(REGION["lon_min"], REGION["lat_min"])
rx1, ry1 = tr.transform(REGION["lon_max"], REGION["lat_max"])
axA.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                        edgecolor="#c9463d", linewidth=1.6, zorder=5,
                        label="V4 study region"))
axA.scatter(ctr_x, ctr_y, s=90, c="#c9463d", marker="*", zorder=6,
            edgecolors="white", linewidths=0.8)
axA.annotate("NYCU", (ctr_x, ctr_y), fontsize=8.5, fontweight="bold",
             color="#c9463d", xytext=(6, 6), textcoords="offset points", zorder=6)

try:
    ctx.add_basemap(axA, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                     zoom=8, attribution="")
except Exception as e:
    print(f"    [WARN] basemap fetch failed for panel (a): {e}")

axA.set_xticks([]); axA.set_yticks([])
axA.set_title("(a) Taiwan — Case Study Location")
axA.legend(loc="lower left", framealpha=0.85, fontsize=7.5)

# ── Panel (b): expanded V4 region, real sensors + selected sites ─────────────
print("[3] Rendering panel (b): expanded V4 region ...")
axB.set_xlim(rx0, rx1)
axB.set_ylim(ry0, ry1)
axB.set_aspect("equal")

axB.scatter(meta_df["mx"], meta_df["my"], s=4, c="#2f6690", alpha=0.45,
            edgecolors="none", zorder=3, label=f"IoT sensor (n={len(meta_df)})")
axB.scatter(site_df["mx"], site_df["my"], s=10, c="#3fa34d", alpha=0.9,
            edgecolors="black", linewidths=0.2, zorder=4,
            label=f"selected V4 site (n={len(site_df)})")
axB.scatter(ctr_x, ctr_y, s=110, c="#c9463d", marker="*", zorder=6,
            edgecolors="white", linewidths=0.7, label="NYCU anchor")

try:
    ctx.add_basemap(axB, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                     zoom=10, attribution="")
except Exception as e:
    print(f"    [WARN] basemap fetch failed for panel (b): {e}")

axB.set_xticks([]); axB.set_yticks([])
axB.set_title("(b) Expanded V4 region — real IoT sensors & selected sites")
axB.legend(loc="lower left", framealpha=0.85, fontsize=7.0)

# ── Panel (c): a REAL 80x80 m V4 scene (OSM geometry), not a schematic ───────
print("[4] Rendering panel (c): real V4 scene geometry ...")
import pickle as _pk
_scen = _pk.load(open(SCEN_PKL, "rb"))
# pick a readable urban scene: moderate building count
_sc = sorted(_scen, key=lambda s: abs(len(s["buildings"]) - 8))[0]
axC.set_xlim(-46, 46); axC.set_ylim(-46, 46)
axC.set_aspect("equal")
axC.add_patch(Rectangle((-40, -40), 80, 80, fill=False, edgecolor="black",
                        linewidth=1.3, label="80×80 m site"))
for b in _sc["buildings"]:
    fp = b["footprint"]
    xy = list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)
    from matplotlib.patches import Polygon as _MP
    axC.add_patch(_MP(xy, closed=True, facecolor="#5b7fa6", edgecolor="black",
                      linewidth=0.5, alpha=0.85, zorder=3))
for t in _sc["trees"][:40]:
    x, y = t["pos"]
    axC.add_patch(Circle((x, y), max(1.2, t["height"] * 0.15), facecolor="#7a9e6a",
                         edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3))
gx = np.arange(-39, 40, 4)
GX, GY = np.meshgrid(gx, gx)
axC.scatter(GX, GY, s=1.2, c="#d98c3d", alpha=0.45, zorder=2, label="Air nodes (grid)")

axC.set_xlabel("x (m)"); axC.set_ylabel("y (m)")
axC.set_title("(c) 80×80 m study site — real V4 scene\n(OSM buildings + ETH canopy)", fontsize=9.5)
axC.legend(loc="upper right", framealpha=0.85, fontsize=6.8)

fig.suptitle("Site Context: Taiwan National Scale → V4 Study Region → Real 80×80 m Urban Block",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()

out_pdf = FIG_DIR / "fig_geo1_site_context.pdf"
out_png = FIG_DIR / "fig_geo1_site_context.png"
fig.savefig(out_pdf)
fig.savefig(out_png)
print(f"[5] Saved: {out_pdf}\n           {out_png}")
