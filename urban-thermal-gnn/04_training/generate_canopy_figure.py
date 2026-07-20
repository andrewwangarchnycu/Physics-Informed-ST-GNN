"""
04_training/generate_canopy_figure.py
════════════════════════════════════════════════════════════════
Figure: ETH GlobalCanopyHeight (10m, 2020) real-world canopy height
raster draped over a real basemap, with the research site boundary
marked, for the thesis GIS-foundation section.

Data source: ETH_GlobalCanopyHeight_10m_2020_N24E120_Map.tif (full 3x3
  degree tile, downloaded from the official ETH libdrive mirror linked
  from https://langnico.github.io/globalcanopyheight/)
  (Lang, Jetz, Schindler & Wegner 2023, "A high-resolution canopy
   height model of the Earth", Nature Ecology & Evolution)

Panels:
  (a) Regional context (~15 km) around the Hsinchu research site,
      canopy height overlaid on a real basemap, same 15 km radius
      used in fig_iot_sensor_map for visual consistency.
  (b) Local zoom (~1.5 km) around the site, with the 80x80 m study
      boundary drawn to scale.

Usage:
    python generate_canopy_figure.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import contextily as cx
from pyproj import Transformer

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TIF_PATH = HERE.parent / "data" / "canopy" / "ETH_GlobalCanopyHeight_10m_2020_N24E120_Map.tif"
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

SITE_LAT, SITE_LON = 24.80, 120.97   # research site anchor (WGS84)
SITE_SIZE_M = 80.0                    # 80 x 80 m study boundary

REGIONAL_RADIUS_DEG = 0.135   # ~15 km, matches fig_iot_sensor_map
LOCAL_RADIUS_DEG = 0.0135     # ~1.5 km


def _reproject_to_3857(src_path, bounds_wgs84):
    """Crop the WGS84 GeoTIFF to bounds and reproject to Web Mercator."""
    with rasterio.open(src_path) as src:
        window = from_bounds(*bounds_wgs84, transform=src.transform)
        window = window.round_lengths().round_offsets()
        src_transform = src.window_transform(window)
        src_arr = src.read(1, window=window)
        src_crs = src.crs

    dst_crs = "EPSG:3857"
    dst_transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src_arr.shape[1], src_arr.shape[0],
        *rasterio.transform.array_bounds(src_arr.shape[0], src_arr.shape[1], src_transform),
    )
    dst_arr = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=255,
        dst_nodata=np.nan,
    )
    # Web-Mercator extent of the reprojected array, for imshow(extent=...)
    left, top = dst_transform * (0, 0)
    right, bottom = dst_transform * (width, height)
    return dst_arr, (left, right, bottom, top)


def _site_xy_3857():
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tf.transform(SITE_LON, SITE_LAT)
    return x, y


def make_panel(ax, radius_deg, title, show_site_box, show_15km_circle):
    bounds = (SITE_LON - radius_deg, SITE_LAT - radius_deg,
              SITE_LON + radius_deg, SITE_LAT + radius_deg)
    canopy, extent = _reproject_to_3857(TIF_PATH, bounds)

    site_x, site_y = _site_xy_3857()

    # Basemap first (real-world imagery)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.Esri.WorldImagery,
                    attribution=False, zoom="auto")

    # Canopy height overlay: mask true no-data / bare-ground (0 m) pixels as
    # transparent so roads, buildings and open ground show the basemap
    # underneath; forested / vegetated pixels are shaded with the same
    # dark-purple -> orange "inferno" ramp used by the official ETH
    # GlobalCanopyHeight browser app, for direct visual comparability.
    masked = np.ma.masked_where(~(canopy > 0.5), canopy)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(masked, extent=extent, origin="upper", cmap=cmap,
                    vmin=0, vmax=40, alpha=0.85, interpolation="nearest", zorder=3)

    if show_15km_circle:
        circ = plt.Circle((site_x, site_y), 15_000, fill=False,
                           edgecolor="white", linestyle="--", linewidth=1.4, zorder=4)
        ax.add_patch(circ)

    if show_site_box:
        half = SITE_SIZE_M / 2.0
        # Exaggerate linewidth so the 80 m box is visible at print size.
        # Cyan reads clearly against the inferno ramp's black/purple/orange range.
        box = mpatches.Rectangle((site_x - half, site_y - half), SITE_SIZE_M, SITE_SIZE_M,
                                  fill=False, edgecolor="#00E5FF", linewidth=2.4, zorder=5)
        ax.add_patch(box)
    else:
        ax.plot(site_x, site_y, marker="*", color="#00E5FF", markersize=18,
                markeredgecolor="black", markeredgewidth=0.7, zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    make_panel(axes[0], REGIONAL_RADIUS_DEG,
               "(a) 研究場域周邊 15 km 樹冠高度分布", show_site_box=False, show_15km_circle=True)
    im = make_panel(axes[1], LOCAL_RADIUS_DEG,
                     "(b) 研究場域局部放大（青框 = 80×80 m 場域）", show_site_box=True, show_15km_circle=False)

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.5)
    cbar.set_label("樹冠高度 Canopy Height (m)", fontsize=9)

    fig.suptitle("ETH GlobalCanopyHeight (10 m, 2020) 樹冠高度資料與研究場域對照", fontsize=13, fontweight="bold", y=1.02)

    out_path = OUT_DIR / "fig_canopy_height_map.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
