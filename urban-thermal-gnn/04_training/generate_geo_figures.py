"""
generate_geo_figures.py
Academic-quality GIS visualization of Colife IoT temperature & humidity
sensor network over real Hsinchu basemap (CartoDB Positron).
Produces:
  figures/fig_iot_sensor_map.pdf
  figures/fig_iot_sensor_map.png
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from pathlib import Path
import contextily as ctx
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR.parent / "01_data_generation" / "outputs" / "iot_data"
FIG_DIR   = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# V4 update: regional roster (2,018 stations) + real 2025 summer snapshot,
# spanning the expanded study region instead of the 15 km Hsinchu catchment.
ROSTER_PKL = DATA_DIR / "all_devices_region.pkl"
_IN = _SCRIPT_DIR.parent / "01_data_generation" / "inputs" / "iot_data"
TEMP_CSV  = _IN / "moenviot_temperature" / "moenviot_temperature_20250715.csv"
HUMI_CSV  = _IN / "moenviot_humidity"    / "moenviot_humidity_20250715.csv"

CTR_LAT, CTR_LON = 24.78805, 120.99754
REGION = dict(lat_min=23.95, lat_max=25.10, lon_min=120.55, lon_max=121.30)

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

print("[1] Loading data ...")
import pickle
with open(ROSTER_PKL, "rb") as f:
    roster = pickle.load(f)
meta_df = pd.DataFrame([(str(did), la, lo) for did, (la, lo) in roster.items()
                        if REGION["lat_min"] <= la <= REGION["lat_max"]
                        and REGION["lon_min"] <= lo <= REGION["lon_max"]],
                       columns=["id", "lat", "lon"])
print(f"    {len(meta_df)} regional stations")

def read_csv(path, val_col):
    # noon snapshot for a clean spatial field
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["deviceId", val_col, "time"])
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df["deviceId"] = df["deviceId"].astype(str)
    t = pd.to_datetime(df["time"], errors="coerce")
    df = df[t.dt.hour == 12]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return (df.groupby("deviceId")[val_col].mean().reset_index()
              .rename(columns={"deviceId":"id", val_col:val_col}))

temp_mean = read_csv(TEMP_CSV, "temperature")
humi_mean = read_csv(HUMI_CSV, "humidity")

df = (meta_df
      .merge(temp_mean, on="id", how="inner")
      .merge(humi_mean, on="id", how="inner"))
df = df[(df["temperature"] > 10) & (df["temperature"] < 50)]
df = df[(df["humidity"] >= 5)    & (df["humidity"] <= 100)]
print(f"    {len(df)} stations  T=[{df['temperature'].min():.1f},{df['temperature'].max():.1f}]  RH=[{df['humidity'].min():.1f},{df['humidity'].max():.1f}]")

print("[2] Projecting to Web-Mercator ...")
tr     = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
tr_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
df["mx"], df["my"] = tr.transform(df["lon"].values, df["lat"].values)
ctr_x, ctr_y = tr.transform(CTR_LON, CTR_LAT)
x_min, y_min = tr.transform(REGION["lon_min"], REGION["lat_min"])
x_max, y_max = tr.transform(REGION["lon_max"], REGION["lat_max"])

T_lo  = float(max(28.0, df["temperature"].quantile(0.02)))
T_hi  = float(min(42.0, df["temperature"].quantile(0.98)))
RH_lo = float(max(30.0, df["humidity"].quantile(0.02)))
RH_hi = float(min(90.0, df["humidity"].quantile(0.98)))

norm_T  = Normalize(vmin=T_lo,  vmax=T_hi)
norm_RH = Normalize(vmin=RH_lo, vmax=RH_hi)
cmap_T  = plt.get_cmap("RdYlGn_r")
cmap_RH = plt.get_cmap("YlGnBu")

print("[3] Rendering figure ...")
fig = plt.figure(figsize=(15, 7.6))
ax_T  = fig.add_axes([0.030, 0.07, 0.425, 0.84])
ax_RH = fig.add_axes([0.505, 0.07, 0.425, 0.84])
cax_T  = fig.add_axes([0.458, 0.13, 0.012, 0.68])
cax_RH = fig.add_axes([0.934, 0.13, 0.012, 0.68])

def draw_panel(ax, col, cmap, norm, label, unit, cax, letter):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=12, attribution=False, reset_extent=False)
    except Exception as e:
        print(f"    [WARN] basemap: {e}")
        ax.set_facecolor("#e0e0e0")

    ax.scatter(df["mx"], df["my"], c=df[col], cmap=cmap, norm=norm,
               s=13, alpha=0.85, linewidths=0.2, edgecolors="white", zorder=4)

    ax.plot(ctr_x, ctr_y, marker="*", ms=13, mew=0.8, color="red",
            mec="white", zorder=6)
    ax.annotate("NYCU anchor",
                xy=(ctr_x, ctr_y),
                xytext=(ctr_x + 0.10*(x_max-x_min), ctr_y + 0.05*(y_max-y_min)),
                fontsize=7.0, color="#111111",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555"),
                zorder=6)

    na_x   = x_min + 0.055*(x_max - x_min)
    na_bot = y_max - 0.120*(y_max - y_min)
    na_top = y_max - 0.042*(y_max - y_min)
    ax.annotate("", xy=(na_x, na_top), xytext=(na_x, na_bot),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=7)
    ax.text(na_x, na_top + 0.013*(y_max-y_min), "N",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", zorder=7)

    sb_x0 = x_min + 0.58*(x_max - x_min)
    sb_x1 = sb_x0 + 20000
    sb_y  = y_min + 0.040*(y_max - y_min)
    tk    = 0.004*(y_max - y_min)
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y], lw=3.5, color="black",
            solid_capstyle="butt", zorder=7)
    for xp in [sb_x0, (sb_x0+sb_x1)/2, sb_x1]:
        ax.plot([xp, xp], [sb_y-tk, sb_y+tk], lw=1.3, color="black", zorder=7)
    ax.text(sb_x0,              sb_y+2.4*tk, "0",     ha="center", fontsize=6.8, zorder=7)
    ax.text((sb_x0+sb_x1)/2,   sb_y+2.4*tk, "10",    ha="center", fontsize=6.8, zorder=7)
    ax.text(sb_x1,              sb_y+2.4*tk, "20 km", ha="center", fontsize=6.8, zorder=7)

    lon_ticks = np.arange(120.6, 121.31, 0.2)
    lat_ticks = np.arange(24.0,  25.11,  0.2)
    xt = [tr.transform(lo, CTR_LAT)[0] for lo in lon_ticks]
    yt = [tr.transform(CTR_LON, la)[1] for la in lat_ticks]
    ax.set_xticks([x for x in xt if x_min <= x <= x_max])
    ax.set_xticklabels([f"{lo:.2f}E" for lo, x in zip(lon_ticks, xt) if x_min <= x <= x_max], fontsize=7)
    ax.set_yticks([y for y in yt if y_min <= y <= y_max])
    ax.set_yticklabels([f"{la:.2f}N" for la, y in zip(lat_ticks, yt) if y_min <= y <= y_max], fontsize=7)
    ax.tick_params(length=3, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    ax.text(0.014, 0.977, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.80, lw=0), zorder=8)

    s = df[col]
    stats_txt = (f"mean = {s.mean():.1f} {unit}\n"
                 f"s.d. = {s.std():.2f} {unit}\n"
                 f"[{s.min():.1f}, {s.max():.1f}]")
    ax.text(0.985, 0.012, stats_txt, transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.30", fc="white", alpha=0.82, lw=0.5), zorder=9)

    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(f"{label} [{unit}]", fontsize=8.5, labelpad=4)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

    ax.set_title(
        f"({letter}) {label}  |  {len(df):,} Microstations  |  2025-07-15  12 h CST (V4 region)",
        fontsize=9.5, pad=6)

draw_panel(ax_T,  "temperature", cmap_T,  norm_T,  "Air Temperature",  "C",  cax_T,  "a")
draw_panel(ax_RH, "humidity",    cmap_RH, norm_RH, "Relative Humidity", "%", cax_RH, "b")

fig.text(0.5, 0.997,
    "Spatial Distribution of MOENV IoT Microstation Measurements — V4 Expanded Study Region",
    ha="center", va="top", fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.002,
    "Data: EPA IoT SensorThings API (sta.colife.org.tw)  |  Basemap: CartoDB Positron / OpenStreetMap contributors",
    ha="center", fontsize=6.2, color="#666666")

out_pdf = FIG_DIR / "fig_iot_sensor_map.pdf"
out_png = FIG_DIR / "fig_iot_sensor_map.png"
fig.savefig(out_pdf, format="pdf")
fig.savefig(out_png, format="png", dpi=300)
print(f"[4] Saved:\n    {out_pdf}\n    {out_png}")
plt.close(fig)
print("[Done]")
