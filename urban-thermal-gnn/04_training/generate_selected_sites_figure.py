"""
04_training/generate_selected_sites_figure.py
════════════════════════════════════════════════════════════════
Figure: the real-world training sites selected by
01_data_generation/scripts/06_select_real_sites_v4.py, drawn to scale
on a real OSM basemap.

Every plotted box is a genuine 80x80 m site that simultaneously passed
all four selection criteria (real OSM buildings w/ footprints & heights,
>= 2 distinct physical surface classes, an ETH canopy-height signal, and
proximity to a dense cluster of real MOENV IoT stations). Nothing is
padded or synthetic -- the count shown is exactly what qualified.

Panels:
  (a) Regional distribution across the expanded search area (Hsinchu
      City/County, Taoyuan, Miaoli, Taichung), anchored on NYCU Guangfu
      Campus, points colour-coded by selection score.
  (b) Zoom on the NYCU-centred core, 80x80 m site boxes drawn to scale.

Usage:
    python generate_selected_sites_figure.py \
        --sites ../01_data_generation/outputs/real_sites_v4/selected_real_sites.json
"""
from __future__ import annotations

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

try:
    import contextily as cx
    from pyproj import Transformer
    _BASEMAP = True
except ImportError:
    _BASEMAP = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ANCHOR_LAT, ANCHOR_LON = 24.78805, 120.99754
M_PER_DEG_LAT = 111_320.0


def _to_web_mercator(lon, lat):
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return tr.transform(lon, lat)


def main(sites_json: str, out_png: str):
    data = json.loads(Path(sites_json).read_text(encoding="utf-8"))
    sites = data.get("sites", [])
    n = len(sites)
    if n == 0:
        print("[fig] no sites in JSON -- nothing to draw"); return

    lats = np.array([s["lat"] for s in sites])
    lons = np.array([s["lon"] for s in sites])
    scores = np.array([s.get("score", 0.0) for s in sites])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7.2))

    # ── Panel (a): regional distribution ────────────────────────
    xs, ys = _to_web_mercator(lons, lats)
    ax_lon, ax_lat = _to_web_mercator(ANCHOR_LON, ANCHOR_LAT)
    scA = axA.scatter(xs, ys, c=scores, cmap="viridis", s=55,
                      edgecolor="white", linewidth=0.6, zorder=5)
    axA.scatter([ax_lon], [ax_lat], marker="*", s=420, c="red",
                edgecolor="black", linewidth=1.0, zorder=6,
                label="NYCU 光復校區 (錨點)")
    pad = 6000
    axA.set_xlim(xs.min() - pad, xs.max() + pad)
    axA.set_ylim(ys.min() - pad, ys.max() + pad)
    if _BASEMAP:
        try:
            cx.add_basemap(axA, source=cx.providers.CartoDB.Positron,
                           crs="EPSG:3857", attribution_size=5)
        except Exception as e:
            print(f"[fig] basemap (a) failed: {e}")
    cb = fig.colorbar(scA, ax=axA, fraction=0.046, pad=0.02)
    cb.set_label("選址評分 (建物數 + IoT密度 + 樹冠)", fontsize=9)
    axA.set_title(f"(a) {n} 個真實訓練場址之區域分布\n(新竹/桃園/苗栗/台中，錨定陽明交大)", fontsize=11)
    axA.legend(loc="upper right", fontsize=9)
    axA.set_xticks([]); axA.set_yticks([])

    # ── Panel (b): NYCU-core zoom, 80x80 m boxes to scale ───────
    zoom_m = 6000.0
    dlat = zoom_m / M_PER_DEG_LAT
    dlon = zoom_m / (M_PER_DEG_LAT * np.cos(np.radians(ANCHOR_LAT)))
    core = [(s, abs(s["lat"] - ANCHOR_LAT) < dlat and abs(s["lon"] - ANCHOR_LON) < dlon)
            for s in sites]
    core_sites = [s for s, keep in core if keep]

    half = 40.0  # 80x80 m
    for s in core_sites:
        cx_m, cy_m = _to_web_mercator(s["lon"], s["lat"])
        # metres->mercator scale factor at this latitude
        sf = 1.0 / np.cos(np.radians(s["lat"]))
        w = 2 * half * sf
        axB.add_patch(mpatches.Rectangle(
            (cx_m - half * sf, cy_m - half * sf), w, w,
            facecolor="none", edgecolor="#d1495b", linewidth=1.6, zorder=5))
    axB.scatter([ax_lon], [ax_lat], marker="*", s=420, c="red",
                edgecolor="black", linewidth=1.0, zorder=6)
    zpad = _to_web_mercator(ANCHOR_LON + dlon, ANCHOR_LAT + dlat)
    axB.set_xlim(2 * ax_lon - zpad[0], zpad[0])
    axB.set_ylim(2 * ax_lat - zpad[1], zpad[1])
    if _BASEMAP:
        try:
            cx.add_basemap(axB, source=cx.providers.CartoDB.Positron,
                           crs="EPSG:3857", attribution_size=5)
        except Exception as e:
            print(f"[fig] basemap (b) failed: {e}")
    axB.set_title(f"(b) 錨點核心區 {len(core_sites)} 個場址\n(80x80 m 邊界按比例繪製)", fontsize=11)
    axB.set_xticks([]); axB.set_yticks([])

    fig.suptitle(f"PI-ST-GNN V4 真實訓練場址選取 (共 {n} 個合格場址)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[fig] saved {out}  ({n} sites, {len(core_sites)} in core zoom)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=str(
        HERE.parent / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"))
    ap.add_argument("--out", default=str(HERE / "figures" / "fig_selected_real_sites.png"))
    args = ap.parse_args()
    main(args.sites, args.out)
