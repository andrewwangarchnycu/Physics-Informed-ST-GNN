"""
04_training/generate_v5_3d_grid.py
════════════════════════════════════════════════════════════════
3D counterpart of generate_v5_heatmap_grid.py: the exact same 9 scenarios
(ordered top-to-bottom by increasing tree count) x 6 hours (08/10/12/14/16/
18:00) x the same tight per-scene zoom (to the real UTCI-covered sensor
grid extent), but rendered as the 3D massing view (real OSM building
volumes, real ETH canopy ellipsoids, sensor points coloured by UTCI with
view-occlusion fading) instead of a 2D heatmap.

Usage:
    python generate_v5_3d_grid.py
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v5_scene_3d_figures as g3d
import generate_v5_heatmap_grid as ghm  # reuse scene order / zoom-bounds logic

SCENE_IDS = ghm.SCENE_IDS
HOURS = ghm.HOURS
HOURS_ALL = ghm.HOURS_ALL
VMIN, VMAX = ghm.VMIN, ghm.VMAX
ZOOM_MARGIN = ghm.ZOOM_MARGIN


def _sensor_bounds(sensor_pts: np.ndarray) -> tuple[float, float, float, float]:
    x0, x1 = sensor_pts[:, 0].min() - ZOOM_MARGIN, sensor_pts[:, 0].max() + ZOOM_MARGIN
    y0, y1 = sensor_pts[:, 1].min() - ZOOM_MARGIN, sensor_pts[:, 1].max() + ZOOM_MARGIN
    return x0, x1, y0, y1


def main():
    with open(g3d.SCEN, "rb") as f:
        raw_scen = {s["scenario_id"]: s for s in pickle.load(f)}
    scen = {sid: g3d.prune_scene(sc) for sid, sc in raw_scen.items()}

    rows = [(sid, len(scen[sid].get("trees", [])), len(scen[sid]["buildings"]))
            for sid in SCENE_IDS]
    rows.sort(key=lambda r: r[1])
    print("[generate_v5_3d_grid] row order (scene, n_trees, n_bldgs):", rows)

    nrows, ncols = len(rows), len(HOURS)
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)

    sc_p = None
    for r, (sid, n_tree, n_bldg) in enumerate(rows):
        sc = scen[sid]
        d = np.load(g3d.SIMDIR / f"sim_{sid:04d}.npz")
        sensor_pts = d["sensor_pts"]; utci_all = d["utci"]
        xlim = _sensor_bounds(sensor_pts)[:2]
        ylim = _sensor_bounds(sensor_pts)[2:]
        for c, hr in enumerate(HOURS):
            ax = fig.add_subplot(nrows, ncols, r * ncols + c + 1, projection="3d")
            t = HOURS_ALL.index(hr)
            sc_p = g3d.draw_scene_3d(ax, sc, sensor_pts, utci_all[t], VMIN, VMAX, hr,
                                     xlim=xlim, ylim=ylim, show_title=False)
            if r == 0:
                ax.set_title(f"{hr}:00", fontsize=8, pad=-6)
            if c == 0:
                ax.text2D(-0.05, 0.5, f"#{sid}\n{n_bldg}棟/{n_tree}樹", fontsize=6.5,
                         ha="right", va="center", transform=ax.transAxes)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.02,
                        wspace=-0.15, hspace=0.15)

    sm = plt.cm.ScalarMappable(cmap=g3d.UTCI_CMAP,
                                norm=mpl.colors.Normalize(vmin=VMIN, vmax=VMAX))
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("UTCI (°C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("V5 Radiance+EnergyPlus 模擬：立體場景陣列\n"
                 "（9 場景由上到下依樹木數量遞增排序；各場景值為其真實模擬之夏季代表日，非跨月平均）",
                 fontsize=9.5, y=0.985)

    out = g3d.FIGDIR / "fig_v5_3d_grid_9x6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
