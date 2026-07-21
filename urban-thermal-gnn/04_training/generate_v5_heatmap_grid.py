"""
04_training/generate_v5_heatmap_grid.py
════════════════════════════════════════════════════════════════
Large, high-resolution (A4-page-sized) 2D heatmap grid: the same 9 V5
scenarios used in generate_v5_scene_3d_figures.py, ordered top-to-bottom by
increasing tree count, x 6 representative hours (08/10/12/14/16/18:00) --
54 panels total. Each panel is a zoomed-in, top-down UTCI heatmap
(triangulated contour fill over the real sensor grid) with real OSM
building footprints and real ETH canopy circles overlaid, using the same
Blue-Yellow-Red thermal-comfort colour scheme as the 3D figures.

Each scenario was only ever simulated for its own real assigned month (no
scenario has data for all of June/July/August), so panels are NOT a
true multi-month average -- they are each scenario's real simulated
representative day, labelled generically as a "summer (Jun-Aug)
representative value" rather than a specific month, per user direction.

Usage:
    python generate_v5_heatmap_grid.py
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon, Circle

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_v5_scene_3d_figures as g3d  # reuse prune_scene, UTCI_CMAP, paths

SCENE_IDS = [172, 182, 208, 213, 157, 148, 123, 109, 81]
HOURS = [8, 10, 12, 14, 16, 18]
HOURS_ALL = list(range(8, 19))
VMIN, VMAX = 26, 50
ZOOM_MARGIN = 1.5  # metres of padding around the UTCI-covered (sensor grid) extent


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def _scene_bounds(sc: dict, sensor_pts: np.ndarray) -> tuple[float, float, float, float]:
    """Zoomed-in bounds: tight to the UTCI-covered sensor grid extent only,
    so the heatmap fills the panel -- buildings/trees outside this extent
    are still drawn (for context) but may extend past the frame edge."""
    x0, x1 = sensor_pts[:, 0].min() - ZOOM_MARGIN, sensor_pts[:, 0].max() + ZOOM_MARGIN
    y0, y1 = sensor_pts[:, 1].min() - ZOOM_MARGIN, sensor_pts[:, 1].max() + ZOOM_MARGIN
    return x0, x1, y0, y1


def draw_heatmap_panel(ax, sc: dict, sensor_pts: np.ndarray, utci_vals: np.ndarray):
    x0, x1, y0, y1 = _scene_bounds(sc, sensor_pts)

    # heatmap: Delaunay-triangulated contour fill over the real sensor grid
    ax.tricontourf(sensor_pts[:, 0], sensor_pts[:, 1], utci_vals,
                   levels=24, cmap=g3d.UTCI_CMAP, vmin=VMIN, vmax=VMAX, zorder=1)

    # real OSM building footprints, opaque, masking the heatmap beneath them
    for b in sc["buildings"]:
        ax.add_patch(MplPolygon(_poly_xy(b["footprint"]), closed=True,
                                facecolor="#9a9a9a", edgecolor="#555555",
                                linewidth=0.3, zorder=3))
    # real ETH canopy footprints (top-down circle, real radius)
    for t in sc.get("trees", []):
        cx, cy = t["pos"]
        r = float(t.get("radius", t["height"] * 0.4))
        ax.add_patch(Circle((cx, cy), r, facecolor="#3f8f46", edgecolor="none",
                            alpha=0.75, zorder=2))

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4); spine.set_color("#999999")


def main():
    with open(g3d.SCEN, "rb") as f:
        raw_scen = {s["scenario_id"]: s for s in pickle.load(f)}
    scen = {sid: g3d.prune_scene(sc) for sid, sc in raw_scen.items()}

    rows = [(sid, len(scen[sid].get("trees", [])), len(scen[sid]["buildings"]))
            for sid in SCENE_IDS]
    rows.sort(key=lambda r: r[1])  # ascending tree count, top -> bottom
    print("[generate_v5_heatmap_grid] row order (scene, n_trees, n_bldgs):", rows)

    nrows, ncols = len(rows), len(HOURS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.27, 11.69), dpi=300)

    im_ref = None
    for r, (sid, n_tree, n_bldg) in enumerate(rows):
        sc = scen[sid]
        d = np.load(g3d.SIMDIR / f"sim_{sid:04d}.npz")
        sensor_pts = d["sensor_pts"]; utci_all = d["utci"]  # (T, N)
        for c, hr in enumerate(HOURS):
            ax = axes[r, c]
            t = HOURS_ALL.index(hr)
            draw_heatmap_panel(ax, sc, sensor_pts, utci_all[t])
            if r == 0:
                ax.set_title(f"{hr}:00", fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(f"#{sid}\n{n_bldg}棟/{n_tree}樹", fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=8)

    fig.subplots_adjust(left=0.11, right=0.90, top=0.94, bottom=0.03,
                        wspace=0.04, hspace=0.06)

    # shared colourbar
    sm = plt.cm.ScalarMappable(cmap=g3d.UTCI_CMAP,
                                norm=mpl.colors.Normalize(vmin=VMIN, vmax=VMAX))
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("UTCI (°C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("V5 Radiance+EnergyPlus 模擬：UTCI 熱力圖陣列\n"
                 "（9 場景由上到下依樹木數量遞增排序；各場景值為其真實模擬之夏季代表日，非跨月平均）",
                 fontsize=9.5, y=0.985)

    out = g3d.FIGDIR / "fig_v5_heatmap_grid_9x6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
