"""
04_training/generate_random_scenarios_figure.py
════════════════════════════════════════════════════════════════
Regenerate fig_geo4_scenarios.png (six typical procedurally-generated
V4-baseline scenes: building massing + tree placement) using the same
visual style as fig_v5_heatmap_grid_9x6.png (generate_v5_heatmap_grid.py) --
flat grey building footprints (#9a9a9a / edge #555555) and real-radius
green canopy circles (#3f8f46, alpha 0.75) on a white ground.

Scenes are restricted to the subset whose tree canopies do not
geometrically overlap any building footprint, then 6 are picked to span
the 2-5 building / 2-5 tree range described in the text.

Usage:
    python generate_random_scenarios_figure.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon, Circle, Rectangle
from shapely.geometry import Point

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "scenarios.pkl"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

# hand-picked to span the 2-5 building / 2-5 tree range, all overlap-free
SCENE_IDS = [78, 103, 7, 166, 67, 42]


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def _no_overlap(sc: dict) -> bool:
    for t in sc["trees"]:
        p = Point(t["pos"]).buffer(t.get("radius", 3.0))
        for b in sc["buildings"]:
            if p.intersects(b["footprint"]):
                return False
    return True


def draw_scene(ax, sc: dict):
    x0, y0, x1, y1 = sc["site_polygon"].bounds
    for b in sc["buildings"]:
        ax.add_patch(MplPolygon(_poly_xy(b["footprint"]), closed=True,
                                facecolor="#9a9a9a", edgecolor="#555555",
                                linewidth=0.6, zorder=3))
    for t in sc["trees"]:
        cx, cy = t["pos"]
        r = float(t.get("radius", t["height"] * 0.4))
        ax.add_patch(Circle((cx, cy), r, facecolor="#3f8f46", edgecolor="none",
                            alpha=0.75, zorder=2))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="#999999", lw=0.4, ls="--", zorder=1))
    ax.set_xlim(x0 - 3, x1 + 3); ax.set_ylim(y0 - 3, y1 + 3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4); spine.set_color("#999999")


def main():
    with open(SCEN, "rb") as f:
        raw = {s["scenario_id"]: s for s in pickle.load(f)}

    for sid in SCENE_IDS:
        assert _no_overlap(raw[sid]), f"scenario {sid} has tree/building overlap"

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), dpi=300)
    for ax, sid in zip(axes.ravel(), SCENE_IDS):
        sc = raw[sid]
        draw_scene(ax, sc)
        ax.set_title(f"#{sid} {len(sc['buildings'])}棟/{len(sc['trees'])}樹",
                     fontsize=9, pad=4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                        wspace=0.08, hspace=0.18)
    fig.suptitle("六個典型隨機生成場景之建築量體與喬木配置示意", fontsize=13, y=0.97)

    out = FIGDIR / "fig_geo4_scenarios.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
