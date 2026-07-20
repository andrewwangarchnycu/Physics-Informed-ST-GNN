"""
04_training/generate_scene_array_figures.py
════════════════════════════════════════════════════════════════
Two figures for the V4 real-scene training data:

  fig_scene_array_25.png
      A 5x5 small-multiples array of 25 clear, readable real OSM scenes
      selected from the 300, each showing its genuine OSM building
      footprints (shaded by height) and ETH canopy trees within the
      80x80 m site -- a visual survey of the real geometric diversity the
      model trains on.

  fig_timestep_utci.png
      A scene x time-step array visualising how the thermal-comfort
      training label (UTCI field) is generated: for a few representative
      real scenes, the per-node UTCI is drawn at successive daytime hours
      (08->18 h), showing the diurnal build-up and decay of heat stress
      and the persistent spatial structure imposed by real buildings and
      vegetation.

Usage:
    python generate_scene_array_figures.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly, Rectangle
from matplotlib.collections import PatchCollection

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"
SIMDIR = ROOT / "01_data_generation" / "outputs" / "real_simulations_v4"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
HALF = 40.0


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def _readability(sc):
    """Score a scene for 'clear & readable': prefer 4-14 buildings that sit
    mostly inside the 80x80 m site, plus some canopy."""
    b_in = [b for b in sc["buildings"]
            if abs(b["footprint"].centroid.x) < HALF and abs(b["footprint"].centroid.y) < HALF]
    nb = len(b_in)
    if nb < 3:
        return -1
    # penalty for far from the ideal 8 buildings, reward some trees
    return -abs(nb - 8) + 0.02 * min(sc.get("n_trees_used", 0), 30)


def draw_scene(ax, sc, show_axes=False):
    heights = [b["height"] for b in sc["buildings"]] or [1.0]
    hmax = max(max(heights), 12.0)
    cmap = plt.cm.YlOrBr
    patches, colors = [], []
    for b in sc["buildings"]:
        xy = _poly_xy(b["footprint"])
        patches.append(MplPoly(xy, closed=True))
        colors.append(0.2 + 0.75 * b["height"] / hmax)
    pc = PatchCollection(patches, cmap=cmap, edgecolor="#5a3d00", linewidths=0.4, alpha=0.9)
    pc.set_array(np.array(colors)); pc.set_clim(0, 1)
    ax.add_collection(pc)
    for t in sc["trees"][:50]:
        x, y = t["pos"]
        ax.scatter([x], [y], s=max(3, t["height"] * 1.6), c="#2e8b57",
                   alpha=0.5, edgecolor="none", zorder=4)
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                           edgecolor="#333", lw=1.0, ls="--"))
    ax.set_xlim(-HALF - 4, HALF + 4); ax.set_ylim(-HALF - 4, HALF + 4)
    ax.set_aspect("equal")
    if not show_axes:
        ax.set_xticks([]); ax.set_yticks([])


def fig_scene_array():
    scen = pickle.load(open(SCEN, "rb"))
    ranked = sorted(scen, key=_readability, reverse=True)
    # take 25, spread across the ranking for variety (not just the 25 closest to 8)
    picks = ranked[:60]
    idx = np.linspace(0, len(picks) - 1, 25).astype(int)
    chosen = [picks[i] for i in idx]

    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    for ax, sc in zip(axes.ravel(), chosen):
        draw_scene(ax, sc)
        nb = len(sc["buildings"])
        ax.set_title(f"#{sc['scenario_id']} · {nb}棟 · {sc['assigned_month']}月",
                     fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrBr, norm=mpl.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("相對建物簷高（低→高）", fontsize=10)
    fig.suptitle("25 個真實 OSM 訓練場景（5×5）：真實建物足跡（色階=簷高）＋ ETH 樹冠",
                 fontsize=14, fontweight="bold")
    out = FIGDIR / "fig_scene_array_25.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)
    return [sc["scenario_id"] for sc in chosen]


def fig_timestep(scene_ids):
    """scene x time-step UTCI array from the simulation npz."""
    # pick 4 representative scenes that have npz, with buildings
    scen = {s["scenario_id"]: s for s in pickle.load(open(SCEN, "rb"))}
    hours_all = list(range(8, 19))
    show_hours = [8, 10, 12, 14, 16, 18]
    chosen = []
    for sid in scene_ids:
        npz = SIMDIR / f"sim_{sid:04d}.npz"
        if npz.exists():
            chosen.append(sid)
        if len(chosen) == 4:
            break

    fig, axes = plt.subplots(len(chosen), len(show_hours),
                             figsize=(2.2 * len(show_hours), 2.4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    # shared UTCI colour scale
    vmin, vmax = 30, 48
    for r, sid in enumerate(chosen):
        d = np.load(SIMDIR / f"sim_{sid:04d}.npz")
        pts = d["sensor_pts"]; utci = d["utci"]  # (T, N)
        sc = scen[sid]
        for c, hr in enumerate(show_hours):
            ax = axes[r, c]
            t = hours_all.index(hr)
            # faint building context
            for b in sc["buildings"]:
                ax.add_patch(MplPoly(_poly_xy(b["footprint"]), closed=True,
                                     facecolor="#cccccc", edgecolor="none", alpha=0.5, zorder=1))
            sc_p = ax.scatter(pts[:, 0], pts[:, 1], c=utci[t], cmap="inferno",
                              vmin=vmin, vmax=vmax, s=6, zorder=3)
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{hr}:00", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"場景 #{sid}\n({sc['assigned_month']}月)", fontsize=9)
    cb = fig.colorbar(sc_p, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("UTCI (°C)", fontsize=10)
    fig.suptitle("熱舒適訓練資料生成：真實場景之逐時 UTCI 場（08→18 時）",
                 fontsize=13, fontweight="bold")
    out = FIGDIR / "fig_timestep_utci.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    ids = fig_scene_array()
    fig_timestep(ids)
