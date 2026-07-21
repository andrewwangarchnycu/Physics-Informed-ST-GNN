"""
04_training/generate_v5_scene_array_figures.py
════════════════════════════════════════════════════════════════
Array-plot (grid) visualization of the V5 real Radiance+EnergyPlus
simulation output, adapted from generate_scene_array_figures.py (V4).

Produces two figures from whichever V5 scenarios already have a completed
sim_*.npz (works incrementally while the 300-scenario batch is still
running in the background):

  fig_v5_timestep_utci.png
      Scene x hour array of the real UTCI field (08->18h), same layout as
      V4's fig_timestep_utci.png, for direct visual comparison.

  fig_v5_svf_mrt_array.png
      Scene x {SVF (real Radiance ray-traced, static), MRT at 12:00 (real
      EnergyPlus-coupled longwave + Radiance shortwave)} -- the two fields
      that are now genuinely computed by external engines rather than V4's
      closed-form approximation, shown side by side per scene so the real
      spatial variance these engines produce is directly visible.

Usage:
    python generate_v5_scene_array_figures.py [--n-scenes 4]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPoly

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"
SIMDIR = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "sim"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
HALF = 40.0


def _poly_xy(fp):
    return list(fp.exterior.coords) if fp.geom_type == "Polygon" else list(fp.geoms[0].exterior.coords)


def _completed_scene_ids() -> list[int]:
    ids = sorted(int(p.stem.split("_")[1]) for p in SIMDIR.glob("sim_*.npz"))
    return ids


def _pick_readable(scen: dict, ids: list[int], n: int) -> list[int]:
    """Prefer scenes with a moderate building count (readable) and some
    diversity of assigned month, from whatever is already simulated."""
    scored = []
    for sid in ids:
        sc = scen.get(sid)
        if sc is None:
            continue
        nb = len(sc["buildings"])
        if nb < 2:
            continue
        score = -abs(nb - 8)
        scored.append((score, sid))
    scored.sort(reverse=True)
    chosen, seen_months = [], set()
    # first pass: spread across months
    for _, sid in scored:
        m = scen[sid]["assigned_month"]
        if m not in seen_months:
            chosen.append(sid); seen_months.add(m)
        if len(chosen) == n:
            break
    # top up if not enough month diversity available yet
    for _, sid in scored:
        if len(chosen) == n:
            break
        if sid not in chosen:
            chosen.append(sid)
    return sorted(chosen)[:n]


def fig_timestep_utci(scen: dict, scene_ids: list[int]):
    hours_all = list(range(8, 19))
    show_hours = [8, 10, 12, 14, 16, 18]
    fig, axes = plt.subplots(len(scene_ids), len(show_hours),
                             figsize=(2.2 * len(show_hours), 2.4 * len(scene_ids)))
    if len(scene_ids) == 1:
        axes = axes[None, :]

    vmin, vmax = 26, 50
    sc_p = None
    for r, sid in enumerate(scene_ids):
        d = np.load(SIMDIR / f"sim_{sid:04d}.npz")
        pts = d["sensor_pts"]; utci = d["utci"]  # (T, N)
        sc = scen[sid]
        for c, hr in enumerate(show_hours):
            ax = axes[r, c]
            t = hours_all.index(hr)
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
    fig.suptitle("V5 真實 Radiance+EnergyPlus 模擬：逐時 UTCI 場（08→18 時）",
                 fontsize=13, fontweight="bold")
    out = FIGDIR / "fig_v5_timestep_utci.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


def fig_svf_mrt_array(scen: dict, scene_ids: list[int]):
    hours_all = list(range(8, 19))
    t_noon = hours_all.index(12)
    fig, axes = plt.subplots(len(scene_ids), 2,
                             figsize=(6.4, 2.6 * len(scene_ids)))
    if len(scene_ids) == 1:
        axes = axes[None, :]

    svf_p = mrt_p = None
    for r, sid in enumerate(scene_ids):
        d = np.load(SIMDIR / f"sim_{sid:04d}.npz")
        pts = d["sensor_pts"]; svf = d["svf"]; mrt = d["mrt"]  # svf (N,), mrt (T,N)
        sc = scen[sid]

        ax = axes[r, 0]
        for b in sc["buildings"]:
            ax.add_patch(MplPoly(_poly_xy(b["footprint"]), closed=True,
                                 facecolor="#cccccc", edgecolor="none", alpha=0.5, zorder=1))
        svf_p = ax.scatter(pts[:, 0], pts[:, 1], c=svf, cmap="viridis",
                           vmin=0, vmax=1, s=6, zorder=3)
        ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"場景 #{sid}\n({sc['assigned_month']}月)", fontsize=9)
        if r == 0:
            ax.set_title("SVF（真實 Radiance）", fontsize=10)

        ax2 = axes[r, 1]
        for b in sc["buildings"]:
            ax2.add_patch(MplPoly(_poly_xy(b["footprint"]), closed=True,
                                  facecolor="#cccccc", edgecolor="none", alpha=0.5, zorder=1))
        mrt_p = ax2.scatter(pts[:, 0], pts[:, 1], c=mrt[t_noon], cmap="magma",
                            vmin=25, vmax=80, s=6, zorder=3)
        ax2.set_xlim(-HALF, HALF); ax2.set_ylim(-HALF, HALF); ax2.set_aspect("equal")
        ax2.set_xticks([]); ax2.set_yticks([])
        if r == 0:
            ax2.set_title("MRT 12:00（真實 EnergyPlus+Radiance）", fontsize=10)

    cb1 = fig.colorbar(svf_p, ax=axes[:, 0], fraction=0.04, pad=0.02)
    cb1.set_label("SVF [0,1]", fontsize=9)
    cb2 = fig.colorbar(mrt_p, ax=axes[:, 1], fraction=0.04, pad=0.02)
    cb2.set_label("MRT (°C)", fontsize=9)
    fig.suptitle("V5 真實模擬引擎輸出：天空可視因子（Radiance）與平均輻射溫度（EnergyPlus+Radiance）",
                 fontsize=12, fontweight="bold")
    out = FIGDIR / "fig_v5_svf_mrt_array.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=4)
    args = ap.parse_args()

    scen = {s["scenario_id"]: s for s in pickle.load(open(SCEN, "rb"))}
    ids = _completed_scene_ids()
    if len(ids) < 2:
        raise SystemExit(f"Only {len(ids)} V5 scenarios simulated so far -- wait for more.")
    chosen = _pick_readable(scen, ids, min(args.n_scenes, len(ids)))
    print(f"[generate_v5_scene_array_figures] {len(ids)} scenes available, "
          f"using {chosen}")

    fig_timestep_utci(scen, chosen)
    fig_svf_mrt_array(scen, chosen)
