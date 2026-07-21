"""
04_training/generate_v5_scene_3d_figures.py
════════════════════════════════════════════════════════════════
3D array-plot visualization of the actual Honeybee massing model fed into
the real Radiance+EnergyPlus simulation (V5): real OSM building footprints
extruded to their real heights (the same `Room` volumes EnergyPlus solved
for envelope surface temperature), real ETH canopy trees as translucent
discs at their real height, and the real sensor grid coloured by simulated
UTCI at a chosen hour -- so the 3D geometry that was *actually simulated*
is directly visible, not just its 2D top-down projection.

Works incrementally against whatever V5 scenarios already have a completed
sim_*.npz (the 300-scenario batch may still be running in the background).

Usage:
    python generate_v5_scene_3d_figures.py [--n-scenes 4] [--hour 12]
"""
from __future__ import annotations

import argparse
import pickle
import math
import importlib.util
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False

# Blue-Yellow-Red: the diverging colour scheme conventionally used in
# outdoor thermal-comfort literature (cool -> neutral -> hot), rather than
# a generic perceptual heat colormap like inferno.
UTCI_CMAP = LinearSegmentedColormap.from_list(
    "utci_byr", ["#2166ac", "#67a9cf", "#d1e5f0", "#fef0d9", "#fdae61", "#e31a1c", "#7f0000"])

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCEN = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"
SIMDIR = ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "sim"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
HALF = 40.0

# the same nearest-30-buildings/20-trees pruning the real simulation used
# (14_run_lbt_recipe_v5.py / 12_build_honeybee_model_v5.py both call this on
# the raw scenario before building the Honeybee model) -- reused here so the
# geometry drawn always matches what was actually fed into Radiance/
# EnergyPlus and the sensor grid's real spatial extent, not the full raw
# OSM scenario (which can hold 200+ buildings before pruning).
_spec = importlib.util.spec_from_file_location(
    "v4_runner", str(ROOT / "01_data_generation" / "scripts" / "09_run_real_sim_v4.py"))
_v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v4)
prune_scene = _v4.prune_scene
SENSOR_Z = 1.5
VIEW_ELEV = 48.0
VIEW_AZIM = -50.0


def _view_direction(elev_deg: float, azim_deg: float):
    """Unit vector from the scene toward the camera, matching matplotlib's
    view_init(elev, azim) convention."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def _occluded_by_buildings(px, py, buildings, view_dir, sensor_z=SENSOR_Z):
    """True if a building volume sits between (px,py,sensor_z) and the
    camera along the current view direction -- i.e. this point is a hidden
    surface that mplot3d's naive painter's-algorithm scatter would
    otherwise draw as if unobstructed. Approximated with the same parallel-
    projection assumption matplotlib's 3D view itself uses: walk the ray
    from the point toward the camera and see whether it passes through any
    building's footprint while still below that building's roof."""
    dx, dy, dz = view_dir
    if dz <= 1e-6:
        return False
    for b in buildings:
        t_top = (float(b["height"]) - sensor_z) / dz
        if t_top <= 1e-6:
            continue
        seg = LineString([(px, py), (px + dx * t_top, py + dy * t_top)])
        if b["footprint"].intersects(seg):
            return True
    return False


def _completed_scene_ids() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in SIMDIR.glob("sim_*.npz"))


def _pick_readable(scen: dict, ids: list[int], n: int) -> list[int]:
    scored = []
    for sid in ids:
        sc = scen.get(sid)
        if sc is None:
            continue
        nb = len(sc["buildings"])
        if nb < 2:
            continue
        scored.append((-abs(nb - 8), sid))
    scored.sort(reverse=True)
    chosen, seen_months = [], set()
    for _, sid in scored:
        m = scen[sid]["assigned_month"]
        if m not in seen_months:
            chosen.append(sid); seen_months.add(m)
        if len(chosen) == n:
            break
    for _, sid in scored:
        if len(chosen) == n:
            break
        if sid not in chosen:
            chosen.append(sid)
    return sorted(chosen)[:n]


def _pick_densest_n(scen: dict, ids: list[int], exclude: set[int], n: int) -> list[int]:
    """The N scenes with the most combined building+tree count -- busier,
    higher-information massing/canopy cases than the 'readable' picks."""
    scored = []
    for sid in ids:
        if sid in exclude:
            continue
        sc = scen.get(sid)
        if sc is None:
            continue
        score = len(sc["buildings"]) + len(sc.get("trees", []))
        scored.append((score, sid))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:n]]


def _building_wall_roof_faces(footprint, height: float):
    """Return a list of (N,3) vertex arrays: one per wall quad plus the roof."""
    coords = list(footprint.exterior.coords)[:-1]
    faces = []
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        faces.append(np.array([[x0, y0, 0], [x1, y1, 0],
                                [x1, y1, height], [x0, y0, height]]))
    roof = np.array([[x, y, height] for x, y in coords])
    faces.append(roof)
    return faces


CROWN_BASE_FRAC = 0.4   # trunk clearance below the canopy, as a fraction of tree height


def _canopy_ellipsoid_faces(cx, cy, r, canopy_top_h, n_theta=14, n_phi=8):
    """A 3D canopy ellipsoid whose TOP sits exactly at the real ETH-measured
    tree height (canopy_top_h); its base is lifted by CROWN_BASE_FRAC to
    leave a plausible clear trunk below the crown. Horizontal radius = the
    scenario's real canopy radius (from ETH pixel-derived tree extraction)."""
    z_bottom = canopy_top_h * CROWN_BASE_FRAC
    rz = (canopy_top_h - z_bottom) / 2.0
    cz = (canopy_top_h + z_bottom) / 2.0
    thetas = np.linspace(0, 2 * np.pi, n_theta + 1)
    phis = np.linspace(0, np.pi, n_phi + 1)
    grid = np.zeros((n_phi + 1, n_theta + 1, 3))
    for i, phi in enumerate(phis):
        for j, theta in enumerate(thetas):
            grid[i, j] = [cx + r * np.sin(phi) * np.cos(theta),
                          cy + r * np.sin(phi) * np.sin(theta),
                          cz + rz * np.cos(phi)]
    faces = []
    for i in range(n_phi):
        for j in range(n_theta):
            faces.append([grid[i, j], grid[i, j + 1], grid[i + 1, j + 1], grid[i + 1, j]])
    return faces


def draw_scene_3d(ax, sc: dict, sensor_pts, utci_vals, vmin, vmax, hour: int,
                   xlim=None, ylim=None, show_title=True):
    bldg_heights = [float(b["height"]) for b in sc["buildings"]]
    tree_heights = [float(t["height"]) for t in sc.get("trees", [])]
    z_top = max(bldg_heights + tree_heights + [10.0]) * 1.15

    for b in sc["buildings"]:
        h = float(b["height"])
        for face in _building_wall_roof_faces(b["footprint"], h):
            poly = Poly3DCollection([face], facecolor="#b0b0b0", edgecolor="#666666",
                                     linewidth=0.3, alpha=0.9)
            ax.add_collection3d(poly)

    # trees drawn BEFORE the UTCI sensor points, semi-transparent so the
    # heatmap layer underneath still shows through the canopy.
    for t in sc.get("trees", []):
        cx, cy = t["pos"]
        r = float(t.get("radius", t["height"] * 0.4))
        h = float(t["height"])  # real ETH GlobalCanopyHeight top height
        faces = _canopy_ellipsoid_faces(cx, cy, r, h)
        poly = Poly3DCollection(faces, facecolor="#4a8f4a", edgecolor="#2f6b2f",
                                linewidth=0.15, alpha=0.35, zorder=3)
        ax.add_collection3d(poly)

    # sensor points: this is a perspective view, and matplotlib's 3D scatter
    # does not actually occlude points hidden behind a building's solid
    # volume -- it would otherwise draw a point "through" the building wall
    # at full UTCI colour. Points whose line of sight to the camera passes
    # through a building are instead desaturated (grey) and faded (low
    # alpha) to read as hidden-behind-solid, giving a correct sense of
    # front/back depth; every other point keeps its normal, full-opacity
    # UTCI colour.
    view_dir = _view_direction(VIEW_ELEV, VIEW_AZIM)
    occluded = np.array([_occluded_by_buildings(px, py, sc["buildings"], view_dir)
                         for px, py in sensor_pts], dtype=bool)
    visible = ~occluded

    sc_p = ax.scatter(sensor_pts[visible, 0], sensor_pts[visible, 1],
                      np.full(visible.sum(), SENSOR_Z), c=utci_vals[visible],
                      cmap=UTCI_CMAP, vmin=vmin, vmax=vmax, s=6, alpha=0.95,
                      depthshade=False, zorder=4, label="可見")
    if occluded.any():
        ax.scatter(sensor_pts[occluded, 0], sensor_pts[occluded, 1],
                  np.full(occluded.sum(), SENSOR_Z), color="#808080",
                  edgecolor="none", s=7, alpha=0.55, depthshade=False,
                  zorder=2, label="被建物遮蔽")

    x0, x1 = xlim if xlim is not None else (-HALF, HALF)
    y0, y1 = ylim if ylim is not None else (-HALF, HALF)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0, z_top)
    ax.set_box_aspect((x1 - x0, y1 - y0, z_top))
    ax.set_axis_off()   # no bounding-box panes/grid/ticks, just the geometry
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    if show_title:
        ax.set_title(f"場景 #{sc['scenario_id']}（{sc['assigned_month']}月）  {hour}:00  "
                     f"最高{max(bldg_heights):.0f}m  {len(sc['buildings'])}棟/{len(sc.get('trees', []))}樹",
                     fontsize=10)
    return sc_p


def fig_scene_3d_array(scen: dict, scene_ids: list[int], hour: int):
    hours_all = list(range(8, 19))
    t = hours_all.index(hour)
    n = len(scene_ids)
    ncols = 3 if n >= 5 else min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6.5 * ncols, 6.0 * nrows))

    vmin, vmax = 26, 50
    sc_p = None
    for i, sid in enumerate(scene_ids):
        d = np.load(SIMDIR / f"sim_{sid:04d}.npz")
        pts = d["sensor_pts"]; utci = d["utci"][t]
        sc = scen[sid]
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        sc_p = draw_scene_3d(ax, sc, pts, utci, vmin, vmax, hour)

    cb = fig.colorbar(sc_p, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.7)
    cb.set_label("UTCI (°C)", fontsize=10)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#fdae61",
              markeredgecolor="none", markersize=7, label="視角可見（UTCI 色階）"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#aaaaaa",
              markeredgecolor="none", alpha=0.5, markersize=7, label="被建物遮蔽（灰階＋降低透明度）"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("V5 Radiance+EnergyPlus 模擬：OSM 建物量體＋ETH Canopy Height 樹冠＋UTCI 格點",
                 fontsize=13, fontweight="bold")
    out = FIGDIR / f"fig_v5_scene_3d_array_{hour:02d}h.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=9)
    ap.add_argument("--n-readable", type=int, default=4)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--scene-ids", type=str, default=None,
                    help="comma-separated scenario ids to force (skips auto-selection, "
                         "e.g. to reuse the exact same scenes/order across hours)")
    args = ap.parse_args()

    with open(SCEN, "rb") as f:
        raw_scen = {s["scenario_id"]: s for s in pickle.load(f)}
    # prune every scenario to the nearest 30 buildings / 20 trees, exactly
    # as the real simulation did, so the geometry shown always matches the
    # sensor grid's real spatial extent (a raw OSM scenario can hold 200+
    # buildings before pruning, of which only the nearest 30 were actually
    # simulated).
    scen = {sid: prune_scene(sc) for sid, sc in raw_scen.items()}
    ids = _completed_scene_ids()
    if len(ids) < 1:
        raise SystemExit("No V5 scenarios simulated yet -- wait for the batch to progress.")

    if args.scene_ids:
        chosen = [int(x) for x in args.scene_ids.split(",")]
        missing = [sid for sid in chosen if sid not in set(ids)]
        if missing:
            raise SystemExit(f"scenes not simulated yet: {missing}")
        print(f"[generate_v5_scene_3d_figures] using forced scene list: {chosen}")
    else:
        n_total = min(args.n_scenes, len(ids))
        n_readable = max(0, min(args.n_readable, n_total))
        n_dense = n_total - n_readable
        chosen = _pick_readable(scen, ids, n_readable)
        dense_ids = _pick_densest_n(scen, ids, exclude=set(chosen), n=n_dense)
        chosen = chosen + dense_ids
        print(f"[generate_v5_scene_3d_figures] {len(ids)} scenes available, using {chosen} "
              f"({n_readable} readable + dense scenes: {dense_ids})")

    fig_scene_3d_array(scen, chosen, args.hour)
