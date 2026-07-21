"""
01_data_generation/scripts/16_build_dynamic_edge_cache_v5.py
════════════════════════════════════════════════════════════════
V5 dynamic-edge source cache -- fixes the pre-existing gap (shared by V4
and V5's earlier 39/150-scenario runs) where `dataset.py` silently fell
back to `dynamic_edges = [{}] * T` because no `dynamic_edge_cache.h5` was
ever built next to any real-scenario ground_truth*.h5. Without this file,
the "shadow", "veg_et" and "convective" RGCN relation types declared in
the model architecture (thesis Ch3 Table 3) are structurally present but
carry zero real edges during training.

This is a pure geometry post-processing pass (no re-simulation): for every
V5 scenario it recomputes, per sensor point and per hour, which building
casts its shadow and which tree's canopy covers it, using the exact same
containment logic as the physics core (`03_lbt_batch_runner._in_shadow`),
but with two real-data improvements over the original (never-actually-used
for real scenarios) 11_build_dynamic_edge_cache.py:

  * Real per-scenario solar position (site_lat/site_lon + assigned_month),
    not a single hardcoded lat/lon/month shared by every scenario.
  * Real per-month CWB wind direction (cwb_data_6/7/8.csv), not a single
    fixed value.

Building/tree indices are computed against the RAW (unpruned) scenario
building/tree lists -- the same lists `dataset.py`'s `_extract_object_features`
uses to build object nodes -- so `shadow_src`/`veg_src` indices line up with
the object-node ordering `_build_dynamic_edges` expects. This matches the
existing (V4-inherited) object-node convention; it is a separate, larger
question (not addressed here) that dense scenarios' object nodes include
buildings beyond the nearest-30 subset actually fed to Radiance/EnergyPlus.

Produces:
  01_data_generation/outputs/real_simulations_v5/dynamic_edge_cache.h5
    /wind_dir                     (T,)   float32 degrees -- mean real CWB
                                          wind direction across June/July/
                                          August, hours 8-18 (shared series;
                                          dataset.py's _build_dynamic_edges
                                          only supports one global series)
    /scenarios/{sid}/shadow_src   (T, N) int16   -1 = not shadowed, else
                                          building-local index into
                                          scenario["buildings"]
    /scenarios/{sid}/veg_src      (T, N) int16   -1 = no tree in range, else
                                          tree-local index into
                                          scenario["trees"] (time-invariant,
                                          broadcast across all T hours)
"""
from __future__ import annotations

import sys
import time
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import h5py
import shapely
from shapely.geometry import Polygon
from shapely.affinity import translate as sh_translate
from shapely import STRtree

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from shared import solar_position

_spec = importlib.util.spec_from_file_location(
    "v4_runner", str(_HERE / "09_run_real_sim_v4.py"))
_v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v4)
load_cwb_monthly = _v4.load_cwb_monthly

SCEN_PKL = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"
H5_PATH  = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "ground_truth_v5.h5"
OUT_H5   = _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "dynamic_edge_cache.h5"
SIM_HOURS = list(range(8, 19))
TZ = 8.0


def shadow_src_for_hour(sensor_pts: np.ndarray, buildings: list,
                        sol_alt: float, sol_az: float) -> np.ndarray:
    """(N,) int16 array of building-local index shadowing each point, or -1."""
    n = len(sensor_pts)
    out = np.full(n, -1, dtype=np.int16)
    if sol_alt <= 2.0 or not buildings:
        return out  # sun below horizon (or no buildings) -> everything "shadowed"/none
    alt_r = np.radians(sol_alt)
    az_r = np.radians(sol_az)
    polys, idxs = [], []
    for i, b in enumerate(buildings):
        sl = b["height"] / np.tan(alt_r)
        sdx = -np.sin(az_r) * sl
        sdy = -np.cos(az_r) * sl
        shadow = b["footprint"].union(sh_translate(b["footprint"], sdx, sdy)).convex_hull
        if shadow.is_empty:
            continue
        polys.append(shadow)
        idxs.append(i)
    if not polys:
        return out
    idxs = np.array(idxs)
    tree = STRtree(polys)
    points = shapely.points(sensor_pts[:, 0], sensor_pts[:, 1])
    pt_idx, poly_idx = tree.query(points, predicate="within")
    if len(pt_idx) == 0:
        return out
    # a point may fall inside multiple buildings' shadows; keep the first
    # in original building order (lowest source index), matching the
    # sequential-scan behaviour of the physics core's own shadow test.
    order = np.lexsort((idxs[poly_idx], pt_idx))
    pt_idx_sorted, poly_idx_sorted = pt_idx[order], poly_idx[order]
    _, first_pos = np.unique(pt_idx_sorted, return_index=True)
    out[pt_idx_sorted[first_pos]] = idxs[poly_idx_sorted[first_pos]].astype(np.int16)
    return out


def veg_src_static(sensor_pts: np.ndarray, trees: list) -> np.ndarray:
    """(N,) int16 array of tree-local index covering each point (nearest
    enclosing canopy), or -1. Time-invariant (canopy position doesn't move)."""
    n = len(sensor_pts)
    out = np.full(n, -1, dtype=np.int16)
    if not trees:
        return out
    tpos = np.array([t["pos"] for t in trees])              # (M, 2)
    trad = np.array([t.get("radius", t["height"] * 0.4) for t in trees])  # (M,)
    d = np.linalg.norm(sensor_pts[:, None, :] - tpos[None, :, :], axis=2)  # (N, M)
    within = d <= trad[None, :]
    d_masked = np.where(within, d, np.inf)
    best = np.argmin(d_masked, axis=1)
    has_any = within.any(axis=1)
    out[has_any] = best[has_any].astype(np.int16)
    return out


def real_wind_dir_series(cwb: dict, months=(6, 7, 8)) -> np.ndarray:
    """Mean real CWB wind direction per sim hour, averaged across the
    three summer months (circular mean, since direction wraps at 360)."""
    out = np.zeros(len(SIM_HOURS), dtype=np.float32)
    for i, hr in enumerate(SIM_HOURS):
        wds = [cwb[m][hr]["wd"] for m in months if m in cwb and np.isfinite(cwb[m][hr]["wd"])]
        if not wds:
            out[i] = 180.0
            continue
        rad = np.radians(wds)
        out[i] = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0
    return out


def main():
    print("[1] Loading scenarios + real CWB wind climatology ...")
    with open(SCEN_PKL, "rb") as f:
        scenarios = pickle.load(f)
    scen_map = {s["scenario_id"]: s for s in scenarios}
    cwb = load_cwb_monthly()
    wind_dir = real_wind_dir_series(cwb)
    print(f"    real mean wind_dir (deg) for hours {SIM_HOURS}: {wind_dir.round(1).tolist()}")

    with h5py.File(H5_PATH, "r") as hf_ref, h5py.File(OUT_H5, "w") as hf_out:
        hf_out.create_dataset("wind_dir", data=wind_dir)
        grp_root = hf_out.create_group("scenarios")

        sids = list(hf_ref["scenarios"].keys())
        print(f"[2] Processing {len(sids)} scenarios ...")
        t0 = time.time()

        for k, sid_str in enumerate(sids):
            sid = int(sid_str)
            sc = scen_map.get(sid)
            if sc is None:
                continue
            buildings = sc["buildings"]
            trees = sc.get("trees", [])
            sensor_pts = hf_ref[f"scenarios/{sid_str}/sensor_pts"][()]  # (N, 2)
            N = sensor_pts.shape[0]
            T = len(SIM_HOURS)

            shadow_src = np.full((T, N), -1, dtype=np.int16)
            for t, hr in enumerate(SIM_HOURS):
                sol_alt, sol_az = solar_position(sc["site_lat"], sc["site_lon"], TZ,
                                                  sc["assigned_month"], 15, hr)
                shadow_src[t] = shadow_src_for_hour(sensor_pts, buildings, sol_alt, sol_az)

            veg_src_t0 = veg_src_static(sensor_pts, trees)
            veg_src = np.tile(veg_src_t0, (T, 1))

            g = grp_root.create_group(sid_str)
            g.create_dataset("shadow_src", data=shadow_src, compression="gzip")
            g.create_dataset("veg_src", data=veg_src, compression="gzip")

            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {k+1}/{len(sids)} scenarios done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[3] Done in {elapsed:.1f}s. Saved: {OUT_H5}")


if __name__ == "__main__":
    main()
