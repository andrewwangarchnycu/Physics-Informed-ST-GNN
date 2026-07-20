"""
11_build_dynamic_edge_cache.py
================================
Reconstructs which specific building/tree object is responsible for the
shadow / vegetation-cooling effect at each (timestep, air-node) pair, for
every scenario in scenarios.pkl -- using the exact same geometric logic as
03_lbt_batch_runner.py's `_in_shadow` (shadow polygon containment) and
`_veg_cooling` (canopy-radius containment), but recording the source OBJECT
INDEX instead of just a boolean flag.

This is a pure geometry post-processing pass over already-generated
scenario geometry + solar position -- it does NOT re-run EnergyPlus/Ladybug,
so it is cheap (a few minutes for 300 scenarios x 11 timesteps x ~6241
sensor points).

Why this exists: 02_graph_construction/dataset.py previously set
`dynamic_edges = [{}] * T` unconditionally, meaning the "shadow", "veg_et"
and "convective" RGCN relation types declared in 03_model/layers/rgcn_block.py
were never actually populated with real edges during training -- their
physical effect was folded into air-node features (shadow flag, nearest
building/tree height) instead. This script provides the missing per-edge
source information so dataset.py can build REAL dynamic edges matching the
thesis's architectural description (Ch3 Table 3).

Wind direction for the `convective` relation is not scenario-specific in
this synthetic dataset (single climate profile shared across all 300
scenarios, per 03_lbt_batch_runner.py's use of epw.get_typical_day), so it
is stored once as a single (T,) array rather than per-scenario.

Produces:
  01_data_generation/outputs/raw_simulations/dynamic_edge_cache.h5
    /scenarios/{sid}/shadow_src   (T, N) int16   -1 = not shadowed, else building-local index
    /scenarios/{sid}/veg_src      (T, N) int16   -1 = no tree in range, else tree-local index
    /wind_dir                     (T,)   float32 degrees, shared across all scenarios
"""
import sys, math, pickle, time
from pathlib import Path
import numpy as np
import h5py
from shapely.geometry import Point
from shapely.affinity import translate as sh_translate

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from shared import EPWData, HourlyClimate, solar_position

RAW_DIR      = _ROOT / "01_data_generation" / "outputs" / "raw_simulations"
SCENARIO_PKL = RAW_DIR / "scenarios.pkl"
EPW_PKL      = RAW_DIR / "epw_data.pkl"
H5_PATH      = RAW_DIR / "ground_truth_v2.h5"
OUT_H5       = RAW_DIR / "dynamic_edge_cache.h5"

LAT, LON, TZ = 24.80, 120.97, 8.0
MONTH        = 7   # matches 03_lbt_batch_runner.py main()'s default (NOT June -- verified from source)
SIM_HOURS    = list(range(8, 19))


def shadow_source(pt: Point, buildings: list, sol_alt: float, sol_az: float) -> int:
    """Same geometry as 03_lbt_batch_runner._in_shadow, but returns the
    building-local index that casts the shadow (-1 if none / sun below horizon)."""
    if sol_alt <= 2.0:
        return -1
    alt_r = math.radians(sol_alt)
    az_r  = math.radians(sol_az)
    for i, b in enumerate(buildings):
        sl  = b["height"] / math.tan(alt_r)
        sdx = -math.sin(az_r) * sl
        sdy = -math.cos(az_r) * sl
        shadow = b["footprint"].union(
            sh_translate(b["footprint"], sdx, sdy)
        ).convex_hull
        if shadow.contains(pt):
            return i
    return -1


def veg_source(pt: Point, trees: list) -> int:
    """Same geometry as 03_lbt_batch_runner._veg_cooling / _in_shadow tree
    branch, returns tree-local index of nearest enclosing canopy (-1 if none)."""
    best_i, best_d = -1, float("inf")
    for i, t in enumerate(trees):
        d = math.dist((pt.x, pt.y), t["pos"])
        r = t.get("radius", t["height"] * 0.4)
        if d <= r and d < best_d:
            best_i, best_d = i, d
    return best_i


def main():
    print("[1] Loading scenarios, EPW, and reference H5 sensor grids ...")
    with open(SCENARIO_PKL, "rb") as f:
        scenarios = pickle.load(f)
    with open(EPW_PKL, "rb") as f:
        epw: EPWData = pickle.load(f)

    scenario_map = {s["scenario_id"]: s for s in scenarios}

    # ── wind direction series (shared across all scenarios) ─────────────
    typical  = epw.get_typical_day(month=MONTH, stat="hottest")
    clim_map = {h.hour: h for h in typical}
    wind_dir = np.array(
        [clim_map[h].wind_dir if h in clim_map else 0.0 for h in SIM_HOURS],
        dtype=np.float32,
    )
    print(f"    wind_dir (deg) for hours {SIM_HOURS}: {wind_dir.tolist()}")

    # ── solar position per sim hour (same for every scenario: same site) ─
    sol = [solar_position(LAT, LON, TZ, MONTH, 21, h) for h in SIM_HOURS]
    print(f"    solar altitude/azimuth per hour: {[(round(a,1), round(z,1)) for a,z in sol]}")

    with h5py.File(H5_PATH, "r") as hf_ref, h5py.File(OUT_H5, "w") as hf_out:
        hf_out.create_dataset("wind_dir", data=wind_dir)
        grp_root = hf_out.create_group("scenarios")

        sids = list(hf_ref["scenarios"].keys())
        print(f"[2] Processing {len(sids)} scenarios ...")
        t0 = time.time()

        for k, sid_str in enumerate(sids):
            sid = int(sid_str)
            sc  = scenario_map.get(sid)
            if sc is None:
                continue
            buildings = sc["buildings"]
            trees     = sc.get("trees", [])

            sensor_pts = hf_ref[f"scenarios/{sid_str}/sensor_pts"][()]  # (N, 2)
            N = sensor_pts.shape[0]
            T = len(SIM_HOURS)

            shadow_src = np.full((T, N), -1, dtype=np.int16)
            veg_src    = np.full((T, N), -1, dtype=np.int16)

            for t, (sol_alt, sol_az) in enumerate(sol):
                for i in range(N):
                    pt = Point(sensor_pts[i, 0], sensor_pts[i, 1])
                    if buildings:
                        shadow_src[t, i] = shadow_source(pt, buildings, sol_alt, sol_az)
                    if trees:
                        veg_src[t, i] = veg_source(pt, trees)

            g = grp_root.create_group(sid_str)
            g.create_dataset("shadow_src", data=shadow_src, compression="gzip")
            g.create_dataset("veg_src",    data=veg_src,    compression="gzip")

            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {k+1}/{len(sids)} scenarios done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[3] Done in {elapsed:.1f}s. Saved: {OUT_H5}")


if __name__ == "__main__":
    main()
