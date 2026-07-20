"""
01_data_generation/scripts/14_run_lbt_recipe_v5.py
════════════════════════════════════════════════════════════════
V5 recipe runner -- the real Radiance/EnergyPlus replacement for V4's
custom SVF/shadow/MRT physics core.

For each V5 scenario (HBJSON from 12_build_honeybee_model_v5.py, custom EPW
from 13_build_scenario_epw_v5.py) this runs two official, maintained
Ladybug Tools Honeybee recipes locally (via lbt-recipes' queenbee-local
runner -- no cloud):

  * "utci_comfort_map": the real end-to-end outdoor comfort pipeline --
    Radiance (enhanced 2-phase) for shortwave MRT + real EnergyPlus for
    building envelope surface temperatures feeding longwave MRT, combined
    via ladybug-comfort's OutdoorSolarCal, then UTCI. Restricted to a
    single representative day (the 15th) of the scenario's assigned month,
    hours 8-18 -- the same "typical day, 8-18h" window V4 used, so T=11
    exactly as in V4's sim_hours.
  * "sky_view": a dedicated Radiance sky-view-factor calculation on the
    same sensor grid (fast, point-in-time), giving a real per-sensor SVF
    to replace V4's ray-cast approximation.

mean_radiant_temperature = longwave_mrt + shortwave_mrt is ladybug-comfort's
own decomposition (see OutdoorSolarCal.mean_radiant_temperature); both
components are read directly from the recipe's intermediate outputs so V5's
`mrt` field is assembled from the recipe's own real outputs, not
re-derived. UTCI is taken directly from the recipe's own result (real
Radiance+EnergyPlus driven), not recomputed.

Output: one .npz per scenario in real_simulations_v5/sim/, in the same
schema as V4's sim_*.npz (sensor_pts, ta, mrt, va, rh, utci, svf,
in_shadow, building_height, tree_height, land_cover, sim_hours,
scenario_id, far, bcr, n_buildings, sensor_utci, sensor_mask,
assigned_month) so 15_output_to_hdf5_v5.py needs no schema changes.
"""
from __future__ import annotations

import sys
import os
_venv_scripts = str(os.path.dirname(sys.executable))
os.environ['PATH'] = _venv_scripts + os.pathsep + os.environ.get('PATH', '')

import json
import math
import time
import pickle
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from lbt_recipes.recipe import Recipe, RecipeSettings

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

_spec2 = importlib.util.spec_from_file_location(
    "v4_runner", str(_SCRIPT_DIR / "09_run_real_sim_v4.py"))
_v4 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_v4)
prune_scene = _v4.prune_scene

SIM_HOURS = list(range(8, 19))          # matches V4
IN_SHADOW_SW_THRESHOLD = 3.0            # deg C shortwave MRT delta
QB_PATH = str(Path(sys.executable).parent / "queenbee.exe")


def _static_fields(scp, sensor_pts):
    """building_height / tree_height / land_cover per sensor -- same logic
    as 03_lbt_batch_runner.simulate_one (V4)."""
    N = len(sensor_pts)
    bh = np.zeros(N, dtype=np.float32)
    th = np.zeros(N, dtype=np.float32)
    lc = np.zeros(N, dtype=np.int32)
    land_cover_map = scp.get("land_cover_map") or {}
    for i, (px, py) in enumerate(sensor_pts):
        pt = Point(px, py)
        md = np.inf
        for b in scp["buildings"]:
            d = b["footprint"].distance(pt)
            if d < md:
                md = d
                bh[i] = b["height"]
        lc[i] = land_cover_map.get((round(px), round(py)), 0)
        for t in scp["trees"]:
            if math.dist((px, py), t["pos"]) <= t.get("radius", 3.0):
                th[i] = max(th[i], t["height"])
    return bh, th, lc


def _load_grid_result(results_dir: Path, ext: str):
    """Read the (already-merged, one row per sensor) result file for the
    model's single sensor grid, whose filename is keyed by grid identifier
    rather than a fixed 'grid_0' -- looked up via grids_info.json."""
    info = json.loads((results_dir / "grids_info.json").read_text())
    ident = info[0]["identifier"]
    path = results_dir / f"{ident}.{ext}"
    if ext == "npy":
        return np.load(path)
    return np.loadtxt(path)


def _load_split_concat(dir_path: Path) -> np.ndarray:
    """initial_results/conditions/* are split into N per-worker .csv files
    (numpy-format despite the extension), each a contiguous block of
    sensors in original order (honeybee-radiance's SplitGridFolder always
    partitions sensors into sequential contiguous chunks). Concatenate them
    back into the original sensor order."""
    files = sorted(dir_path.glob("*.csv"), key=lambda p: int(p.stem))
    parts = [np.load(f) for f in files]
    return np.concatenate(parts, axis=0)


def run_scenario(sc: dict, hbjson_dir: Path, epw_dir: Path, run_dir: Path):
    sid = sc["scenario_id"]
    month = sc["assigned_month"]
    model_path = hbjson_dir / f"scene_{sid:04d}.hbjson"
    epw_path = epw_dir / f"scene_{sid:04d}.epw"

    # ── real Radiance + EnergyPlus: utci_comfort_map ──
    utci_folder = run_dir / f"scene_{sid:04d}_utci"
    recipe = Recipe("utci_comfort_map")
    recipe.input_value_by_name("model", str(model_path))
    recipe.input_value_by_name("epw", str(epw_path))
    recipe.input_value_by_name(
        "run-period", f"{month}/15 to {month}/15 between 8 and 18 @1")
    settings = RecipeSettings(folder=str(utci_folder))
    recipe.run(settings, radiance_check=True, openstudio_check=True,
               energyplus_check=True, queenbee_path=QB_PATH, silent=True)

    rf = utci_folder / "utci_comfort_map"
    cond = rf / "initial_results" / "conditions"
    utci_arr = _load_grid_result(rf / "results" / "temperature", "npy")     # (N, T)
    ta_arr = _load_split_concat(cond / "air_temperature")                   # (N, T)
    rh_arr = _load_split_concat(cond / "rel_humidity")                      # (N, T)
    lw_mrt = _load_split_concat(cond / "longwave_mrt")                      # (N, T)
    sw_mrt = _load_split_concat(cond / "shortwave_mrt")                     # (N, T)
    mrt_arr = lw_mrt + sw_mrt
    air_speed_files = sorted((cond / "air_speed").glob("*.json"), key=lambda p: int(p.stem))
    air_speed_json = json.loads(air_speed_files[0].read_text())
    outdoor_profile = np.array(air_speed_json["air_speeds"][0], dtype=np.float32)  # (T,)
    N = utci_arr.shape[0]
    va_arr = np.tile(outdoor_profile[None, :], (N, 1))                      # (N, T)
    in_shadow = (sw_mrt < IN_SHADOW_SW_THRESHOLD)

    # ── real Radiance: sky_view (SVF) ──
    svf_folder = run_dir / f"scene_{sid:04d}_svf"
    recipe2 = Recipe("sky_view")
    recipe2.input_value_by_name("model", str(model_path))
    settings2 = RecipeSettings(folder=str(svf_folder))
    recipe2.run(settings2, radiance_check=True, queenbee_path=QB_PATH, silent=True)
    svf_pct = _load_grid_result(svf_folder / "sky_view" / "results" / "sky_view", "res")
    svf_arr = (svf_pct / 100.0).astype(np.float32)

    return {
        "ta": ta_arr.T.astype(np.float32),          # -> (T, N)
        "mrt": mrt_arr.T.astype(np.float32),
        "va": va_arr.T.astype(np.float32),
        "rh": rh_arr.T.astype(np.float32),
        "utci": utci_arr.T.astype(np.float32),
        "svf": svf_arr,
        "in_shadow": in_shadow.T,
    }


def main(scenarios_pkl: str, hbjson_dir: str, epw_dir: str, run_dir: str, out_dir: str):
    with open(scenarios_pkl, "rb") as f:
        scenarios = pickle.load(f)
    hbjson_dir = Path(hbjson_dir)
    epw_dir = Path(epw_dir)
    run_dir = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    t_start = time.time()
    n_skipped = 0
    for i, sc in enumerate(scenarios):
        sid = sc["scenario_id"]
        if (out / f"sim_{sid:04d}.npz").exists():
            n_skipped += 1
            print(f"  [{i+1}/{len(scenarios)}] scene {sid:04d}: already simulated, skipping",
                  flush=True)
            continue
        scp = prune_scene(sc)
        t0 = time.time()
        try:
            res = run_scenario(sc, hbjson_dir, epw_dir, run_dir)
        except Exception as e:
            warnings.warn(f"scene {sid} failed: {e}")
            continue
        dt = time.time() - t0

        sensor_pts = None
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "lbt_runner", str(_SCRIPT_DIR / "03_lbt_batch_runner.py"))
        _lbt = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_lbt)
        sensor_pts = _lbt.build_sensor_grid(scp, 4.0)
        bh, th, lc = _static_fields(scp, sensor_pts)

        j = int(np.argmin(sensor_pts[:, 0] ** 2 + sensor_pts[:, 1] ** 2))
        T = res["utci"].shape[0]
        sensor_utci = np.full((T, sensor_pts.shape[0]), np.nan, dtype=np.float32)
        sensor_mask = np.zeros((T, sensor_pts.shape[0]), dtype=bool)
        sensor_utci[:, j] = res["utci"][:, j]
        sensor_mask[:, j] = True

        np.savez_compressed(
            out / f"sim_{sid:04d}.npz",
            sensor_pts=sensor_pts, ta=res["ta"], mrt=res["mrt"], va=res["va"],
            rh=res["rh"], utci=res["utci"], svf=res["svf"],
            in_shadow=res["in_shadow"].astype(np.uint8),
            building_height=bh, tree_height=th, land_cover=lc,
            sim_hours=np.array(SIM_HOURS), scenario_id=np.array(sid),
            far=np.array(sc["far_actual"]), bcr=np.array(sc["bcr_actual"]),
            n_buildings=np.array(len(scp["buildings"])),
            sensor_utci=sensor_utci, sensor_mask=sensor_mask.astype(np.uint8),
            assigned_month=np.array(sc["assigned_month"]),
        )
        manifest.append({"scenario_id": sid, "month": sc["assigned_month"],
                          "n_sensors": int(sensor_pts.shape[0]),
                          "runtime_s": round(dt, 1)})

        # disk is tight: the Radiance octrees + EnergyPlus run folders are
        # heavy and no longer needed once the npz has been extracted.
        import shutil
        for sub in (f"scene_{sid:04d}_utci", f"scene_{sid:04d}_svf"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(scenarios)}] scene {sid:04d}: {dt:.1f}s "
              f"(total {elapsed/60:.1f} min)", flush=True)

    (out / "manifest_v5.json").write_text(
        json.dumps({"n_scenes": len(manifest), "scenes": manifest}, indent=2),
        encoding="utf-8")
    print(f"\n[run_lbt_recipe_v5] {len(manifest)}/{len(scenarios)} scenes -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"))
    ap.add_argument("--hbjson", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "hbjson"))
    ap.add_argument("--epw", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "epw"))
    ap.add_argument("--run_dir", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "lbt_runs"))
    ap.add_argument("--out", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "sim"))
    args = ap.parse_args()
    main(args.scenarios, args.hbjson, args.epw, args.run_dir, args.out)
