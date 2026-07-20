"""
01_data_generation/scripts/09_run_real_sim_v4.py
════════════════════════════════════════════════════════════════
V4 real-scene physics simulation.

Runs the (validated) outdoor-thermal-comfort physics core (SVF ray-casting,
building/tree shadowing, empirical MRT energy balance, canyon wind shelter,
pythermalcomfort UTCI) on the 300 REAL-geometry scenarios built by
07_build_real_scenarios_v4.py, forced with REAL measured climate:

  * Air temperature : each scene's OWN real MOENV IoT station hourly diurnal
                      curve for its assigned month (site_iot_v4.pkl).
  * Humidity        : nearest real MOENV humidity station hourly curve.
  * Wind            : real CWB Hsinchu station (C0D660) monthly hourly means.
  * Solar (GHI/DNI) : clear-sky model (Haurwitz global + beam split) --
                      the CWB station does not report irradiance, so this is
                      a modelled input, documented honestly, NOT a real obs.

For every scene the sensor nearest the station location (site centre) has its
UTCI recorded as ``sensor_utci`` with ``sensor_mask=True`` -- a hybrid signal
derived from the REAL measured air temperature + humidity and the modelled
MRT/wind, used for optional direct IoT supervision in training.

Output: per-scene npz in real_simulations_v4/ (same schema as V3 runner)
plus a manifest of real-data provenance and honest fallback counts.
"""
from __future__ import annotations

import sys
import csv
import json
import math
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from shapely.geometry import Point
from shared import EPWData, HourlyClimate, solar_position

# reuse the validated physics core
sys.path.insert(0, str(_SCRIPT_DIR))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lbt_runner", str(_SCRIPT_DIR / "03_lbt_batch_runner.py"))
_lbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbt)
simulate_one = _lbt.simulate_one
save_npz = _lbt.save_npz

CWB_DIR = _ROOT / "01_data_generation" / "inputs" / "cwb_data"

# Physics tractability for dense REAL geometry: the SVF ray-caster only
# reaches ~35 m and shadow influence is dominated by nearby obstacles, so we
# keep the nearest buildings/trees to the site centre (bounding per-scene cost
# while preserving the real immediate surroundings) and use a 4 m sensor grid.
MAX_BUILDINGS = 30
MAX_TREES     = 20
GRID_SPACING  = 4.0


def prune_scene(sc: dict, kb: int = MAX_BUILDINGS, kt: int = MAX_TREES) -> dict:
    """Keep the kb nearest buildings and kt nearest trees to the site centre."""
    s = dict(sc)
    if len(sc["buildings"]) > kb:
        s["buildings"] = sorted(
            sc["buildings"],
            key=lambda b: b["footprint"].centroid.x**2 + b["footprint"].centroid.y**2
        )[:kb]
    if len(sc["trees"]) > kt:
        s["trees"] = sorted(
            sc["trees"], key=lambda t: t["pos"][0]**2 + t["pos"][1]**2)[:kt]
    return s


# ── clear-sky solar (Haurwitz global + simple beam split) ───────
def clear_sky(sol_alt_deg: float):
    """Return (ghi, dni, dhi) W/m^2 for a clear sky at given solar altitude."""
    if sol_alt_deg <= 0:
        return 0.0, 0.0, 0.0
    cz = math.sin(math.radians(sol_alt_deg))          # cos(zenith)
    ghi = 1098.0 * cz * math.exp(-0.059 / max(cz, 1e-3))   # Haurwitz
    dhi = 0.15 * ghi                                   # diffuse fraction (clear)
    dni = max(0.0, (ghi - dhi) / max(cz, 1e-3))
    return float(ghi), float(dni), float(dhi)


# ── regional TMY EPW typical-day solar per month ────────────────
def load_epw_solar(months=(6, 7, 8)):
    """{month: {hour: (ghi, dni, dhi)}} from the regional EPW typical day.

    Uses the 'mean' diurnal profile (not the single hottest day) so the solar
    forcing represents typical summer conditions rather than a clear-sky peak.
    """
    epw_pkl = _ROOT / "01_data_generation" / "outputs" / "raw_simulations" / "epw_data.pkl"
    if not epw_pkl.exists():
        return {}
    import __main__
    from shared import HourlyClimate as _HC, EPWData as _ED
    __main__.HourlyClimate = _HC; __main__.EPWData = _ED
    try:
        epw = pickle.load(open(epw_pkl, "rb"))
    except Exception:
        return {}
    out = {}
    for m in months:
        try:
            day = epw.get_typical_day(m, stat="mean")
        except Exception:
            continue
        out[m] = {h.hour: (float(h.ghi), float(h.dni), float(h.dhi)) for h in day}
    return out


# ── CWB Hsinchu station monthly hourly climatology ──────────────
def load_cwb_monthly(months=(6, 7, 8)):
    """Parse cwb_data_{m}.csv -> {month: {hour: dict(ta,rh,ws,wd)}} (means)."""
    out = {}
    for m in months:
        fp = CWB_DIR / f"cwb_data_{m}.csv"
        if not fp.exists():
            continue
        acc = {h: {"ta": [], "rh": [], "ws": [], "wd": []} for h in range(24)}
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.lstrip("﻿").strip()
                if not s or s.startswith("*") or s.startswith("#"):
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) < 7 or not parts[1].isdigit() or len(parts[1]) != 10:
                    continue
                hr = int(parts[1][8:10]) % 24
                def _f(x):
                    try:
                        v = float(x)
                        return v if v > -90 else None
                    except ValueError:
                        return None
                ta, rh, ws, wd = _f(parts[3]), _f(parts[4]), _f(parts[5]), _f(parts[6])
                if ta is not None: acc[hr]["ta"].append(ta)
                if rh is not None: acc[hr]["rh"].append(rh)
                if ws is not None: acc[hr]["ws"].append(ws)
                if wd is not None: acc[hr]["wd"].append(wd)
        out[m] = {}
        for h in range(24):
            a = acc[h]
            out[m][h] = {
                "ta": float(np.mean(a["ta"])) if a["ta"] else np.nan,
                "rh": float(np.mean(a["rh"])) if a["rh"] else np.nan,
                "ws": float(np.mean(a["ws"])) if a["ws"] else 1.5,
                "wd": float(np.mean(a["wd"])) if a["wd"] else 180.0,
            }
    return out


def build_climate(scenario, iot, cwb, sim_hours, solar_by_month=None, tz=8.0):
    """Assemble {hour: HourlyClimate} for one scene from real data.

    Solar (ghi/dni/dhi) is taken from the regional TMY EPW typical-day profile
    for the scene's month when available (measured solar climatology,
    consistent with the validated V3 physics), falling back to a clear-sky
    model only if the EPW profile is missing -- the CWB station reports no
    irradiance. Returns (clim_map, provenance).
    """
    sid = scenario["scenario_id"]
    month = scenario["assigned_month"]
    lat, lon = scenario["site_lat"], scenario["site_lon"]

    temp_id = str(iot["site_temp_id"][sid]) if iot["site_temp_id"][sid] else None
    rh_id = str(iot["site_rh_id"][sid]) if iot["site_rh_id"][sid] else None
    ta24 = iot["temp"].get(temp_id, {}).get(month) if temp_id else None
    rh24 = iot["rh"].get(rh_id, {}).get(month) if rh_id else None

    # Validate the ASSIGNED month's profile (some IoT temp/humidity sensors
    # are faulty -- stuck at 0, 100, or constant). A profile is used only if
    # it has enough finite hours, a physically plausible mean, and genuine
    # diurnal variation; otherwise the variable falls back to the real CWB
    # regional climatology. This catches month-specific faults that a
    # cross-month sanity check in extraction would miss.
    def _ok(arr, lo, hi):
        if arr is None:
            return False
        f = np.asarray(arr)[np.isfinite(arr)]
        return (f.size >= 6 and lo <= float(np.mean(f)) <= hi
                and float(np.std(f)) >= 0.3)

    use_ta = _ok(ta24, 15.0, 45.0)
    use_rh = _ok(rh24, 25.0, 97.0)
    cwbm = cwb.get(month, {})
    prov = {"ta": "iot" if use_ta else "cwb", "rh": "iot" if use_rh else "cwb"}

    smonth = (solar_by_month or {}).get(month, {})
    clim_map = {}
    for hr in sim_hours:
        sol_alt, _ = solar_position(lat, lon, tz, month, 15, hr)
        if hr in smonth:
            ghi, dni, dhi = smonth[hr]          # real TMY EPW solar
        else:
            ghi, dni, dhi = clear_sky(sol_alt)  # fallback
        # air temperature: real IoT (validated) & plausible this hour, else CWB
        if use_ta and np.isfinite(ta24[hr]) and 10.0 <= ta24[hr] <= 48.0:
            ta = float(ta24[hr])
        else:
            ta = cwbm.get(hr, {}).get("ta", np.nan)
        if not np.isfinite(ta):
            ta = 30.0
        if use_rh and np.isfinite(rh24[hr]) and 10.0 <= rh24[hr] <= 100.0:
            rh = float(min(rh24[hr], 99.0))     # cap saturated to 99 %RH
        else:
            rh = cwbm.get(hr, {}).get("rh", np.nan)
        if not np.isfinite(rh):
            rh = 75.0
        ws = cwbm.get(hr, {}).get("ws", 1.5)
        wd = cwbm.get(hr, {}).get("wd", 180.0)
        clim_map[hr] = HourlyClimate(month=month, day=15, hour=hr, ta=ta, rh=rh,
                                     wind_speed=ws, wind_dir=wd,
                                     ghi=ghi, dni=dni, dhi=dhi)
    return clim_map, prov


def main(scen_pkl, iot_pkl, out_dir, h_start, h_end):
    scenarios = pickle.load(open(scen_pkl, "rb"))
    iot = pickle.load(open(iot_pkl, "rb"))
    cwb = load_cwb_monthly()
    solar_by_month = load_epw_solar()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sim_hours = list(range(h_start, h_end + 1))
    calib = {"roughness_length": 0.5, "albedo_road": 0.15, "ta_bias_offset": 0.0}

    print(f"[sim] {len(scenarios)} real scenes | hours {h_start}-{h_end} | "
          f"CWB months {sorted(cwb.keys())} | EPW solar months {sorted(solar_by_month.keys())}")
    prov_counts = {"ta_iot": 0, "ta_cwb": 0, "rh_iot": 0, "rh_cwb": 0}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    # Cap workers modestly: each worker is a full Python process importing
    # shapely + holding scenario geometry; too many exhaust Windows virtual
    # memory ("paging file too small"). 8 balances speed and RAM.
    workers = int(os.environ.get("V4_WORKERS", "8"))
    workers = max(1, min(workers, (os.cpu_count() or 4) - 2))
    tasks = [(sc, *build_climate(sc, iot, cwb, sim_hours, solar_by_month), sim_hours, calib, str(out))
             for sc in scenarios]
    for _, _, prov, *_ in tasks:
        prov_counts[f"ta_{prov['ta']}"] += 1
        prov_counts[f"rh_{prov['rh']}"] += 1

    print(f"[sim] running {len(tasks)} scenes on {workers} workers "
          f"(prune <= {MAX_BUILDINGS} bldg / {MAX_TREES} trees, grid {GRID_SPACING} m)")
    manifest = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_scene, t): t[0]["scenario_id"] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                manifest.append(r)
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(tasks)} scenes simulated", flush=True)

    with open(out / "manifest_v4.json", "w", encoding="utf-8") as f:
        json.dump({"n_scenes": done, "provenance_counts": prov_counts,
                   "scenes": manifest}, f, indent=2)
    print(f"\n[sim] HONEST PROVENANCE")
    print(f"  scenes simulated        : {done}/{len(scenarios)}")
    print(f"  air-temp from real IoT  : {prov_counts['ta_iot']}  (CWB fallback {prov_counts['ta_cwb']})")
    print(f"  humidity from real IoT  : {prov_counts['rh_iot']}  (CWB fallback {prov_counts['rh_cwb']})")
    print(f"  saved npz + manifest -> {out}")


def _run_scene(task):
    """Worker: prune geometry, simulate, attach real-IoT sensor supervision,
    save npz. Returns a manifest row (or None on failure)."""
    sc, clim_map, prov, sim_hours, calib, out_dir = task
    sid = sc["scenario_id"]
    scp = prune_scene(sc)
    epw = EPWData(city="Hsinchu-real", latitude=sc["site_lat"],
                  longitude=sc["site_lon"], timezone=8.0)
    try:
        res = simulate_one(scp, clim_map, epw, sim_hours, calib, spacing=GRID_SPACING)
    except Exception as e:
        warnings.warn(f"scene {sid} failed: {e}")
        return None
    if res is None:
        return None

    pts = res["sensor_pts"]
    utci = res["utci"]                      # (T, N)
    j = int(np.argmin(pts[:, 0] ** 2 + pts[:, 1] ** 2))   # sensor at station
    T = utci.shape[0]
    sensor_utci = np.full((T, pts.shape[0]), np.nan, dtype=np.float32)
    sensor_mask = np.zeros((T, pts.shape[0]), dtype=bool)
    if prov["ta"] == "iot":                 # only a REAL signal is supervisable
        sensor_utci[:, j] = utci[:, j]
        sensor_mask[:, j] = True
    res["sensor_utci"] = sensor_utci
    res["sensor_mask"] = sensor_mask
    res["assigned_month"] = sc["assigned_month"]
    res["site_lat"] = sc["site_lat"]; res["site_lon"] = sc["site_lon"]
    _save_npz_v4(res, Path(out_dir))
    return {"scenario_id": sid, "month": sc["assigned_month"],
            "ta_source": prov["ta"], "rh_source": prov["rh"],
            "n_sensors": int(pts.shape[0]), "station_sensor_idx": j}


def _save_npz_v4(result, out_dir):
    sid = result["scenario_id"]
    np.savez_compressed(
        out_dir / f"sim_{sid:04d}.npz",
        sensor_pts=result["sensor_pts"], ta=result["ta"], mrt=result["mrt"],
        va=result["va"], rh=result["rh"], utci=result["utci"], svf=result["svf"],
        in_shadow=result["in_shadow"].astype(np.uint8),
        building_height=result["building_height"], tree_height=result["tree_height"],
        land_cover=result["land_cover"], sim_hours=np.array(result["sim_hours"]),
        scenario_id=np.array(result["scenario_id"]), far=np.array(result["far"]),
        bcr=np.array(result["bcr"]), n_buildings=np.array(result["n_buildings"]),
        sensor_utci=result["sensor_utci"], sensor_mask=result["sensor_mask"].astype(np.uint8),
        assigned_month=np.array(result["assigned_month"]),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(_ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "scenarios_v4.pkl"))
    ap.add_argument("--iot", default=str(_ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "site_iot_v4.pkl"))
    ap.add_argument("--out", default=str(_ROOT / "01_data_generation" / "outputs" / "real_simulations_v4"))
    ap.add_argument("--h_start", type=int, default=8)
    ap.add_argument("--h_end", type=int, default=18)
    args = ap.parse_args()
    main(args.scenarios, args.iot, args.out, args.h_start, args.h_end)
