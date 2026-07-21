"""
01_data_generation/scripts/13_build_scenario_epw_v5.py
════════════════════════════════════════════════════════════════
V5 per-scenario custom EPW builder (CWB + IoT calibrated).

Starts from the real regional TMY EPW (TWN_NOR_Hsinchu.City.467570_TMYx,
downloaded from climate.onebuilding.org -- ERA5-reanalysis-based) and, for
each V5 scenario, overrides the assigned month's hours with real observed
data, following the exact same provenance priority as V4's
09_run_real_sim_v4.py build_climate():

  * Air temperature / relative humidity: real MOENV IoT hourly diurnal
    profile for that site (site_iot_v4.pkl) is used when it passes the same
    plausibility check V4 used (>=6 finite hours, physically plausible mean,
    genuine diurnal variation -- catches sensors stuck at 0/100% or a
    constant value); otherwise falls back to the real CWB Hsinchu station
    (C0D660) monthly-hourly climatology; otherwise falls back to the TMY
    EPW's own real value for that hour.
  * Wind speed / direction: real CWB Hsinchu station monthly-hourly
    climatology (cwb_data_6/7/8.csv) when available for that hour;
    otherwise falls back to the TMY EPW's own real wind for that hour.
  * Solar (GHI/DNI/DHI): left as the TMY EPW's real values -- CWB does not
    report irradiance, matching V4's documented limitation.

This restores V4's original CWB-based wind calibration (V5's earlier
39/150-scenario runs used TMY-only wind because the raw CWB CSVs were not
yet available on this machine; the CWB files have since been supplied and
are used here).

Only the scenario's assigned month is touched; the `utci_comfort_map`
recipe is later run with run-period restricted to that month, so the other
11 months of the EPW are never read.

Output: one .epw per scenario in real_simulations_v5/epw/.
"""
from __future__ import annotations

import sys
import copy
import pickle
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from ladybug.epw import EPW

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

TMY_EPW = _ROOT / "data" / "01_raw_epw" / "TWN_NOR_Hsinchu.City.467570_TMYx.2011-2025.epw"

# reuse V4's real CWB monthly-hourly climatology loader verbatim
_spec = importlib.util.spec_from_file_location(
    "v4_runner", str(_SCRIPT_DIR / "09_run_real_sim_v4.py"))
_v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v4)
load_cwb_monthly = _v4.load_cwb_monthly


def _ok(arr, lo, hi):
    """Same plausibility check as 09_run_real_sim_v4.py's build_climate()."""
    if arr is None:
        return False
    f = np.asarray(arr, dtype=float)
    f = f[np.isfinite(f)]
    return f.size >= 6 and lo <= float(np.mean(f)) <= hi and float(np.std(f)) >= 0.3


def build_scenario_epw(base_epw: EPW, scenario: dict, iot: dict, cwb: dict) -> EPW:
    sid = scenario["scenario_id"]
    month = scenario["assigned_month"]

    temp_id = str(iot["site_temp_id"][sid]) if iot["site_temp_id"][sid] else None
    rh_id = str(iot["site_rh_id"][sid]) if iot["site_rh_id"][sid] else None
    ta24 = iot["temp"].get(temp_id, {}).get(month) if temp_id else None
    rh24 = iot["rh"].get(rh_id, {}).get(month) if rh_id else None

    use_ta = _ok(ta24, 15.0, 45.0)
    use_rh = _ok(rh24, 25.0, 97.0)
    cwbm = cwb.get(month, {})

    epw = copy.deepcopy(base_epw)
    dbt = epw.dry_bulb_temperature
    rh_coll = epw.relative_humidity
    wspd = epw.wind_speed
    wdir = epw.wind_direction
    dbt_vals  = list(dbt.values)
    rh_vals   = list(rh_coll.values)
    wspd_vals = list(wspd.values)
    wdir_vals = list(wdir.values)

    n_ta = {"iot": 0, "cwb": 0, "tmy": 0}
    n_rh = {"iot": 0, "cwb": 0, "tmy": 0}
    n_ws = {"cwb": 0, "tmy": 0}
    for i, dt in enumerate(dbt.datetimes):
        if dt.month != month:
            continue
        hr = dt.hour
        cwb_hr = cwbm.get(hr, {})

        # air temperature: IoT (validated) -> CWB -> TMY (already in dbt_vals)
        if use_ta and np.isfinite(ta24[hr]) and 10.0 <= ta24[hr] <= 48.0:
            dbt_vals[i] = float(ta24[hr]); n_ta["iot"] += 1
        elif np.isfinite(cwb_hr.get("ta", np.nan)):
            dbt_vals[i] = float(cwb_hr["ta"]); n_ta["cwb"] += 1
        else:
            n_ta["tmy"] += 1

        # relative humidity: IoT (validated) -> CWB -> TMY
        if use_rh and np.isfinite(rh24[hr]) and 10.0 <= rh24[hr] <= 100.0:
            rh_vals[i] = float(min(rh24[hr], 99.0)); n_rh["iot"] += 1
        elif np.isfinite(cwb_hr.get("rh", np.nan)):
            rh_vals[i] = float(min(cwb_hr["rh"], 99.0)); n_rh["cwb"] += 1
        else:
            n_rh["tmy"] += 1

        # wind speed / direction: CWB -> TMY
        ws = cwb_hr.get("ws", np.nan)
        wd = cwb_hr.get("wd", np.nan)
        if np.isfinite(ws) and np.isfinite(wd):
            wspd_vals[i] = float(max(ws, 0.5))
            wdir_vals[i] = float(wd % 360.0)
            n_ws["cwb"] += 1
        else:
            n_ws["tmy"] += 1

    dbt.values = dbt_vals
    rh_coll.values = rh_vals
    wspd.values = wspd_vals
    wdir.values = wdir_vals

    def _dom(counts):
        return max(counts, key=counts.get)

    return epw, {
        "ta_source": _dom(n_ta), "rh_source": _dom(n_rh), "ws_source": _dom(n_ws),
        "n_ta_hours": n_ta, "n_rh_hours": n_rh, "n_ws_hours": n_ws,
    }


def main(scenarios_pkl: str, iot_pkl: str, out_dir: str):
    with open(scenarios_pkl, "rb") as f:
        scenarios = pickle.load(f)
    with open(iot_pkl, "rb") as f:
        iot = pickle.load(f)
    cwb = load_cwb_monthly()

    base_epw = EPW(str(TMY_EPW))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        epw, prov = build_scenario_epw(base_epw, sc, iot, cwb)
        epw_path = out / f"scene_{sid:04d}.epw"
        epw.save(str(epw_path))
        prov["scenario_id"] = sid
        prov["month"] = sc["assigned_month"]
        manifest.append(prov)
        print(f"  scene {sid:04d} (month {sc['assigned_month']}): "
              f"ta={prov['ta_source']}  rh={prov['rh_source']}  ws={prov['ws_source']}")

    n_iot_ta = sum(1 for m in manifest if m["ta_source"] == "iot")
    n_cwb_ta = sum(1 for m in manifest if m["ta_source"] == "cwb")
    n_iot_rh = sum(1 for m in manifest if m["rh_source"] == "iot")
    n_cwb_rh = sum(1 for m in manifest if m["rh_source"] == "cwb")
    n_cwb_ws = sum(1 for m in manifest if m["ws_source"] == "cwb")
    print(f"\n[build_epw_v5] {len(manifest)} EPWs written -> {out}")
    print(f"  air temp : real-IoT {n_iot_ta}/{len(manifest)}  "
          f"real-CWB fallback {n_cwb_ta}/{len(manifest)}")
    print(f"  humidity : real-IoT {n_iot_rh}/{len(manifest)}  "
          f"real-CWB fallback {n_cwb_rh}/{len(manifest)}")
    print(f"  wind     : real-CWB {n_cwb_ws}/{len(manifest)}")

    import json
    (out / "epw_manifest_v5.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "scenarios_v5_subset.pkl"))
    ap.add_argument("--iot", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "site_iot_v4.pkl"))
    ap.add_argument("--out", default=str(
        _ROOT / "01_data_generation" / "outputs" / "real_simulations_v5" / "epw"))
    args = ap.parse_args()
    main(args.scenarios, args.iot, args.out)
