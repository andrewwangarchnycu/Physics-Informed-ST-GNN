"""
01_data_generation/scripts/13_build_scenario_epw_v5.py
════════════════════════════════════════════════════════════════
V5 per-scenario custom EPW builder.

Starts from the real regional TMY EPW (TWN_NOR_Hsinchu.City.467570_TMYx,
downloaded from climate.onebuilding.org -- ERA5-reanalysis-based, the same
public station V4 used for its solar climatology) and, for each V5
scenario, overrides the dry-bulb temperature and relative humidity hours of
the scenario's assigned month with its own real MOENV IoT hourly diurnal
profile (site_iot_v4.pkl / manifest_v4.json -- exactly the validated real
signal V4 used, reusing the same plausibility checks as
09_run_real_sim_v4.py's build_climate()). Wind speed/direction and solar
(GHI/DNI/DHI) are left as the TMY EPW's real values -- this is a deliberate
provenance change from V4 (which sourced wind from a single CWB station):
here wind and solar both come from the same real regional climatology, and
this is documented explicitly in V5_HONEST_STATUS.md.

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
from pathlib import Path

import numpy as np
from ladybug.epw import EPW

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

TMY_EPW = _ROOT / "data" / "01_raw_epw" / "TWN_NOR_Hsinchu.City.467570_TMYx.2011-2025.epw"


def _ok(arr, lo, hi):
    """Same plausibility check as 09_run_real_sim_v4.py's build_climate()."""
    if arr is None:
        return False
    f = np.asarray(arr, dtype=float)
    f = f[np.isfinite(f)]
    return f.size >= 6 and lo <= float(np.mean(f)) <= hi and float(np.std(f)) >= 0.3


def build_scenario_epw(base_epw: EPW, scenario: dict, iot: dict) -> EPW:
    sid = scenario["scenario_id"]
    month = scenario["assigned_month"]

    temp_id = str(iot["site_temp_id"][sid]) if iot["site_temp_id"][sid] else None
    rh_id = str(iot["site_rh_id"][sid]) if iot["site_rh_id"][sid] else None
    ta24 = iot["temp"].get(temp_id, {}).get(month) if temp_id else None
    rh24 = iot["rh"].get(rh_id, {}).get(month) if rh_id else None

    use_ta = _ok(ta24, 15.0, 45.0)
    use_rh = _ok(rh24, 25.0, 97.0)

    epw = copy.deepcopy(base_epw)
    dbt = epw.dry_bulb_temperature
    rh = epw.relative_humidity
    dbt_vals = list(dbt.values)
    rh_vals = list(rh.values)

    n_ta_overridden = n_rh_overridden = 0
    for i, dt in enumerate(dbt.datetimes):
        if dt.month != month:
            continue
        hr = dt.hour
        if use_ta and np.isfinite(ta24[hr]) and 10.0 <= ta24[hr] <= 48.0:
            dbt_vals[i] = float(ta24[hr])
            n_ta_overridden += 1
        if use_rh and np.isfinite(rh24[hr]) and 10.0 <= rh24[hr] <= 100.0:
            rh_vals[i] = float(min(rh24[hr], 99.0))
            n_rh_overridden += 1

    dbt.values = dbt_vals
    rh.values = rh_vals
    return epw, {"ta_source": "iot" if use_ta else "tmy",
                 "rh_source": "iot" if use_rh else "tmy",
                 "n_ta_hours_overridden": n_ta_overridden,
                 "n_rh_hours_overridden": n_rh_overridden}


def main(scenarios_pkl: str, iot_pkl: str, out_dir: str):
    with open(scenarios_pkl, "rb") as f:
        scenarios = pickle.load(f)
    with open(iot_pkl, "rb") as f:
        iot = pickle.load(f)

    base_epw = EPW(str(TMY_EPW))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        epw, prov = build_scenario_epw(base_epw, sc, iot)
        epw_path = out / f"scene_{sid:04d}.epw"
        epw.save(str(epw_path))
        prov["scenario_id"] = sid
        prov["month"] = sc["assigned_month"]
        manifest.append(prov)
        print(f"  scene {sid:04d} (month {sc['assigned_month']}): "
              f"ta={prov['ta_source']} ({prov['n_ta_hours_overridden']}h)  "
              f"rh={prov['rh_source']} ({prov['n_rh_hours_overridden']}h)")

    n_iot_ta = sum(1 for m in manifest if m["ta_source"] == "iot")
    n_iot_rh = sum(1 for m in manifest if m["rh_source"] == "iot")
    print(f"\n[build_epw_v5] {len(manifest)} EPWs written -> {out}")
    print(f"  real-IoT air temp : {n_iot_ta}/{len(manifest)}")
    print(f"  real-IoT humidity : {n_iot_rh}/{len(manifest)}")

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
