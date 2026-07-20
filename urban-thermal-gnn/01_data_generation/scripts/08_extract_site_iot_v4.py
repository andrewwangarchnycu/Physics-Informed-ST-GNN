"""
01_data_generation/scripts/08_extract_site_iot_v4.py
════════════════════════════════════════════════════════════════
Extract each V4 site's OWN real MOENV IoT hourly air-temperature and
relative-humidity diurnal profile for June / July / August 2025.

Because every selected site is centred exactly on a real MOENV station
(nearest-station distance = 0 m by construction), that station's genuine
measured sub-hourly temperature is the most faithful air-temperature
forcing available for the scene -- far better than a single regional
weather station. This script aggregates the raw sub-hourly readings into
a per-device, per-month, per-hour mean diurnal curve.

Humidity: MOENV humidity devices are a separate set, so each site is
matched to its NEAREST real humidity station (reported distance kept for
honesty); its hourly RH diurnal curve is extracted the same way.

Output: site_iot_v4.pkl
    {
      "temp":  {deviceId: {month: np.array([24]) hourly mean °C, ...}},
      "rh":    {deviceId: {month: np.array([24]) hourly mean %,  ...}},
      "site_temp_id": [deviceId per site, len 300],
      "site_rh_id":   [nearest humidity deviceId per site],
      "site_rh_dist_m":[distance to that humidity station],
      "counts": {...}   # obs counts for transparency
    }
"""
from __future__ import annotations

import sys
import json
import math
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]

TEMP_DIR = _ROOT / "01_data_generation" / "inputs" / "iot_data" / "moenviot_temperature"
HUM_DIR  = _ROOT / "01_data_generation" / "inputs" / "iot_data" / "moenviot_humidity"
ROSTER   = _ROOT / "01_data_generation" / "outputs" / "iot_data" / "all_devices_region.pkl"

M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat):
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def load_sites(sites_json):
    return json.loads(Path(sites_json).read_text(encoding="utf-8"))["sites"]


def site_device_ids(sites, roster):
    """Match each site (which is a device location) to its deviceId."""
    rev = {(round(la, 5), round(lo, 5)): did for did, (la, lo) in roster.items()}
    ids = []
    for s in sites:
        ids.append(rev.get((round(s["lat"], 5), round(s["lon"], 5))))
    return ids


def hourly_profiles(csv_dir, value_col, id_set, months=(6, 7, 8), valrange=None):
    """Scan daily CSVs; return {deviceId: {month: hourly-mean[24]}} and counts.

    ``valrange=(lo, hi)`` drops readings outside [lo, hi] before aggregating --
    used to filter out faulty humidity sensors that report physically
    impossible values (0 or >100 %RH).
    """
    files = sorted(csv_dir.glob("*.csv"))
    id_set = {str(i) for i in id_set if i is not None}
    parts = []
    for k, f in enumerate(files):
        try:
            df = pd.read_csv(f, usecols=["deviceId", value_col, "time"])
        except Exception:
            continue
        df["deviceId"] = df["deviceId"].astype(str)
        df = df[df["deviceId"].isin(id_set)]
        if df.empty:
            continue
        t = pd.to_datetime(df["time"], errors="coerce")
        v = pd.to_numeric(df[value_col], errors="coerce")
        sub = pd.DataFrame({"deviceId": df["deviceId"].values,
                            "_mon": t.dt.month.values, "_hr": t.dt.hour.values,
                            "_v": v.values})
        sub = sub[sub["_mon"].isin(months)].dropna()
        if valrange is not None:
            sub = sub[(sub["_v"] >= valrange[0]) & (sub["_v"] <= valrange[1])]
        if not sub.empty:
            parts.append(sub)
        if (k + 1) % 20 == 0:
            print(f"    [{value_col}] {k+1}/{len(files)} files scanned", flush=True)
    out, counts = {}, {}
    if not parts:
        return out, counts
    allrows = pd.concat(parts, ignore_index=True)
    allrows["_hr"] = allrows["_hr"].astype(int)
    allrows["_mon"] = allrows["_mon"].astype(int)
    grp = allrows.groupby(["deviceId", "_mon", "_hr"])["_v"].agg(["mean", "count"])
    for (did, mon, hr), row in grp.iterrows():
        d = out.setdefault(did, {})
        arr = d.setdefault(mon, np.full(24, np.nan, dtype=np.float32))
        arr[hr] = row["mean"]
        c = counts.setdefault(did, {})
        c[mon] = c.get(mon, 0) + int(row["count"])
    return out, counts


def nearest_hum_station(sites, hum_roster):
    """For each site, nearest humidity deviceId + distance (m)."""
    hids = list(hum_roster.keys())
    hll = np.array([hum_roster[h] for h in hids])  # (H,2) lat,lon
    ids, dists = [], []
    for s in sites:
        la, lo = s["lat"], s["lon"]
        dlat = (hll[:, 0] - la) * M_PER_DEG_LAT
        dlon = (hll[:, 1] - lo) * _m_per_deg_lon(la)
        d = np.hypot(dlat, dlon)
        j = int(np.argmin(d))
        ids.append(hids[j]); dists.append(float(d[j]))
    return ids, dists


def build_hum_roster(hum_dir, n_sample=4):
    """Roster of humidity-station locations (sample a few days), NaN-filtered."""
    files = sorted(hum_dir.glob("*.csv"))
    step = max(1, len(files) // n_sample)
    roster = {}
    for f in files[::step]:
        try:
            df = pd.read_csv(f, usecols=["deviceId", "lat", "lon"])
        except Exception:
            continue
        for did, la, lo in zip(df["deviceId"].astype(str), df["lat"], df["lon"]):
            try:
                la, lo = float(la), float(lo)
            except (ValueError, TypeError):
                continue
            if np.isfinite(la) and np.isfinite(lo):
                roster[did] = (la, lo)
    return roster


def _sane_rh(prof_by_month) -> bool:
    """A humidity diurnal profile is usable if it has enough finite hours, a
    physically plausible mean, and is not a stuck/constant sensor."""
    if not prof_by_month:
        return False
    for arr in prof_by_month.values():
        finite = arr[np.isfinite(arr)]
        if finite.size >= 6 and 25.0 <= float(np.mean(finite)) <= 97.0 \
                and float(np.std(finite)) >= 0.5:
            return True
    return False


def main(sites_json, out_pkl):
    sites = load_sites(sites_json)
    roster = pickle.load(open(ROSTER, "rb"))
    temp_ids = site_device_ids(sites, roster)
    n_matched = sum(1 for x in temp_ids if x is not None)
    print(f"[iot] {n_matched}/{len(sites)} sites matched to a temperature deviceId")

    # ── temperature: reuse existing extraction if present (slow to rescan) ──
    temp_prof = temp_cnt = None
    if Path(out_pkl).exists():
        try:
            prev = pickle.load(open(out_pkl, "rb"))
            if prev.get("site_temp_id") == temp_ids and prev.get("temp"):
                temp_prof = prev["temp"]; temp_cnt = prev["counts"]["temp"]
                print("[iot] reusing cached TEMPERATURE profiles from existing pkl")
        except Exception:
            pass
    if temp_prof is None:
        print("[iot] extracting real hourly TEMPERATURE profiles ...")
        temp_prof, temp_cnt = hourly_profiles(TEMP_DIR, "temperature",
                                              set(temp_ids), months=(6, 7, 8))

    # ── humidity: candidate pool = co-located temp stations (292/300 also
    #    report humidity) + nearest humidity station per site. Extract all,
    #    then per-site pick the nearest SANE station; faulty (0/100/stuck)
    #    sensors are rejected and left for a CWB fallback in the sim step. ──
    print("[iot] building humidity-station roster + nearest match ...")
    hum_roster = build_hum_roster(HUM_DIR)
    near_ids, near_dist = nearest_hum_station(sites, hum_roster)
    cand = set(str(x) for x in temp_ids if x) | set(str(x) for x in near_ids)
    print(f"[iot] {len(hum_roster)} humidity stations; {len(cand)} candidate ids")

    print("[iot] extracting real hourly HUMIDITY profiles (0/>100 filtered) ...")
    rh_prof, rh_cnt = hourly_profiles(HUM_DIR, "humidity", cand,
                                      months=(6, 7, 8), valrange=(5.0, 100.0))

    # per-site selection: prefer co-located station, else nearest, if sane
    site_rh_id, site_rh_dist, n_iot_rh = [], [], 0
    for i, s in enumerate(sites):
        chosen, dist = None, float("nan")
        tid = str(temp_ids[i]) if temp_ids[i] else None
        if tid and tid in rh_prof and _sane_rh(rh_prof[tid]):
            chosen, dist = tid, 0.0
        elif str(near_ids[i]) in rh_prof and _sane_rh(rh_prof[str(near_ids[i])]):
            chosen, dist = str(near_ids[i]), near_dist[i]
        site_rh_id.append(chosen)
        site_rh_dist.append(dist)
        if chosen is not None:
            n_iot_rh += 1

    out = {
        "temp": temp_prof, "rh": rh_prof,
        "site_temp_id": temp_ids,
        "site_rh_id": site_rh_id, "site_rh_dist_m": site_rh_dist,
        "counts": {"temp": temp_cnt, "rh": rh_cnt},
    }
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)

    def _cov_temp(prof, ids):
        return sum(1 for did in ids if did and str(did) in prof
                   and any(np.isfinite(prof[str(did)][m]).sum() >= 6 for m in prof[str(did)]))
    print(f"\n[iot] HONEST COVERAGE")
    print(f"  sites with real temp diurnal curve: {_cov_temp(temp_prof, temp_ids)}/{len(sites)}")
    print(f"  sites with SANE real rh   diurnal curve: {n_iot_rh}/{len(sites)} "
          f"(remainder -> CWB regional humidity fallback in sim)")
    print(f"  saved: {out_pkl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=str(_ROOT / "01_data_generation" / "outputs" / "real_sites_v4" / "selected_real_sites.json"))
    ap.add_argument("--out", default=str(_ROOT / "01_data_generation" / "outputs" / "real_simulations_v4" / "site_iot_v4.pkl"))
    args = ap.parse_args()
    main(args.sites, args.out)
