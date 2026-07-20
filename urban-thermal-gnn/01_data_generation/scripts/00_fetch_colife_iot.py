"""
00_fetch_colife_iot.py
═══════════════════════════════════════════════════════════════════════════
Stage 0: Fetch real EPA IoT sensor observations for GIS spatial calibration.

Data source
───────────
OGC SensorThings API (STA) — real-time endpoint maintained by Colife:
  https://sta.colife.org.tw/STA_AirQuality_EPAIoT/v1.0/

NOTE: The STA endpoint is a real-time sliding window (≈ 2–3 hours of
1-minute readings per station).  It does NOT serve historical archives.
The archived CSV portal (history.colife.org.tw) requires credentials.
This script therefore collects whatever summer observations are currently
available and saves them in the IotSensorLoader-compatible folder layout.
Run during peak summer hours (10:00–18:00 Taiwan time) for best coverage.

Pipeline
────────
Step 1  Fetch station metadata — all Things within radius_km of site
         → station_metadata.json  (id, name, lat, lon, dist_km)
         Datastream ID formula (verified empirically):
           PM2.5 DS id      = thing_id × 3 − 2
           Relative humidity DS id = thing_id × 3 − 1
           Temperature DS id       = thing_id × 3

Step 2  Download current observations for all nearby stations via STA
         → moenviot_temperature/moenviot_temperature_YYYYMMDD.csv
         → moenviot_humidity/moenviot_humidity_YYYYMMDD.csv
         (date labelled as the actual UTC date of each observation)

Usage
─────
  python 00_fetch_colife_iot.py
  python 00_fetch_colife_iot.py --radius 15 --lat 24.80 --lon 120.97
  python 00_fetch_colife_iot.py --metadata_only
  python 00_fetch_colife_iot.py --download_only
  python 00_fetch_colife_iot.py --max_stations 50   # limit for testing
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False
    print("[00] WARNING: 'requests' not installed. Run: pip install requests")

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ═════════════════════════════════════════════════════════════════════════════
# 0. Constants
# ═════════════════════════════════════════════════════════════════════════════
STA_BASE   = "https://sta.colife.org.tw/STA_AirQuality_EPAIoT/v1.0"
PAGE_SIZE  = 1000
MAX_OBS    = 500    # max observations to fetch per datastream (STA rolling window ~157)
REQ_TIMEOUT= 30
RETRY_WAIT = 2
MAX_RETRIES= 3

SITE_LAT_DEFAULT = 24.80
SITE_LON_DEFAULT = 120.97

# Verified DS-ID formula (Thing(1)→DS3=Temp, Thing(4524)→DS13572=Temp):
#   Temperature DS id = thing_id × 3
#   Humidity    DS id = thing_id × 3 - 1
DS_OFFSET = {"temperature": 0, "humidity": -1}   # offset from thing_id*3


# ═════════════════════════════════════════════════════════════════════════════
# 1. Utilities
# ═════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _get(url: str, params: Optional[dict] = None,
         session: Optional["requests.Session"] = None) -> Optional["requests.Response"]:
    requester = session or requests
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requester.get(url, params=params, timeout=REQ_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 404):
                return None
        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                print(f"    [warn] {url[:70]}: {exc}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Step 1 — Station metadata
# ═════════════════════════════════════════════════════════════════════════════

def _parse_coords(thing: dict) -> Optional[Tuple[float, float]]:
    for loc in thing.get("Locations", []):
        coords = loc.get("location", {}).get("coordinates", [])
        if len(coords) >= 2:
            try:
                return float(coords[1]), float(coords[0])   # GeoJSON [lon, lat]
            except (TypeError, ValueError):
                pass
    return None


def fetch_station_metadata(site_lat: float,
                           site_lon: float,
                           radius_km: float = 15.0,
                           out_path: Optional[Path] = None) -> List[Dict]:
    if not _REQUESTS:
        return []

    print(f"\n[Step 1] Fetching station metadata from STA API …")
    print(f"  Site ({site_lat}, {site_lon}), radius = {radius_km} km")

    url = (f"{STA_BASE}/Things"
           f"?$expand=Locations"
           f"&$select=@iot.id,name,properties"
           f"&$top={PAGE_SIZE}")

    stations, total_seen = [], 0
    session = requests.Session()
    session.headers["Accept"] = "application/json"

    while url:
        resp = _get(url, session=session)
        if resp is None:
            print("  [WARN] STA API unreachable — stopping pagination")
            break
        data = resp.json()
        items = data.get("value", [])
        total_seen += len(items)

        for thing in items:
            coords = _parse_coords(thing)
            if coords is None:
                continue
            lat, lon = coords
            dist = haversine_km(site_lat, site_lon, lat, lon)
            if dist <= radius_km:
                stations.append({
                    "id":         thing.get("@iot.id"),
                    "name":       thing.get("name", ""),
                    "lat":        lat,
                    "lon":        lon,
                    "dist_km":    round(dist, 4),
                    "properties": thing.get("properties", {}),
                })

        url = data.get("@iot.nextLink")
        if url:
            time.sleep(0.1)

    stations.sort(key=lambda s: s["dist_km"])
    print(f"  Found {len(stations)} stations within {radius_km} km "
          f"(scanned {total_seen} total Things)")
    for s in stations[:5]:
        print(f"    [{s['id']:>5}] {s['name'][:35]:35s} "
              f"({s['lat']:.4f},{s['lon']:.4f})  {s['dist_km']:.2f} km")
    if len(stations) > 5:
        print(f"    ... and {len(stations)-5} more")

    if out_path and stations:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stations, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metadata → {out_path}")

    return stations


# ═════════════════════════════════════════════════════════════════════════════
# 3. Step 2 — Download STA observations
# ═════════════════════════════════════════════════════════════════════════════

def _temp_ds_id(thing_id: int) -> int:
    return thing_id * 3

def _humi_ds_id(thing_id: int) -> int:
    return thing_id * 3 - 1


def _fetch_observations(ds_id: int, max_obs: int,
                         session: "requests.Session") -> List[Dict]:
    """Return list of {phenomenonTime, result} dicts for a datastream."""
    resp = _get(
        f"{STA_BASE}/Datastreams({ds_id})/Observations",
        params={
            "$top":     max_obs,
            "$orderby": "phenomenonTime asc",
            "$select":  "phenomenonTime,result",
        },
        session=session,
    )
    if resp is None:
        return []
    return resp.json().get("value", [])


def _obs_to_daily_rows(obs: List[Dict],
                        thing_id: int,
                        lat: float,
                        lon: float,
                        value_col: str,
                        tz_offset_h: int = 8   # Taiwan UTC+8
                        ) -> Dict[str, List[Dict]]:
    """
    Convert STA observation list to {date_str: [row, ...]} keyed by
    Taiwan-local date (YYYYMMDD).
    """
    from datetime import timedelta
    daily: Dict[str, List[Dict]] = defaultdict(list)
    offset = timedelta(hours=tz_offset_h)
    for o in obs:
        pt = o.get("phenomenonTime", "")
        result = o.get("result")
        if result is None or not pt:
            continue
        try:
            # Parse ISO8601 UTC → local Taiwan time
            dt_utc = datetime.fromisoformat(pt.replace("Z", "+00:00"))
            dt_local = dt_utc + offset
            date_str = dt_local.strftime("%Y%m%d")
        except (ValueError, TypeError):
            continue
        daily[date_str].append({
            "deviceId":  str(thing_id),   # numeric, matches station_metadata.json id field
            "time":      dt_local.strftime("%Y-%m-%d %H:%M:%S"),
            "lat":       lat,
            "lon":       lon,
            value_col:   result,
        })
    return daily


def download_sta_observations(stations:     List[Dict],
                               out_dir:      Path,
                               max_obs:      int   = MAX_OBS,
                               max_stations: int   = 0,
                               skip_existing:bool  = False
                               ) -> Dict[str, int]:
    """
    For every station in the list, fetch temperature and humidity observations
    from the STA API (rolling window) and save daily CSV files.

    DS-ID formula: Temperature = thing_id × 3,  Humidity = thing_id × 3 − 1
    """
    if not (_REQUESTS and _PANDAS):
        print("[00] pandas or requests not available — skipping download")
        return {"fetched": 0, "empty": 0, "errors": 0}

    temp_dir = out_dir / "moenviot_temperature"
    humi_dir = out_dir / "moenviot_humidity"
    temp_dir.mkdir(parents=True, exist_ok=True)
    humi_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["Accept"] = "application/json"

    work_list = stations[:max_stations] if max_stations > 0 else stations
    print(f"\n[Step 2] Downloading STA observations for {len(work_list)} stations …")
    print(f"  (STA rolling window ≈ {max_obs} most-recent 1-min readings per station)")

    # Accumulate rows: {date_str: {type: [rows]}}
    daily_temp: Dict[str, List[Dict]] = defaultdict(list)
    daily_humi: Dict[str, List[Dict]] = defaultdict(list)
    stats = {"fetched": 0, "empty": 0, "errors": 0}

    for idx, sta in enumerate(work_list, 1):
        tid  = sta["id"]
        lat  = sta["lat"]
        lon  = sta["lon"]
        name = sta["name"]

        if tid is None:
            stats["errors"] += 1
            continue

        # Temperature
        t_obs = _fetch_observations(_temp_ds_id(tid), max_obs, session)
        if t_obs:
            for date_str, rows in _obs_to_daily_rows(
                    t_obs, tid, lat, lon, "temperature").items():
                daily_temp[date_str].extend(rows)
        else:
            stats["empty"] += 1

        # Humidity
        h_obs = _fetch_observations(_humi_ds_id(tid), max_obs, session)
        if h_obs:
            for date_str, rows in _obs_to_daily_rows(
                    h_obs, tid, lat, lon, "humidity").items():
                daily_humi[date_str].extend(rows)

        if idx % 10 == 0 or idx == len(work_list):
            print(f"  [{idx:>3}/{len(work_list)}] last: {name[:30]:30s} "
                  f"T={len(t_obs)} H={len(h_obs)} obs")

        time.sleep(0.1)   # polite rate limit

    # Write per-date CSV files
    print(f"\n  Writing CSV files …")
    n_written = 0
    for date_str, rows in sorted(daily_temp.items()):
        fname  = f"moenviot_temperature_{date_str}.csv"
        out_fp = temp_dir / fname
        if skip_existing and out_fp.exists():
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_fp, index=False, encoding="utf-8-sig")
        print(f"  ✓ {fname}: {len(df)} rows, {df['deviceId'].nunique()} stations")
        n_written += 1

    for date_str, rows in sorted(daily_humi.items()):
        fname  = f"moenviot_humidity_{date_str}.csv"
        out_fp = humi_dir / fname
        if skip_existing and out_fp.exists():
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_fp, index=False, encoding="utf-8-sig")
        n_written += 1

    stats["fetched"] = n_written
    print(f"\n  CSV files written: {n_written} "
          f"({len(daily_temp)} temperature days, {len(daily_humi)} humidity days)")
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# 4. Summary
# ═════════════════════════════════════════════════════════════════════════════

def summarise_downloaded(out_dir: Path) -> Dict:
    if not _PANDAS:
        return {}

    temp_dir = out_dir / "moenviot_temperature"
    humi_dir = out_dir / "moenviot_humidity"

    def _scan(folder: Path) -> Tuple[int, set]:
        if not folder.is_dir():
            return 0, set()
        files, stations = 0, set()
        for f in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
                if "deviceId" in df.columns:
                    stations.update(df["deviceId"].dropna().astype(str).unique())
                files += 1
            except Exception:
                pass
        return files, stations

    tf, ts = _scan(temp_dir)
    hf, hs = _scan(humi_dir)
    all_sta = ts | hs

    summary = {
        "temperature_files":  tf,
        "humidity_files":     hf,
        "unique_stations":    len(all_sta),
        "station_ids_sample": sorted(all_sta)[:10],
    }
    print(f"\n[Summary] IoT data at {out_dir}")
    print(f"  Temperature CSVs : {tf}")
    print(f"  Humidity CSVs    : {hf}")
    print(f"  Unique stations  : {len(all_sta)}")
    if all_sta:
        print(f"  Stations (sample): {sorted(all_sta)[:5]}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════════════

def main(site_lat:      float      = SITE_LAT_DEFAULT,
         site_lon:      float      = SITE_LON_DEFAULT,
         radius_km:     float      = 15.0,
         out_dir:       str        = "",
         metadata_only: bool       = False,
         download_only: bool       = False,
         max_stations:  int        = 0,
         skip_existing: bool       = False) -> Dict:

    script_dir = Path(__file__).resolve().parent
    root_dir   = script_dir.parent
    iot_dir    = Path(out_dir) if out_dir else root_dir / "outputs" / "iot_data"
    iot_dir.mkdir(parents=True, exist_ok=True)

    meta_path = iot_dir / "station_metadata.json"
    result    = {}

    # ── Step 1: Metadata ──────────────────────────────────────────
    if not download_only:
        stations = fetch_station_metadata(
            site_lat=site_lat, site_lon=site_lon,
            radius_km=radius_km, out_path=meta_path)
        result["n_stations"] = len(stations)
    else:
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                stations = json.load(f)
            print(f"[Step 1] Loaded {len(stations)} stations from {meta_path}")
        else:
            stations = []
            print(f"[Step 1] No metadata at {meta_path} — no stations to download")
        result["n_stations"] = len(stations)

    if metadata_only or not stations:
        return result

    # ── Step 2: Observations from STA ────────────────────────────
    dl_stats = download_sta_observations(
        stations      = stations,
        out_dir       = iot_dir,
        max_stations  = max_stations,
        skip_existing = skip_existing,
    )
    result.update(dl_stats)

    # ── Step 3: Summary ────────────────────────────────────────────
    summary = summarise_downloaded(iot_dir)
    result.update(summary)

    # Save run manifest
    now_utc  = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_time_utc":   now_utc,
        "site_lat":       site_lat,
        "site_lon":       site_lon,
        "radius_km":      radius_km,
        "n_stations":     result.get("n_stations", 0),
        "csvs_written":   dl_stats.get("fetched", 0),
        "unique_stations_in_csvs": summary.get("unique_stations", 0),
        "data_note": (
            "STA real-time rolling window (~2-3h). "
            "Run during Taiwan summer peak hours (10:00-18:00 CST) for "
            "best summer diurnal coverage. "
            "For historical archives (2022-2025), portal credentials are required."
        ),
    }
    with open(iot_dir / "fetch_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Fetch summary → {iot_dir / 'fetch_summary.json'}")
    print(f"\n[00_fetch_colife_iot] Complete.")
    print(f"  Next: python 04_sensing_calibration.py "
          f"--iot_dir {iot_dir} --months 6,7,8,9")
    return result


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fetch EPA IoT sensor data (STA real-time API) for Hsinchu area."
    )
    ap.add_argument("--lat",        type=float, default=SITE_LAT_DEFAULT)
    ap.add_argument("--lon",        type=float, default=SITE_LON_DEFAULT)
    ap.add_argument("--radius",     type=float, default=15.0)
    ap.add_argument("--out",        default="",
                    help="Output directory (default: outputs/iot_data/)")
    ap.add_argument("--metadata_only",  action="store_true")
    ap.add_argument("--download_only",  action="store_true",
                    help="Skip metadata fetch, use saved station_metadata.json")
    ap.add_argument("--max_stations",   type=int, default=0,
                    help="Limit number of stations (0=all, useful for testing)")
    ap.add_argument("--skip_existing",  action="store_true",
                    help="Skip dates whose CSV files already exist")
    args = ap.parse_args()

    main(
        site_lat      = args.lat,
        site_lon      = args.lon,
        radius_km     = args.radius,
        out_dir       = args.out,
        metadata_only = args.metadata_only,
        download_only = args.download_only,
        max_stations  = args.max_stations,
        skip_existing = args.skip_existing,
    )
