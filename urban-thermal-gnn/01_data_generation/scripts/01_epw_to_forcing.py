"""
01_epw_to_forcing.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/
[REMOVED_ZH:2] : [REMOVED_ZH:2] EPW [REMOVED_ZH:3]，[REMOVED_ZH:8]，[REMOVED_ZH:2]：
       (1) forcing_MMDD.json  ── ENVI-met / LBT [REMOVED_ZH:4]
       (2) forcing_MMDD.csv   ── [REMOVED_ZH:5]
       (3) EPWData pickle     ── [REMOVED_ZH:3] scripts [REMOVED_ZH:2]

[REMOVED_ZH:2] :
  pip install pyyaml numpy
  (shapely / pythermalcomfort [REMOVED_ZH:3] script [REMOVED_ZH:2])

Run :
  python 01_epw_to_forcing.py
  python 01_epw_to_forcing.py --config ../config/epw_parser_config.yaml
"""

import csv
import json
import math
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import numpy as np
import yaml  # pip install pyyaml


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:4]  (and lbt_batch_sim [REMOVED_ZH:4] dataclass，[REMOVED_ZH:4])
# ════════════════════════════════════════════════════════════════
@dataclass
class HourlyClimate:
    """[REMOVED_ZH:10]，[REMOVED_ZH:2] EPW [REMOVED_ZH:2]。"""
    month:      int
    day:        int
    hour:       int       # 1–24
    ta:         float     # [REMOVED_ZH:4] [°C]
    rh:         float     # Relative Humidity [%]
    wind_speed: float     # 10 m Wind Speed [m/s]
    wind_dir:   float     # [REMOVED_ZH:2] [deg], 0=N, 90=E
    ghi:        float     # [REMOVED_ZH:7] [W/m²]
    dni:        float     # [REMOVED_ZH:6] [W/m²]
    dhi:        float     # [REMOVED_ZH:6] [W/m²]
    dew_point:  float = 0.0


@dataclass
class EPWData:
    """[REMOVED_ZH:2] EPW [REMOVED_ZH:4]（8760 [REMOVED_ZH:2]）。"""
    city:      str   = ""
    country:   str   = ""
    latitude:  float = 0.0
    longitude: float = 0.0
    timezone:  float = 0.0
    elevation: float = 0.0
    hours: List[HourlyClimate] = field(default_factory=list)

    # ── [REMOVED_ZH:4] ──────────────────────────────
    def get_month(self, month: int) -> List[HourlyClimate]:
        return [h for h in self.hours if h.month == month]

    def get_typical_day(self, month: int,
                         stat: str = "hottest") -> List[HourlyClimate]:
        """
        [REMOVED_ZH:6] (24 [REMOVED_ZH:2])。
        stat = 'hottest' : [REMOVED_ZH:8]
        stat = 'mean'    : [REMOVED_ZH:6]
        """
        month_data = self.get_month(month)
        if not month_data:
            raise ValueError(f"EPW [REMOVED_ZH:5] {month} [REMOVED_ZH:3]")

        if stat == "hottest":
            days: Dict[int, List[HourlyClimate]] = {}
            for h in month_data:
                days.setdefault(h.day, []).append(h)
            best = max(days, key=lambda d: float(np.mean([h.ta for h in days[d]])))
            return sorted(days[best], key=lambda h: h.hour)

        # mean
        by_hour: Dict[int, List[HourlyClimate]] = {}
        for h in month_data:
            by_hour.setdefault(h.hour, []).append(h)
        result = []
        for hr in sorted(by_hour):
            hs = by_hour[hr]
            result.append(HourlyClimate(
                month=month, day=0, hour=hr,
                ta         = float(np.mean([h.ta         for h in hs])),
                rh         = float(np.mean([h.rh         for h in hs])),
                wind_speed = float(np.mean([h.wind_speed for h in hs])),
                wind_dir   = float(np.mean([h.wind_dir   for h in hs])),
                ghi        = float(np.mean([h.ghi        for h in hs])),
                dni        = float(np.mean([h.dni        for h in hs])),
                dhi        = float(np.mean([h.dhi        for h in hs])),
                dew_point  = float(np.mean([h.dew_point  for h in hs])),
            ))
        return result


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:4]  (Spencer 1971 + Iqbal 1983)
# ════════════════════════════════════════════════════════════════
def solar_position(lat: float, lon: float, tz: float,
                   month: int, day: int, hour: int
                   ) -> tuple[float, float]:
    """
    [REMOVED_ZH:2] (altitude_deg, azimuth_deg)。
    altitude < 0 [REMOVED_ZH:4]; azimuth [REMOVED_ZH:3] 0°, [REMOVED_ZH:3]。
    """
    doy = sum([31,28,31,30,31,30,31,31,30,31,30,31][:month-1]) + day
    B   = math.radians(360 / 365 * (doy - 81))

    # [REMOVED_ZH:2]
    dec = math.radians(23.45 * math.sin(B))

    # [REMOVED_ZH:5] [min]
    eot  = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)
    lst  = hour - 0.5 + (lon - tz * 15) / 15 + eot / 60   # [REMOVED_ZH:3]
    ha   = math.radians(15 * (lst - 12))                    # [REMOVED_ZH:2]

    lat_r = math.radians(lat)
    sin_alt = (math.sin(lat_r)*math.sin(dec) +
               math.cos(lat_r)*math.cos(dec)*math.cos(ha))
    alt_r = math.asin(max(-1.0, min(1.0, sin_alt)))
    alt   = math.degrees(alt_r)

    # [REMOVED_ZH:3]
    cos_az = ((math.sin(dec) - math.sin(lat_r)*math.sin(alt_r)) /
              (math.cos(lat_r)*math.cos(alt_r) + 1e-9))
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if lst > 12:
        az = 360 - az      # [REMOVED_ZH:7]

    return float(alt), float(az)


# [REMOVED_ZH:10]
def solar_altitude_deg(lat, lon, month, day, hour, tz) -> float:
    return solar_position(lat, lon, tz, month, day, hour)[0]


# ════════════════════════════════════════════════════════════════
# 3. EPW [REMOVED_ZH:3]
# ════════════════════════════════════════════════════════════════
class EPWParser:
    """[REMOVED_ZH:2] .epw [REMOVED_ZH:3] EPWData (8760 [REMOVED_ZH:1])。"""

    # EnergyPlus 9.x [REMOVED_ZH:6] (0-based)
    _COL = {
        "year": 0, "month": 1, "day": 2, "hour": 3,
        "ta":  6, "dew": 7, "rh": 8,
        "ghi": 13, "dni": 14, "dhi": 15,
        "wind_dir": 20, "wind_spd": 21,
    }

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        col_override = self.cfg.get("columns", {})
        self._COL = {**self._COL, **col_override}
        c = self.cfg.get("constraints", {})
        self._v_min  = float(c.get("wind_speed_min", 0.5))
        self._ghi_min = float(c.get("ghi_min", 0.0))

    def parse(self, epw_path: str) -> EPWData:
        p = Path(epw_path)
        if not p.exists():
            raise FileNotFoundError(f"EPW [REMOVED_ZH:3]: {epw_path}")

        epw = EPWData()
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader):
                if line_no == 0:
                    # LOCATION,City,State,Country,Type,WMO,Lat,Lon,TZ,Elev
                    epw.city      = row[1].strip() if len(row) > 1 else ""
                    epw.country   = row[3].strip() if len(row) > 3 else ""
                    epw.latitude  = float(row[6])  if len(row) > 6  else 0.0
                    epw.longitude = float(row[7])  if len(row) > 7  else 0.0
                    epw.timezone  = float(row[8])  if len(row) > 8  else 0.0
                    epw.elevation = float(row[9])  if len(row) > 9  else 0.0
                    continue
                if line_no < 8:         # [REMOVED_ZH:1] 8 [REMOVED_ZH:4]
                    continue
                if len(row) < 22:
                    continue

                C = self._COL
                try:
                    epw.hours.append(HourlyClimate(
                        month      = int(row[C["month"]]),
                        day        = int(row[C["day"]]),
                        hour       = int(row[C["hour"]]),
                        ta         = float(row[C["ta"]]),
                        dew_point  = float(row[C["dew"]]),
                        rh         = float(row[C["rh"]]),
                        wind_speed = max(self._v_min, float(row[C["wind_spd"]])),
                        wind_dir   = float(row[C["wind_dir"]]),
                        ghi        = max(self._ghi_min, float(row[C["ghi"]])),
                        dni        = max(self._ghi_min, float(row[C["dni"]])),
                        dhi        = max(self._ghi_min, float(row[C["dhi"]])),
                    ))
                except (ValueError, IndexError):
                    continue

        n = len(epw.hours)
        if n < 8760:
            print(f"  [EPWParser] ⚠  [REMOVED_ZH:3] {n} [REMOVED_ZH:2] ([REMOVED_ZH:2] 8760)")
        print(f"  [EPWParser] ✓  {epw.city}, {epw.country}  "
              f"lat={epw.latitude:.2f}  lon={epw.longitude:.2f}  "
              f"tz=UTC+{epw.timezone:.0f}  {n} hrs")
        return epw


# ════════════════════════════════════════════════════════════════
# 4. Demo EPW [REMOVED_ZH:3]
# ════════════════════════════════════════════════════════════════
def create_demo_epw(path: str,
                    city: str = "Taipei",
                    lat: float = 25.07,
                    lon: float = 121.55,
                    tz: float = 8.0):
    """
    [REMOVED_ZH:13] EPW。
    [REMOVED_ZH:7] EnergyPlus [REMOVED_ZH:3] Climate.OneBuilding [REMOVED_ZH:3] EPW。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    month_ta_base = [16,17,19,22,26,28,29,29,28,28,28,26]

    lines = [
        f"LOCATION,{city},,Taiwan,ASHRAE Intl.,466990,"
        f"{lat},{lon},{tz:+.1f},9.0",
    ]
    for i in range(1, 8):
        lines.append(f"HEADER_{i},placeholder")

    for m in range(1, 13):
        days_in_m = [31,28,31,30,31,30,31,31,30,31,30,31][m-1]
        ta_b = month_ta_base[m-1]
        for d in range(1, days_in_m+1):
            for h in range(1, 25):
                ta  = ta_b + 3.5 * math.sin(math.pi * (h-6)/12)
                rh  = 75 - 12 * math.sin(math.pi * (h-6)/12)
                ghi = max(0.0, 850 * math.sin(math.pi*(h-6)/12)) if 6<h<18 else 0.0
                dni = ghi * 0.72
                dhi = ghi * 0.28
                row = [
                    "1988", str(m), str(d), str(h), "60",
                    "?",
                    f"{ta:.1f}",      # col 6
                    f"{ta-3:.1f}",    # col 7 dew
                    f"{max(30,min(99,rh)):.0f}",  # col 8
                    "101325",         # col 9
                    "?","?","?",
                    f"{ghi:.0f}",     # col 13
                    f"{dni:.0f}",     # col 14
                    f"{dhi:.0f}",     # col 15
                    "?","?","?","?",
                    "180",            # col 20 wind_dir
                    "3.2",            # col 21 wind_spd
                    "?","?","?","?","?","?","?","?","?","?","?","?","?",
                ]
                lines.append(",".join(row))

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [Demo] ✓ Build[REMOVED_ZH:2] EPW: {p}")


# ════════════════════════════════════════════════════════════════
# 5. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def export_forcing_json(typical_day: List[HourlyClimate],
                         epw: EPWData,
                         out_path: Path,
                         month: int) -> None:
    """[REMOVED_ZH:2] ENVI-met [REMOVED_ZH:4] JSON。"""
    payload = {
        "metadata": {
            "city":      epw.city,
            "country":   epw.country,
            "latitude":  epw.latitude,
            "longitude": epw.longitude,
            "timezone":  epw.timezone,
            "month":     month,
        },
        "hourly": []
    }
    for h in typical_day:
        alt, az = solar_position(
            epw.latitude, epw.longitude, epw.timezone,
            h.month, h.day if h.day else 15, h.hour
        )
        entry = asdict(h)
        entry["solar_altitude_deg"] = round(alt, 2)
        entry["solar_azimuth_deg"]  = round(az,  2)
        payload["hourly"].append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_forcing_csv(typical_day: List[HourlyClimate],
                        epw: EPWData,
                        out_path: Path) -> None:
    """[REMOVED_ZH:8] CSV（[REMOVED_ZH:5]）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["hour","ta","rh","wind_speed","wind_dir",
                  "ghi","dni","dhi","solar_alt","solar_az"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in typical_day:
            alt, az = solar_position(
                epw.latitude, epw.longitude, epw.timezone,
                h.month, h.day if h.day else 15, h.hour
            )
            writer.writerow({
                "hour":       h.hour,
                "ta":         round(h.ta, 2),
                "rh":         round(h.rh, 1),
                "wind_speed": round(h.wind_speed, 2),
                "wind_dir":   round(h.wind_dir, 1),
                "ghi":        round(h.ghi, 1),
                "dni":        round(h.dni, 1),
                "dhi":        round(h.dhi, 1),
                "solar_alt":  round(alt, 2),
                "solar_az":   round(az,  2),
            })


# ════════════════════════════════════════════════════════════════
# 6. Main Program
# ════════════════════════════════════════════════════════════════
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "../config/epw_parser_config.yaml"):
    print("\n[01_epw_to_forcing] ── EPW [REMOVED_ZH:2]and[REMOVED_ZH:6] ──")
    cfg = load_config(config_path)

    epw_path   = Path(cfg["epw"]["path"])
    out_dir    = Path(cfg["output"]["forcing_dir"])
    save_json  = cfg["output"].get("save_json", True)
    save_csv   = cfg["output"].get("save_csv",  True)
    months     = cfg["epw"].get("target_months", [7])
    stat       = cfg["epw"].get("typical_day_stat", "hottest")

    # ── [REMOVED_ZH:2]/Build EPW ──────────────────────────
    if not epw_path.exists():
        if cfg["epw"].get("auto_demo", False):
            create_demo_epw(
                str(epw_path),
                city=cfg["epw"].get("demo_city", "Taipei"),
            )
        else:
            raise FileNotFoundError(f"EPW [REMOVED_ZH:3]: {epw_path}")

    parser   = EPWParser(cfg)
    epw_data = parser.parse(str(epw_path))

    # ── [REMOVED_ZH:11] ─────────────────
    for month in months:
        print(f"\n  [REMOVED_ZH:2] {month} [REMOVED_ZH:4] (stat={stat})...")
        typical = epw_data.get_typical_day(month=month, stat=stat)

        tag = f"M{month:02d}"
        if save_json:
            jp = out_dir / f"forcing_{tag}.json"
            export_forcing_json(typical, epw_data, jp, month)
            print(f"    ✓ JSON: {jp}")

        if save_csv:
            cp = out_dir / f"forcing_{tag}.csv"
            export_forcing_csv(typical, epw_data, cp)
            print(f"    ✓ CSV:  {cp}")

    # ── Sequence[REMOVED_ZH:1] EPWData [REMOVED_ZH:3] scripts [REMOVED_ZH:2] ────
    pkl_path = out_dir / "epw_data.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(epw_data, f)
    print(f"\n  ✓ EPWData pickle: {pkl_path}")
    print("[01_epw_to_forcing] [REMOVED_ZH:2]。\n")

    return epw_data


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../00_config/epw_parser_config.yaml",
                    help="epw_parser_config.yaml [REMOVED_ZH:2]")
    args = ap.parse_args()
    main(args.config)