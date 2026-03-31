"""
_shared.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] scripts [REMOVED_ZH:7]and[REMOVED_ZH:4]。
[REMOVED_ZH:3]Run，[REMOVED_ZH:1] 01–05 [REMOVED_ZH:1] script import。
════════════════════════════════════════════════════════════════
"""
import math
import csv
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class HourlyClimate:
    month:      int
    day:        int
    hour:       int
    ta:         float
    rh:         float
    wind_speed: float
    wind_dir:   float
    ghi:        float
    dni:        float
    dhi:        float
    dew_point:  float = 0.0


@dataclass
class EPWData:
    city:      str   = ""
    country:   str   = ""
    latitude:  float = 0.0
    longitude: float = 0.0
    timezone:  float = 0.0
    elevation: float = 0.0
    hours: List[HourlyClimate] = field(default_factory=list)

    def get_month(self, month: int) -> List[HourlyClimate]:
        return [h for h in self.hours if h.month == month]

    def get_typical_day(self, month: int, stat: str = "hottest") -> List[HourlyClimate]:
        month_data = self.get_month(month)
        if not month_data:
            raise ValueError(f"No data for month {month}")
        if stat == "hottest":
            days: Dict[int, List] = {}
            for h in month_data:
                days.setdefault(h.day, []).append(h)
            best = max(days, key=lambda d: float(np.mean([h.ta for h in days[d]])))
            return sorted(days[best], key=lambda h: h.hour)
        # mean
        by_hour: Dict[int, List] = {}
        for h in month_data:
            by_hour.setdefault(h.hour, []).append(h)
        result = []
        for hr in sorted(by_hour):
            hs = by_hour[hr]
            avg = lambda attr: float(np.mean([getattr(h, attr) for h in hs]))
            result.append(HourlyClimate(
                month=month, day=0, hour=hr,
                ta=avg("ta"), rh=avg("rh"),
                wind_speed=avg("wind_speed"), wind_dir=avg("wind_dir"),
                ghi=avg("ghi"), dni=avg("dni"), dhi=avg("dhi"),
                dew_point=avg("dew_point"),
            ))
        return result


def solar_position(lat: float, lon: float, tz: float,
                   month: int, day: int, hour: int) -> tuple:
    """[REMOVED_ZH:2] (altitude_deg, azimuth_deg)。"""
    doy = sum([31,28,31,30,31,30,31,31,30,31,30,31][:month-1]) + day
    B   = math.radians(360/365*(doy-81))
    dec = math.radians(23.45*math.sin(B))
    eot = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)
    lst = hour - 0.5 + (lon - tz*15)/15 + eot/60
    ha  = math.radians(15*(lst-12))
    lat_r = math.radians(lat)
    sin_alt = (math.sin(lat_r)*math.sin(dec) +
               math.cos(lat_r)*math.cos(dec)*math.cos(ha))
    alt_r = math.asin(max(-1.0, min(1.0, sin_alt)))
    alt   = math.degrees(alt_r)
    cos_az = ((math.sin(dec) - math.sin(lat_r)*math.sin(alt_r)) /
              (math.cos(lat_r)*max(math.cos(alt_r), 1e-9)))
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if lst > 12:
        az = 360 - az
    return float(alt), float(az)


def solar_altitude_deg(lat, lon, month, day, hour, tz) -> float:
    return solar_position(lat, lon, tz, month, day, hour)[0]