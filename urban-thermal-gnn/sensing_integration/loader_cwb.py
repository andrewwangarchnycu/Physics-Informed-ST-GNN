"""
sensing_integration/loader_cwb.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/sensing_integration/
[REMOVED_ZH:2] : [REMOVED_ZH:12] CSV（[REMOVED_ZH:3]），
       [REMOVED_ZH:11]，[REMOVED_ZH:8]。

[REMOVED_ZH:4]: [REMOVED_ZH:13]
  https://e-service.cwb.gov.tw/HistoryDataQuery/
  [REMOVED_ZH:2]: [REMOVED_ZH:2](hPa) [REMOVED_ZH:2](℃) [REMOVED_ZH:4](%) Wind Speed(m/s) [REMOVED_ZH:2](360deg)
        [REMOVED_ZH:5](m/s) [REMOVED_ZH:7] [REMOVED_ZH:3](mm)
  [REMOVED_ZH:2]: [REMOVED_ZH:3]

[REMOVED_ZH:7]: ta, rh, ws, wd（[REMOVED_ZH:9]，[REMOVED_ZH:5] UTCI）
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from noise_removal import clean_cwb_dataframe


# ════════════════════════════════════════════════════════════════
# 1. CWB [REMOVED_ZH:6]
# ════════════════════════════════════════════════════════════════
CWB_COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "ObsTime", "[REMOVED_ZH:4]", "time"],
    "station_id": ["StationId", "station_id", "[REMOVED_ZH:4]"],
    "station_name": ["StationName", "[REMOVED_ZH:4]"],
    "lat": ["lat", "Latitude", "[REMOVED_ZH:2]"],
    "lon": ["lon", "Longitude", "[REMOVED_ZH:2]"],
    "pres": ["[REMOVED_ZH:2](hPa)", "Pressure", "pres", "SeaPressure"],
    "ta":   ["[REMOVED_ZH:2](℃)", "Temperature", "ta", "AirTemperature"],
    "rh":   ["[REMOVED_ZH:4](%)", "RHumidity", "rh", "RelativeHumidity"],
    "ws":   ["Wind Speed(m/s)", "WindSpeed", "ws", "WS"],
    "wd":   ["[REMOVED_ZH:2](360degree)", "WindDirection", "wd", "WD"],
    "ws_max": ["[REMOVED_ZH:5](m/s)", "GustSpeed", "ws_max"],
    "rain": ["[REMOVED_ZH:3](mm)", "Precipitation", "rain"],
}

# [REMOVED_ZH:5]：[REMOVED_ZH:5] UTCI compute[REMOVED_ZH:4]，[REMOVED_ZH:7]
RAIN_THRESHOLD_MM = 0.1


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for alias in CWB_COLUMN_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    return None


# ════════════════════════════════════════════════════════════════
# 2. CWB [REMOVED_ZH:3]
# ════════════════════════════════════════════════════════════════
class CWBStationLoader:
    """
    [REMOVED_ZH:14]and[REMOVED_ZH:2]。

    Parameters
    ----------
    csv_path : str | Path
        CSV [REMOVED_ZH:3]
    site_lat : float
        [REMOVED_ZH:6]
    site_lon : float
        [REMOVED_ZH:6]
    encoding : str
        CSV [REMOVED_ZH:2]
    """

    def __init__(self,
                 csv_path:  str,
                 site_lat:  float = 25.07,
                 site_lon:  float = 121.55,
                 encoding:  str   = "utf-8-sig"):
        self.csv_path = Path(csv_path)
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.encoding = encoding

    def _make_demo(self, month: int = 7) -> pd.DataFrame:
        """[REMOVED_ZH:12]。"""
        np.random.seed(0)
        n   = 24 * 31   # [REMOVED_ZH:3] 24h
        idx = pd.date_range(f"2023-{month:02d}-01 00:00",
                             periods=n, freq="H")
        ta_cycle = 28 + 4 * np.sin(np.pi * (idx.hour - 6) / 12)
        rh_cycle = 75 - 10 * np.sin(np.pi * (idx.hour - 6) / 12)
        ws_cycle = 2.5 + 1.5 * np.sin(np.pi * (idx.hour - 8) / 12)
        demo = pd.DataFrame({
            "timestamp":  idx,
            "station_id": "466920",
            "station_name": "[REMOVED_ZH:2]",
            "lat": self.site_lat,
            "lon": self.site_lon,
            "ta":   ta_cycle + np.random.randn(n) * 0.5,
            "rh":   rh_cycle + np.random.randn(n) * 2.0,
            "ws":   np.abs(ws_cycle + np.random.randn(n) * 0.5),
            "wd":   np.random.uniform(0, 360, n),
            "rain": np.maximum(0, np.random.randn(n) * 0.05),
        })
        print(f"  [CWB Demo] [REMOVED_ZH:6]（[REMOVED_ZH:3]，[REMOVED_ZH:2] {month}）")
        return demo

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        std_names = ["timestamp", "station_id", "station_name",
                     "lat", "lon", "pres", "ta", "rh",
                     "ws", "wd", "ws_max", "rain"]
        for std in std_names:
            found = _find_col(df, std)
            if found and found != std:
                rename_map[found] = std
        return df.rename(columns=rename_map)

    def _handle_special_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """CWB [REMOVED_ZH:1] -9991 / -9996 / -9997 / -9998 / -9999 [REMOVED_ZH:6]。"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace(
                [-9991, -9996, -9997, -9998, -9999], np.nan
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def load_and_clean(self,
                        month:     int   = 7,
                        verbose:   bool  = True,
                        radius_km: float = 30.0) -> pd.DataFrame:
        """
        [REMOVED_ZH:2]、[REMOVED_ZH:3]、[REMOVED_ZH:6]。

        Returns
        -------
        pd.DataFrame
            [REMOVED_ZH:4]，index=DatetimeIndex
            [REMOVED_ZH:2]: ta, rh, ws, wd, rain_flag, data_quality
        """
        if not self.csv_path.exists():
            warnings.warn(f"[CWB] CSV [REMOVED_ZH:3]: {self.csv_path}")
            raw = self._make_demo(month)
        else:
            try:
                raw = pd.read_csv(self.csv_path, encoding=self.encoding,
                                   low_memory=False)
            except UnicodeDecodeError:
                raw = pd.read_csv(self.csv_path, encoding="big5",
                                   low_memory=False)

        raw = self._normalise_columns(raw)
        raw = self._handle_special_values(raw)

        # [REMOVED_ZH:4]
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"])

        # [REMOVED_ZH:4]
        raw = raw[raw["timestamp"].dt.month == month].copy()
        if raw.empty:
            warnings.warn(f"[CWB] [REMOVED_ZH:2] {month} [REMOVED_ZH:3]")
            return pd.DataFrame()

        # Spatial Filtering（[REMOVED_ZH:5]）
        if "lat" in raw.columns and "lon" in raw.columns:
            from loader_iot import haversine_km
            raw["dist_km"] = haversine_km(
                self.site_lat, self.site_lon,
                raw["lat"].values, raw["lon"].values
            )
            raw = raw[raw["dist_km"] <= radius_km]

        # [REMOVED_ZH:2] DatetimeIndex
        raw = raw.set_index("timestamp").sort_index()

        # [REMOVED_ZH:9]
        htc_cols = [c for c in ["ta", "rh", "ws", "wd", "rain"]
                    if c in raw.columns]
        df = raw[htc_cols].copy()

        # [REMOVED_ZH:3]
        cleaned = clean_cwb_dataframe(df, verbose=verbose)

        # [REMOVED_ZH:4]：[REMOVED_ZH:3] > 0.1 mm [REMOVED_ZH:6] rain_flag=1
        if "rain" in cleaned.columns:
            cleaned["rain_flag"] = (
                cleaned["rain"].fillna(0) > RAIN_THRESHOLD_MM
            ).astype(int)
            rain_hours = cleaned["rain_flag"].sum()
            if verbose:
                print(f"  [CWB] [REMOVED_ZH:4] {rain_hours} [REMOVED_ZH:2]（[REMOVED_ZH:2] rain_flag，"
                      f"[REMOVED_ZH:3] UTCI [REMOVED_ZH:2]）")

        if verbose:
            n_good = (cleaned["data_quality"] == "good").sum()
            print(f"\n  [CWB [REMOVED_ZH:2]] {len(cleaned)} [REMOVED_ZH:2]，"
                  f"[REMOVED_ZH:4] {n_good} [REMOVED_ZH:1] ({n_good/max(len(cleaned),1)*100:.0f}%)")
        return cleaned

    def extract_epw_comparison(self,
                                 cleaned_df: pd.DataFrame,
                                 epw_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        [REMOVED_ZH:1] CWB [REMOVED_ZH:2]and EPW [REMOVED_ZH:6]，compute Ta/RH/WS [REMOVED_ZH:3]。
        epw_hourly [REMOVED_ZH:2] hour [REMOVED_ZH:2]（1–24）[REMOVED_ZH:1] ta, rh, wind_speed [REMOVED_ZH:2]。

        Returns
        -------
        pd.DataFrame
            [REMOVED_ZH:1] delta_ta, delta_rh, delta_ws [REMOVED_ZH:1]（[REMOVED_ZH:2] - EPW）
        """
        if cleaned_df.empty or epw_hourly is None:
            return pd.DataFrame()

        # [REMOVED_ZH:5]（EPW [REMOVED_ZH:3] → CWB [REMOVED_ZH:3]）
        cwb_hourly_mean = cleaned_df.groupby(
            cleaned_df.index.hour)[["ta", "rh", "ws"]].median()
        cwb_hourly_mean.index.name = "hour"

        epw_df = pd.DataFrame(epw_hourly)[["hour", "ta", "rh", "wind_speed"]]
        epw_df = epw_df.set_index("hour")

        merged = cwb_hourly_mean.join(epw_df, how="inner", rsuffix="_epw")
        merged["delta_ta"] = merged["ta"]          - merged["ta_epw"]
        merged["delta_rh"] = merged["rh"]          - merged["rh_epw"]
        merged["delta_ws"] = merged["ws"]          - merged["wind_speed"]
        return merged[["delta_ta", "delta_rh", "delta_ws"]]


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    loader = CWBStationLoader("nonexistent.csv", site_lat=25.07, site_lon=121.55)
    result = loader.load_and_clean(month=7, verbose=True)
    print(f"\nCWB [REMOVED_ZH:4] shape: {result.shape}")
    if not result.empty:
        print(result[["ta", "rh", "ws", "wd", "data_quality"]].head(6))