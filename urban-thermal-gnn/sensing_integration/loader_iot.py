"""
sensing_integration/loader_iot.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/sensing_integration/
[REMOVED_ZH:2] : [REMOVED_ZH:14] CSV，
       [REMOVED_ZH:5] Ta(℃) / RH(%) [REMOVED_ZH:4]，
       [REMOVED_ZH:5]，[REMOVED_ZH:9] DataFrame。

[REMOVED_ZH:4]: [REMOVED_ZH:12]
  https://airtw.moenv.gov.tw/
  [REMOVED_ZH:2]: timestamp, device_id, SiteName, lat, lon, [REMOVED_ZH:2](℃), [REMOVED_ZH:4](%)
  [REMOVED_ZH:2]: [REMOVED_ZH:3] ([REMOVED_ZH:4] 1–5 [REMOVED_ZH:4])

[REMOVED_ZH:4]:
  loader = IotSensorLoader("path/to/iot_data.csv", lat=25.07, lon=121.55)
  hourly = loader.load_and_clean(month=7, radius_km=5.0)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, List, Dict

import re

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from noise_removal import clean_iot_dataframe, resample_iot_to_hourly


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:6]（[REMOVED_ZH:10] CSV [REMOVED_ZH:2]）
# ════════════════════════════════════════════════════════════════
IOT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "DataCreationDate", "time", "[REMOVED_ZH:4]"],
    "station_id": ["device_id", "SiteCode", "[REMOVED_ZH:4]"],
    "station_name": ["SiteName", "[REMOVED_ZH:2]", "site_name"],
    "lat": ["lat", "Latitude", "[REMOVED_ZH:2]"],
    "lon": ["lon", "Longitude", "[REMOVED_ZH:2]"],
    "ta":  ["[REMOVED_ZH:2](℃)", "Temperature", "ta", "AT", "temperature"],
    "rh":  ["[REMOVED_ZH:4](%)", "Humidity", "rh", "RH", "humidity"],
}


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    """[REMOVED_ZH:1] df.columns [REMOVED_ZH:3] key [REMOVED_ZH:7]（[REMOVED_ZH:4]）。"""
    for alias in IOT_COLUMN_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    return None


# ════════════════════════════════════════════════════════════════
# 2. Spatial Filtering[REMOVED_ZH:2]
# ════════════════════════════════════════════════════════════════
def haversine_km(lat1: float, lon1: float,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """compute (lat1,lon1) [REMOVED_ZH:3] (lat2,lon2) [REMOVED_ZH:1] Haversine [REMOVED_ZH:2] [km]。"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ════════════════════════════════════════════════════════════════
# 3. IoT [REMOVED_ZH:3]
# ════════════════════════════════════════════════════════════════
class IotSensorLoader:
    """
    [REMOVED_ZH:10] CSV [REMOVED_ZH:2]and[REMOVED_ZH:2]。

    Parameters
    ----------
    csv_path : str | Path
        CSV [REMOVED_ZH:3]（[REMOVED_ZH:12]）
    site_lat : float
        [REMOVED_ZH:6]（[REMOVED_ZH:2]Spatial Filtering）
    site_lon : float
        [REMOVED_ZH:6]
    encoding : str
        CSV [REMOVED_ZH:2]，[REMOVED_ZH:2] utf-8-sig（[REMOVED_ZH:1] BOM [REMOVED_ZH:3] CSV）
    """

    def __init__(self,
                 csv_path: str,
                 site_lat: float = 25.07,
                 site_lon: float = 121.55,
                 encoding: str = "utf-8-sig"):
        self.csv_path = Path(csv_path)
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.encoding = encoding
        self._raw: Optional[pd.DataFrame] = None

    # ── Folder-based loader (moenviot_temperature/ + moenviot_humidity/) ──

    @staticmethod
    def _month_from_filename(stem: str) -> Optional[int]:
        """Extract month (int) from filename like moenviot_temperature_20250605."""
        m = re.search(r"(\d{8})$", stem)
        if m:
            try:
                return int(m.group(1)[4:6])   # YYYYMMDD → MM
            except (ValueError, IndexError):
                pass
        return None

    def _read_folder(self, month: int,
                      radius_km: float = 10.0) -> pd.DataFrame:
        """
        Load daily IoT CSVs from folder structure:
          <csv_path>/moenviot_temperature/moenviot_temperature_YYYYMMDD.csv
          <csv_path>/moenviot_humidity/moenviot_humidity_YYYYMMDD.csv

        Spatial pre-filtering is applied per-file to avoid loading all-Taiwan
        data into memory (hundreds of millions of rows).

        Returns merged DataFrame with columns:
          station_id, timestamp, lat, lon, ta, rh
        """
        iot_dir  = self.csv_path
        temp_dir = iot_dir / "moenviot_temperature"
        humi_dir = iot_dir / "moenviot_humidity"

        def _within_radius(df: pd.DataFrame) -> pd.DataFrame:
            """Keep only rows within radius_km of the site."""
            if "lat" not in df.columns or "lon" not in df.columns:
                return df
            lat = pd.to_numeric(df["lat"], errors="coerce").fillna(999)
            lon = pd.to_numeric(df["lon"], errors="coerce").fillna(999)
            dist = haversine_km(self.site_lat, self.site_lon,
                                lat.values, lon.values)
            return df[dist <= radius_km].copy()

        def _load_daily(folder: Path, value_col: str) -> pd.DataFrame:
            frames = []
            for fp in sorted(folder.glob("*.csv")):
                if self._month_from_filename(fp.stem) != month:
                    continue
                try:
                    df = pd.read_csv(fp, encoding="utf-8-sig",
                                     low_memory=False)
                    df = _within_radius(df)       # filter BEFORE concat
                    if not df.empty:
                        frames.append(df)
                except Exception as exc:
                    warnings.warn(f"[IoT] Cannot read {fp.name}: {exc}")
            if not frames:
                return pd.DataFrame()
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.rename(columns={
                "deviceId": "station_id",
                value_col:  "ta" if value_col == "temperature" else "rh",
                "time":     "timestamp",
            })
            combined["lat"] = pd.to_numeric(combined["lat"], errors="coerce")
            combined["lon"] = pd.to_numeric(combined["lon"], errors="coerce")
            return combined

        temp_df = _load_daily(temp_dir, "temperature") if temp_dir.is_dir() else pd.DataFrame()
        humi_df = _load_daily(humi_dir, "humidity")    if humi_dir.is_dir() else pd.DataFrame()

        if temp_df.empty and humi_df.empty:
            warnings.warn(f"[IoT] No daily CSV files found for month {month} "
                          f"within {radius_km} km of site in {iot_dir}")
            return pd.DataFrame()

        if temp_df.empty:
            temp_df = humi_df.copy()
            temp_df["ta"] = np.nan

        # Merge humidity onto temperature by (station_id, timestamp)
        if not humi_df.empty and "rh" in humi_df.columns:
            merged = temp_df.merge(
                humi_df[["station_id", "timestamp", "rh"]].drop_duplicates(),
                on=["station_id", "timestamp"],
                how="left",
            )
        else:
            merged = temp_df.copy()
            if "rh" not in merged.columns:
                merged["rh"] = np.nan

        print(f"  [IoT Folder] month={month}: "
              f"{len(merged)} rows, "
              f"{merged['station_id'].nunique()} unique devices "
              f"(within {radius_km} km)")
        return merged

    def _read_raw(self, month: Optional[int] = None,
                  radius_km: float = 10.0) -> pd.DataFrame:
        # Folder mode: csv_path points to a directory containing daily CSVs
        if self.csv_path.is_dir():
            return self._read_folder(month or 7, radius_km=radius_km)

        if self._raw is not None:
            return self._raw
        if not self.csv_path.exists():
            warnings.warn(f"[IoT] CSV [REMOVED_ZH:3]: {self.csv_path}，[REMOVED_ZH:2] demo [REMOVED_ZH:2]")
            return self._make_demo()
        try:
            df = pd.read_csv(self.csv_path, encoding=self.encoding,
                              low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding="big5",
                              low_memory=False)
        self._raw = df
        return df

    def _make_demo(self) -> pd.DataFrame:
        """[REMOVED_ZH:2] 3 [REMOVED_ZH:3]、[REMOVED_ZH:7]（[REMOVED_ZH:3] CSV [REMOVED_ZH:6]）。"""
        np.random.seed(42)
        n   = 1440
        idx = pd.date_range("2023-07-15 00:00", periods=n, freq="T")
        rows = []
        for sid, (dlat, dlon) in enumerate([
                (0.002, 0.001), (-0.001, 0.003), (0.003, -0.002)], 1):
            ta_base = 28 + 3 * np.sin(np.pi * np.arange(n) / 720)
            ta  = ta_base + np.random.randn(n) * 0.8
            rh  = 72 - 8 * np.sin(np.pi * np.arange(n) / 720) + np.random.randn(n) * 3
            # [REMOVED_ZH:4]
            ta[np.random.choice(n, 15, replace=False)] = np.random.uniform(55, 99, 15)
            rh[np.random.choice(n, 10, replace=False)] = np.random.uniform(-20, 130, 10)
            tmp = pd.DataFrame({
                "timestamp":  idx,
                "station_id": f"DEMO_{sid:03d}",
                "lat": self.site_lat + dlat,
                "lon": self.site_lon + dlon,
                "ta":  ta,
                "rh":  rh,
            })
            rows.append(tmp)
        print("  [IoT Demo] [REMOVED_ZH:6]（3 [REMOVED_ZH:2]，1440 [REMOVED_ZH:2]）")
        return pd.concat(rows, ignore_index=True)

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """[REMOVED_ZH:15]。"""
        rename_map = {}
        for std_name in ["timestamp", "station_id", "station_name",
                          "lat", "lon", "ta", "rh"]:
            found = _find_col(df, std_name)
            if found and found != std_name:
                rename_map[found] = std_name
        return df.rename(columns=rename_map)

    def load_and_clean(self,
                        month:      int   = 7,
                        radius_km:  float = 5.0,
                        verbose:    bool  = True) -> pd.DataFrame:
        """
        [REMOVED_ZH:2]、Spatial Filtering、[REMOVED_ZH:3]、[REMOVED_ZH:4]。

        Parameters
        ----------
        month : int
            [REMOVED_ZH:4]（and EPW [REMOVED_ZH:5]）
        radius_km : float
            [REMOVED_ZH:13]
        verbose : bool

        Returns
        -------
        pd.DataFrame
            [REMOVED_ZH:4]，index=DatetimeIndex，[REMOVED_ZH:2]: ta, rh,
            valid_ratio, data_quality, n_stations
        """
        raw = self._read_raw(month=month, radius_km=radius_km)
        raw = self._normalise_columns(raw)

        if verbose:
            print(f"\n[IoT] [REMOVED_ZH:4] {len(raw)} [REMOVED_ZH:1]，[REMOVED_ZH:2]: {list(raw.columns)}")

        # [REMOVED_ZH:4]
        if "timestamp" in raw.columns:
            raw["timestamp"] = pd.to_datetime(raw["timestamp"],
                                               errors="coerce")
            raw = raw.dropna(subset=["timestamp"])
        else:
            raise ValueError("[IoT] [REMOVED_ZH:3] timestamp [REMOVED_ZH:2]")

        # [REMOVED_ZH:4]
        raw = raw[raw["timestamp"].dt.month == month].copy()
        if raw.empty:
            warnings.warn(f"[IoT] [REMOVED_ZH:2] {month} [REMOVED_ZH:3]")
            return pd.DataFrame()

        # Spatial Filtering
        if "lat" in raw.columns and "lon" in raw.columns:
            raw["dist_km"] = haversine_km(
                self.site_lat, self.site_lon,
                raw["lat"].values, raw["lon"].values
            )
            raw = raw[raw["dist_km"] <= radius_km].copy()
            if verbose:
                stations = raw.get("station_id", pd.Series()).nunique()
                print(f"  Spatial Filtering ([REMOVED_ZH:2] {radius_km} km): {stations} [REMOVED_ZH:2]")

        if raw.empty:
            warnings.warn(f"[IoT] [REMOVED_ZH:2] {radius_km} km [REMOVED_ZH:4]")
            return pd.DataFrame()

        # [REMOVED_ZH:7]
        hourly_parts = []
        station_col  = "station_id" if "station_id" in raw.columns else None
        stations     = (raw[station_col].unique()
                        if station_col else ["all"])

        for sid in stations:
            sdf = (raw[raw[station_col] == sid].copy()
                   if station_col else raw.copy())
            sdf = sdf.set_index("timestamp").sort_index()
            sdf = sdf[["ta", "rh"]].apply(pd.to_numeric, errors="coerce")

            cleaned = clean_iot_dataframe(sdf, station_id=str(sid),
                                           verbose=verbose)
            hourly  = resample_iot_to_hourly(cleaned)
            hourly["station_id"] = sid
            hourly_parts.append(hourly)

        if not hourly_parts:
            return pd.DataFrame()

        # [REMOVED_ZH:4]：[REMOVED_ZH:12]
        combined = pd.concat(hourly_parts)
        result   = combined.groupby(combined.index).agg({
            "ta":          "median",
            "rh":          "median",
            "valid_ratio": "mean",
        })
        result["data_quality"] = result["valid_ratio"].apply(
            lambda r: "good" if r >= 0.5 else
                      ("interpolated" if r > 0 else "missing")
        )
        result["n_stations"] = combined.groupby(
            combined.index)["station_id"].nunique()

        if verbose:
            n_good = (result["data_quality"] == "good").sum()
            print(f"\n  [IoT [REMOVED_ZH:2]] [REMOVED_ZH:4] {len(result)} [REMOVED_ZH:1]，"
                  f"[REMOVED_ZH:4] {n_good} [REMOVED_ZH:1] ({n_good/max(len(result),1)*100:.0f}%)")
        return result


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    loader = IotSensorLoader(
        csv_path="nonexistent.csv",
        site_lat=25.07, site_lon=121.55
    )
    result = loader.load_and_clean(month=7, radius_km=5.0, verbose=True)
    print(f"\nIoT [REMOVED_ZH:4] shape: {result.shape}")
    if not result.empty:
        print(result.head(6))
        