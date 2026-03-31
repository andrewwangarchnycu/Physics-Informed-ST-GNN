"""
sensing_integration/noise_removal.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/sensing_integration/
[REMOVED_ZH:2] : [REMOVED_ZH:8]（[REMOVED_ZH:3]）and[REMOVED_ZH:5]（[REMOVED_ZH:3]）[REMOVED_ZH:2]
       [REMOVED_ZH:10]，[REMOVED_ZH:8] DataFrame。

[REMOVED_ZH:7]:
  Layer 1 — [REMOVED_ZH:6] (Physical bound check)
             [REMOVED_ZH:10]，[REMOVED_ZH:9]。
  Layer 2 — [REMOVED_ZH:6] (Robust statistical outlier)
             IQR + MAD [REMOVED_ZH:4]，[REMOVED_ZH:8]and[REMOVED_ZH:3]。
  Layer 3 — [REMOVED_ZH:5] (Temporal spike detection)
             Rolling std + [REMOVED_ZH:5]，[REMOVED_ZH:7]。

[REMOVED_ZH:4]:
  [REMOVED_ZH:6] (≤2 hr) [REMOVED_ZH:4]；[REMOVED_ZH:2] 2 hr [REMOVED_ZH:2] NaN，
  [REMOVED_ZH:1] data_quality [REMOVED_ZH:4] [good / interpolated / missing]。
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:6]（[REMOVED_ZH:4]）
# ════════════════════════════════════════════════════════════════
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ta":   (-5.0,  50.0),   # [REMOVED_ZH:4] [°C]
    "rh":   (0.0,  100.0),   # Relative Humidity [%]
    "ws":   (0.0,   60.0),   # Wind Speed [m/s]，60 m/s [REMOVED_ZH:7]
    "wd":   (0.0,  360.0),   # [REMOVED_ZH:2] [deg]
    "pres": (950.0, 1040.0), # [REMOVED_ZH:2] [hPa]
    "rain": (0.0,  300.0),   # [REMOVED_ZH:3] [mm/hr]
}

# [REMOVED_ZH:7]（[REMOVED_ZH:11]）
TEMPORAL_THRESHOLDS: Dict[str, float] = {
    "ta":   3.0,    # [REMOVED_ZH:9] [°C]（IoT）
    "rh":   10.0,   # [REMOVED_ZH:9] [%]
    "ws":   8.0,    # [REMOVED_ZH:5]Wind Speed[REMOVED_ZH:2] [m/s]（CWB）
}

# MAD [REMOVED_ZH:2]（[REMOVED_ZH:8] sigma [REMOVED_ZH:2]，3.5 [REMOVED_ZH:2] ~99.98%）
MAD_MULTIPLIER  = 3.5
# IQR [REMOVED_ZH:2]（[REMOVED_ZH:2] Tukey fences）
IQR_MULTIPLIER  = 1.5
# [REMOVED_ZH:10]（[REMOVED_ZH:7]）
MAX_INTERP_GAP_MIN = 120   # 120 [REMOVED_ZH:2]（IoT）
MAX_INTERP_GAP_HR  = 2     # 2 [REMOVED_ZH:2]（CWB）


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:7]
# ════════════════════════════════════════════════════════════════
class SensorNoiseRemover:
    """
    [REMOVED_ZH:10]，[REMOVED_ZH:2] IoT（[REMOVED_ZH:3]）and CWB（[REMOVED_ZH:3]）[REMOVED_ZH:4]。

    Parameters
    ----------
    freq : str
        [REMOVED_ZH:4]: 'T' ([REMOVED_ZH:3], IoT) [REMOVED_ZH:1] 'H' ([REMOVED_ZH:3], CWB)
    variables : list[str]
        [REMOVED_ZH:10]（[REMOVED_ZH:3] PHYSICAL_BOUNDS [REMOVED_ZH:3]）
    mad_mult : float
        MAD [REMOVED_ZH:2]，[REMOVED_ZH:2] 3.5
    iqr_mult : float
        IQR [REMOVED_ZH:2]，[REMOVED_ZH:2] 1.5
    verbose : bool
        [REMOVED_ZH:11]
    """

    def __init__(self,
                 freq:      str   = "T",
                 variables: Optional[List[str]] = None,
                 mad_mult:  float = MAD_MULTIPLIER,
                 iqr_mult:  float = IQR_MULTIPLIER,
                 verbose:   bool  = True):
        self.freq      = freq
        self.variables = variables or list(PHYSICAL_BOUNDS.keys())
        self.mad_mult  = mad_mult
        self.iqr_mult  = iqr_mult
        self.verbose   = verbose
        self._max_gap  = MAX_INTERP_GAP_MIN if freq == "T" else MAX_INTERP_GAP_HR

    # ── Layer 1: [REMOVED_ZH:6] ─────────────────────────────────
    def _layer1_physical_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        total_removed = 0
        for col in self.variables:
            if col not in df.columns:
                continue
            lo, hi = PHYSICAL_BOUNDS.get(col, (-np.inf, np.inf))
            mask = (df[col] < lo) | (df[col] > hi)
            n = mask.sum()
            if n > 0:
                df.loc[mask, col] = np.nan
                total_removed += n
        if self.verbose:
            print(f"    [L1 Physical]  [REMOVED_ZH:2] {total_removed} [REMOVED_ZH:4]")
        return df

    # ── Layer 2: IQR + MAD [REMOVED_ZH:8] ──────────────────
    def _layer2_statistical_outlier(self, df: pd.DataFrame,
                                     group_by: str = "month") -> pd.DataFrame:
        """
        [REMOVED_ZH:5]compute[REMOVED_ZH:4]，[REMOVED_ZH:9]。
        IQR [REMOVED_ZH:1]and MAD [REMOVED_ZH:7]，[REMOVED_ZH:5]。
        """
        if group_by == "month" and "month" in df.columns:
            groups = df.groupby("month")
        else:
            groups = [("all", df)]

        total_removed = 0
        for _, grp_df in groups:
            for col in self.variables:
                if col not in grp_df.columns:
                    continue
                s = grp_df[col].dropna()
                if len(s) < 10:
                    continue

                # IQR [REMOVED_ZH:1]
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                iqr_lo = q1 - self.iqr_mult * iqr
                iqr_hi = q3 + self.iqr_mult * iqr

                # MAD [REMOVED_ZH:1]（[REMOVED_ZH:9]）
                med    = s.median()
                mad    = (s - med).abs().median()
                if mad < 1e-9:
                    mad = s.std() * 0.6745   # fallback
                mad_lo = med - self.mad_mult * mad / 0.6745
                mad_hi = med + self.mad_mult * mad / 0.6745

                # [REMOVED_ZH:2]：[REMOVED_ZH:12]
                # ([REMOVED_ZH:5]，[REMOVED_ZH:10])
                outlier_mask = (
                    ((grp_df[col] < iqr_lo) | (grp_df[col] > iqr_hi)) &
                    ((grp_df[col] < mad_lo) | (grp_df[col] > mad_hi))
                )
                n = outlier_mask.sum()
                if n > 0:
                    df.loc[grp_df.index[outlier_mask], col] = np.nan
                    total_removed += n

        if self.verbose:
            print(f"    [L2 IQR+MAD]   [REMOVED_ZH:2] {total_removed} [REMOVED_ZH:6]")
        return df

    # ── Layer 3: [REMOVED_ZH:7]（[REMOVED_ZH:2] Spike Detection）────────
    def _layer3_temporal_spike(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        compute[REMOVED_ZH:6]，[REMOVED_ZH:2] TEMPORAL_THRESHOLDS [REMOVED_ZH:6]。
        [REMOVED_ZH:1]compute rolling(window=5) [REMOVED_ZH:1] std，[REMOVED_ZH:2] 3σ [REMOVED_ZH:7]。
        """
        total_removed = 0
        for col in self.variables:
            if col not in df.columns:
                continue
            thresh = TEMPORAL_THRESHOLDS.get(col, None)
            if thresh is None:
                continue

            s = df[col].copy()
            # [REMOVED_ZH:3]
            diff_mask = s.diff().abs() > thresh
            # Rolling std [REMOVED_ZH:1]（5 [REMOVED_ZH:5]）
            roll_std  = s.rolling(window=5, center=True).std()
            roll_mean = s.rolling(window=5, center=True).mean()
            spike_mask = (s - roll_mean).abs() > 3 * roll_std.clip(lower=0.01)

            combined = diff_mask & spike_mask
            n = combined.sum()
            if n > 0:
                df.loc[combined, col] = np.nan
                total_removed += n

        if self.verbose:
            print(f"    [L3 Spike]     [REMOVED_ZH:2] {total_removed} [REMOVED_ZH:4]")
        return df

    # ── [REMOVED_ZH:4]and[REMOVED_ZH:4] ────────────────────────────────────
    def _interpolate_and_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [REMOVED_ZH:9]（≤ max_gap），[REMOVED_ZH:6] NaN。
        [REMOVED_ZH:2] data_quality [REMOVED_ZH:2]: good / interpolated / missing
        """
        for col in self.variables:
            if col not in df.columns:
                continue

            missing_before = df[col].isna()
            # compute[REMOVED_ZH:7]
            gap_lengths = missing_before.groupby(
                (~missing_before).cumsum()
            ).transform("sum")
            can_interp = missing_before & (gap_lengths <= self._max_gap)

            # [REMOVED_ZH:11] linear interpolation
            df_interp = df[col].copy()
            df_interp[~can_interp & missing_before] = np.nan
            df[col] = df_interp.interpolate(method="time" if isinstance(
                df.index, pd.DatetimeIndex) else "linear",
                limit_direction="both",
                limit=self._max_gap
            )

        # [REMOVED_ZH:4]
        all_vars = [c for c in self.variables if c in df.columns]
        if not all_vars:
            df["data_quality"] = "good"
            return df

        still_missing  = df[all_vars].isna().any(axis=1)
        was_missing    = pd.concat(
            [df[c].isna() for c in all_vars], axis=1
        ).any(axis=1)
        interpolated   = was_missing & ~still_missing

        df["data_quality"] = "good"
        df.loc[interpolated,  "data_quality"] = "interpolated"
        df.loc[still_missing, "data_quality"] = "missing"

        n_i = interpolated.sum()
        n_m = still_missing.sum()
        if self.verbose:
            print(f"    [Interp]       [REMOVED_ZH:2] {n_i} [REMOVED_ZH:1]，[REMOVED_ZH:2] NaN {n_m} [REMOVED_ZH:1]")
        return df

    # ── [REMOVED_ZH:3] ────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame,
               station_id: Optional[str] = None) -> pd.DataFrame:
        """
        Run[REMOVED_ZH:9]。

        Parameters
        ----------
        df : pd.DataFrame
            [REMOVED_ZH:6]，index [REMOVED_ZH:2] DatetimeIndex（[REMOVED_ZH:2] timestamp [REMOVED_ZH:1]）
        station_id : str, optional
            [REMOVED_ZH:2] ID（[REMOVED_ZH:2] verbose [REMOVED_ZH:4]）

        Returns
        -------
        pd.DataFrame
            [REMOVED_ZH:4] DataFrame，[REMOVED_ZH:1] data_quality [REMOVED_ZH:2]
        """
        sid = station_id or "unknown"
        n0  = len(df)
        if self.verbose:
            print(f"\n  [NoiseRemoval] [REMOVED_ZH:2] {sid}  [REMOVED_ZH:2] {n0} [REMOVED_ZH:1]")

        # [REMOVED_ZH:2] DatetimeIndex
        if "timestamp" in df.columns and not isinstance(
                df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # [REMOVED_ZH:2] month [REMOVED_ZH:1]（[REMOVED_ZH:3]）
        df["month"] = df.index.month

        df = self._layer1_physical_bounds(df)
        df = self._layer2_statistical_outlier(df, group_by="month")
        df = self._layer3_temporal_spike(df)
        df = self._interpolate_and_flag(df)

        df = df.drop(columns=["month"], errors="ignore")

        n_good = (df["data_quality"] == "good").sum()
        if self.verbose:
            print(f"    [Done]         [REMOVED_ZH:4] {n_good}/{n0} "
                  f"({n_good/max(n0,1)*100:.1f}%)")
        return df


# ════════════════════════════════════════════════════════════════
# 3. [REMOVED_ZH:8]
# ════════════════════════════════════════════════════════════════
def clean_iot_dataframe(df: pd.DataFrame,
                         station_id: str = "iot",
                         verbose: bool = True) -> pd.DataFrame:
    """
    [REMOVED_ZH:9]（[REMOVED_ZH:3]）[REMOVED_ZH:2]。
    [REMOVED_ZH:4]: ta, rh（[REMOVED_ZH:5]，[REMOVED_ZH:1]Wind Speed）
    """
    remover = SensorNoiseRemover(
        freq="T", variables=["ta", "rh"],
        verbose=verbose
    )
    return remover.clean(df.copy(), station_id)


def clean_cwb_dataframe(df: pd.DataFrame,
                         station_id: str = "cwb",
                         verbose: bool = True) -> pd.DataFrame:
    """
    [REMOVED_ZH:10]（[REMOVED_ZH:3]）[REMOVED_ZH:2]。
    [REMOVED_ZH:4]: pres, ta, rh, ws, wd, rain
    [REMOVED_ZH:7]: ta, rh, ws, wd（[REMOVED_ZH:2]）
    """
    htc_vars = ["ta", "rh", "ws", "wd"]   # [REMOVED_ZH:11]
    remover = SensorNoiseRemover(
        freq="H", variables=htc_vars,
        verbose=verbose
    )
    return remover.clean(df.copy(), station_id)


def resample_iot_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    [REMOVED_ZH:4] IoT [REMOVED_ZH:7]（[REMOVED_ZH:7]）。
    [REMOVED_ZH:2]compute[REMOVED_ZH:9] valid_ratio。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    quality_col  = "data_quality" if "data_quality" in df.columns else None

    agg_dict = {c: "median" for c in numeric_cols}
    hourly   = df[numeric_cols].resample("H").agg(agg_dict)

    if quality_col:
        valid_counts = (df["data_quality"] == "good").resample("H").sum()
        total_counts = df["data_quality"].resample("H").count()
        hourly["valid_ratio"] = (valid_counts / total_counts.clip(lower=1)
                                  ).reindex(hourly.index).fillna(0.0)
        hourly["data_quality"] = hourly["valid_ratio"].apply(
            lambda r: "good" if r >= 0.5 else
                      ("interpolated" if r > 0 else "missing")
        )

    return hourly


# ════════════════════════════════════════════════════════════════
# 4. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)
    n = 1440   # [REMOVED_ZH:2] 1440 [REMOVED_ZH:2]
    idx = pd.date_range("2023-07-15 00:00", periods=n, freq="T")
    ta_raw = 28 + 4 * np.sin(np.pi * np.arange(n) / 720) + np.random.randn(n) * 0.5
    # [REMOVED_ZH:4]
    ta_raw[100:102] = 99.0    # [REMOVED_ZH:2]
    ta_raw[300]     = 15.0    # [REMOVED_ZH:2]
    ta_raw[600:660] = np.nan  # [REMOVED_ZH:3]（60[REMOVED_ZH:2]，[REMOVED_ZH:3]）
    ta_raw[900:1020]= np.nan  # [REMOVED_ZH:3]（120[REMOVED_ZH:2]，[REMOVED_ZH:2]）

    rh_raw = 70 + 10 * np.random.randn(n)
    rh_raw[200]     = 150.0   # [REMOVED_ZH:2]
    rh_raw[400]     = -5.0    # [REMOVED_ZH:2]

    demo = pd.DataFrame({"ta": ta_raw, "rh": rh_raw}, index=idx)
    print("=== IoT [REMOVED_ZH:5] ===")
    cleaned = clean_iot_dataframe(demo, station_id="demo_iot")

    quality_counts = cleaned["data_quality"].value_counts()
    print(f"\n[REMOVED_ZH:4]:\n{quality_counts}")

    hourly = resample_iot_to_hourly(cleaned)
    print(f"\n[REMOVED_ZH:4] shape: {hourly.shape}")
    print(hourly[["ta", "rh", "valid_ratio", "data_quality"]].head(6))