"""
sensing_integration/sensor_to_graph_features.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/sensing_integration/
[REMOVED_ZH:2] : [REMOVED_ZH:9]（[REMOVED_ZH:4]）[REMOVED_ZH:6] air node，
       compute LBT [REMOVED_ZH:11]。

[REMOVED_ZH:2]：
  1. [REMOVED_ZH:1] KD-tree [REMOVED_ZH:12] air node index
  2. [REMOVED_ZH:6] LBT [REMOVED_ZH:3]and[REMOVED_ZH:5]
  3. compute per-hour bias: Δta, Δrh （[REMOVED_ZH:2] - [REMOVED_ZH:2]）
  4. [REMOVED_ZH:2] bias_correction.json（[REMOVED_ZH:1] 04_sensing_calibration.py [REMOVED_ZH:2]）
  5. compute[REMOVED_ZH:5] UTCI [REMOVED_ZH:10]
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("[sensor_to_graph] scipy [REMOVED_ZH:3]，[REMOVED_ZH:7]。")

try:
    from pythermalcomfort.models import utci as calc_utci
    _PTC = True
except ImportError:
    _PTC = False
    warnings.warn("[sensor_to_graph] pythermalcomfort [REMOVED_ZH:3]，[REMOVED_ZH:2]compute[REMOVED_ZH:2] UTCI。")


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:5]
# ════════════════════════════════════════════════════════════════
def map_stations_to_air_nodes(sensor_coords: np.ndarray,
                                air_node_coords: np.ndarray,
                                max_dist_m: float = 50.0
                                ) -> Dict[int, int]:
    """
    [REMOVED_ZH:12] air node。

    Parameters
    ----------
    sensor_coords  : (N_sensors, 2) float — [REMOVED_ZH:3] (x, y) [m]
    air_node_coords: (N_nodes, 2) float   — air node (x, y) [m]
    max_dist_m     : float                — [REMOVED_ZH:12]

    Returns
    -------
    dict {sensor_idx: air_node_idx}
    """
    if _SCIPY:
        tree = cKDTree(air_node_coords)
        dists, idxs = tree.query(sensor_coords, k=1, workers=-1)
    else:
        dists, idxs = [], []
        for sc in sensor_coords:
            d = np.linalg.norm(air_node_coords - sc, axis=1)
            best = int(np.argmin(d))
            dists.append(d[best])
            idxs.append(best)
        dists = np.array(dists)
        idxs  = np.array(idxs)

    mapping = {}
    for s_idx, (dist, n_idx) in enumerate(zip(dists, idxs)):
        if dist <= max_dist_m:
            mapping[s_idx] = int(n_idx)
        else:
            warnings.warn(f"  [REMOVED_ZH:3] {s_idx} [REMOVED_ZH:3] air node {dist:.0f}m，"
                          f"[REMOVED_ZH:2] {max_dist_m}m，[REMOVED_ZH:2]")
    return mapping


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:2]compute
# ════════════════════════════════════════════════════════════════
def compute_bias(sim_values: np.ndarray,
                  obs_values: np.ndarray,
                  quality_flags: Optional[np.ndarray] = None
                  ) -> Dict[str, float]:
    """
    compute[REMOVED_ZH:2]and[REMOVED_ZH:7]。

    Parameters
    ----------
    sim_values    : (T,) float — LBT [REMOVED_ZH:3]
    obs_values    : (T,) float — [REMOVED_ZH:5]（[REMOVED_ZH:4]）
    quality_flags : (T,) str  — data_quality [REMOVED_ZH:2]，None=[REMOVED_ZH:4]

    Returns
    -------
    dict: bias (obs-sim), rmse, mae, n_valid
    """
    mask = ~(np.isnan(sim_values) | np.isnan(obs_values))
    if quality_flags is not None:
        good_mask = np.array(quality_flags) == "good"
        mask = mask & good_mask

    if mask.sum() < 3:
        return {"bias": 0.0, "rmse": np.nan, "mae": np.nan, "n_valid": 0}

    diff = obs_values[mask] - sim_values[mask]
    return {
        "bias":    float(np.mean(diff)),
        "rmse":    float(np.sqrt(np.mean(diff**2))),
        "mae":     float(np.mean(np.abs(diff))),
        "n_valid": int(mask.sum()),
    }


# ════════════════════════════════════════════════════════════════
# 3. [REMOVED_ZH:2] UTCI compute
# ════════════════════════════════════════════════════════════════
def compute_sensor_utci(ta_obs: np.ndarray,
                         rh_obs: np.ndarray,
                         ws_obs: Optional[np.ndarray] = None,
                         mrt_proxy: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """
    [REMOVED_ZH:6]compute UTCI，[REMOVED_ZH:10]。
    MRT [REMOVED_ZH:4] Ta + 8°C [REMOVED_ZH:8]（Shadow[REMOVED_ZH:2] Ta + 2°C）。
    WS [REMOVED_ZH:4] 1.5 m/s [REMOVED_ZH:9]。
    """
    T = len(ta_obs)
    ws  = ws_obs   if ws_obs   is not None else np.full(T, 1.5)
    mrt = mrt_proxy if mrt_proxy is not None else ta_obs + 8.0

    # [REMOVED_ZH:2] UTCI [REMOVED_ZH:4]
    ws  = np.clip(ws,  0.5, 20.0)
    mrt = np.clip(mrt, ta_obs - 5, ta_obs + 70)

    if not _PTC:
        # [REMOVED_ZH:2] UTCI [REMOVED_ZH:2]（Bröde 2012 [REMOVED_ZH:5]，[REMOVED_ZH:2] < 2°C）
        utci_approx = ta_obs + 0.33 * (mrt - ta_obs) - 0.7 * ws - 4.0
        return utci_approx.astype(np.float32)

    result = calc_utci(
        tdb=ta_obs.ravel(), tr=mrt.ravel(),
        v=ws.ravel(), rh=rh_obs.ravel(), units="SI"
    )
    return np.array(result["utci"], dtype=np.float32)


# ════════════════════════════════════════════════════════════════
# 4. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
class SensorToGraphProjector:
    """
    [REMOVED_ZH:9] air node [REMOVED_ZH:3]，[REMOVED_ZH:6]and[REMOVED_ZH:4]。

    Parameters
    ----------
    air_node_coords : (N, 2) ndarray — air node (x, y) [m]，[REMOVED_ZH:4]
    sim_hours       : list[int]       — [REMOVED_ZH:4]（[REMOVED_ZH:1] [8,9,...,18]）
    site_origin     : (float, float)  — [REMOVED_ZH:6] (x0, y0)，[REMOVED_ZH:8]
    """

    def __init__(self,
                 air_node_coords: np.ndarray,
                 sim_hours:       List[int],
                 site_origin:     Tuple[float, float] = (0.0, 0.0)):
        self.nodes     = air_node_coords
        self.sim_hours = sim_hours
        self.origin    = site_origin

    def _geo_to_scene(self, lat: float, lon: float) -> Tuple[float, float]:
        """[REMOVED_ZH:16] (m)。"""
        R = 6371000.0
        x = R * np.radians(lon - self.origin[1]) * np.cos(np.radians(self.origin[0]))
        y = R * np.radians(lat - self.origin[0])
        return x, y

    def project(self,
                 iot_df:     Optional[pd.DataFrame],
                 cwb_df:     Optional[pd.DataFrame],
                 sim_ta:     np.ndarray,
                 sim_rh:     np.ndarray,
                 sim_ws:     Optional[np.ndarray] = None,
                 sensor_locs: Optional[List[Dict]] = None,
                 ) -> Dict:
        """
        [REMOVED_ZH:5]。

        Parameters
        ----------
        iot_df       : [REMOVED_ZH:2] IoT DataFrame（index=DatetimeIndex）
        cwb_df       : [REMOVED_ZH:2] CWB DataFrame
        sim_ta       : (T, N_nodes) — LBT [REMOVED_ZH:6]
        sim_rh       : (T, N_nodes) — LBT [REMOVED_ZH:2]Relative Humidity
        sim_ws       : (T, N_nodes) — LBT [REMOVED_ZH:2]Wind Speed（[REMOVED_ZH:2]）
        sensor_locs  : [{"lat":, "lon":, "source":}] [REMOVED_ZH:7]

        Returns
        -------
        dict:
          bias_ta, bias_rh, bias_ws — [REMOVED_ZH:6] (T,)
          sensor_utci               — (T, N_nodes) [REMOVED_ZH:4] UTCI（[REMOVED_ZH:7]）
          mapping                   — {sensor_idx: node_idx}
          quality_weights           — (T,) [REMOVED_ZH:4] [0, 1]
        """
        T = len(self.sim_hours)
        N = len(self.nodes)

        bias_ta = np.zeros(T)
        bias_rh = np.zeros(T)
        bias_ws = np.zeros(T)
        sensor_utci     = np.full((T, N), np.nan, dtype=np.float32)
        quality_weights = np.ones(T)   # [REMOVED_ZH:5]

        # ── [REMOVED_ZH:2] IoT + CWB [REMOVED_ZH:2] ────────────────────────
        combined_ta, combined_rh, combined_ws, combined_qflags = (
            [], [], [], []
        )

        def _extract_hourly(df: pd.DataFrame, col: str) -> Dict[int, float]:
            """[REMOVED_ZH:1] DataFrame [REMOVED_ZH:2] sim_hours [REMOVED_ZH:6]。"""
            if df is None or df.empty or col not in df.columns:
                return {}
            result = {}
            for hr in self.sim_hours:
                subset = df[df.index.hour == hr]
                if not subset.empty:
                    vals = subset[col].dropna()
                    if len(vals):
                        result[hr] = float(vals.median())
            return result

        iot_ta  = _extract_hourly(iot_df, "ta") if iot_df is not None else {}
        iot_rh  = _extract_hourly(iot_df, "rh") if iot_df is not None else {}
        cwb_ta  = _extract_hourly(cwb_df, "ta") if cwb_df is not None else {}
        cwb_rh  = _extract_hourly(cwb_df, "rh") if cwb_df is not None else {}
        cwb_ws  = _extract_hourly(cwb_df, "ws") if cwb_df is not None else {}

        for t_idx, hr in enumerate(self.sim_hours):
            # [REMOVED_ZH:5]（IoT [REMOVED_ZH:2]，CWB [REMOVED_ZH:2]）
            obs_ta = iot_ta.get(hr, cwb_ta.get(hr, np.nan))
            obs_rh = iot_rh.get(hr, cwb_rh.get(hr, np.nan))
            obs_ws = cwb_ws.get(hr, np.nan)

            # [REMOVED_ZH:6]（[REMOVED_ZH:3] air nodes）
            sim_ta_mean = float(np.nanmean(sim_ta[t_idx]))
            sim_rh_mean = float(np.nanmean(sim_rh[t_idx]))
            sim_ws_mean = float(np.nanmean(sim_ws[t_idx])) if sim_ws is not None else np.nan

            if not np.isnan(obs_ta):
                bias_ta[t_idx] = obs_ta - sim_ta_mean
            if not np.isnan(obs_rh):
                bias_rh[t_idx] = obs_rh - sim_rh_mean
            if not np.isnan(obs_ws) and not np.isnan(sim_ws_mean):
                bias_ws[t_idx] = obs_ws - sim_ws_mean

            # [REMOVED_ZH:2] UTCI（[REMOVED_ZH:1] CWB [REMOVED_ZH:2]+RH+WS compute，[REMOVED_ZH:7]）
            if not np.isnan(obs_ta) and not np.isnan(obs_rh):
                ws_v = np.full(N, max(0.5, obs_ws if not np.isnan(obs_ws) else 1.5))
                ta_v = np.full(N, obs_ta)
                rh_v = np.full(N, obs_rh)
                sensor_utci[t_idx] = compute_sensor_utci(ta_v, rh_v, ws_v)

        # ── [REMOVED_ZH:4]（[REMOVED_ZH:13]）────────
        has_obs = np.array([
            not np.isnan(iot_ta.get(hr, cwb_ta.get(hr, np.nan)))
            for hr in self.sim_hours
        ], dtype=float)
        quality_weights = has_obs   # 0=[REMOVED_ZH:3] / 1=[REMOVED_ZH:3]

        return {
            "bias_ta":         bias_ta,
            "bias_rh":         bias_rh,
            "bias_ws":         bias_ws,
            "sensor_utci":     sensor_utci,
            "quality_weights": quality_weights,
            "n_hours_with_obs": int(has_obs.sum()),
        }


# ════════════════════════════════════════════════════════════════
# 5. [REMOVED_ZH:6]
# ════════════════════════════════════════════════════════════════
def save_bias_correction(bias_dict: Dict, out_path: str) -> None:
    """[REMOVED_ZH:7] JSON，[REMOVED_ZH:1] 04_sensing_calibration.py [REMOVED_ZH:2]。"""
    payload = {
        "bias_ta_hourly":  bias_dict["bias_ta"].tolist(),
        "bias_rh_hourly":  bias_dict["bias_rh"].tolist(),
        "bias_ws_hourly":  bias_dict["bias_ws"].tolist(),
        "quality_weights": bias_dict["quality_weights"].tolist(),
        "n_hours_with_obs": bias_dict["n_hours_with_obs"],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [Projector] ✓ [REMOVED_ZH:6]: {out_path}")


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    N = 200
    T = 11
    sim_hours = list(range(8, 19))
    nodes = np.random.uniform(0, 60, (N, 2)).astype(np.float32)
    sim_ta = np.random.uniform(28, 33, (T, N)).astype(np.float32)
    sim_rh = np.random.uniform(65, 80, (T, N)).astype(np.float32)

    projector = SensorToGraphProjector(nodes, sim_hours)
    result = projector.project(None, None, sim_ta, sim_rh)
    print(f"bias_ta: {result['bias_ta']}")
    print(f"sensor_utci shape: {result['sensor_utci'].shape}")
    print(f"[REMOVED_ZH:8]: {result['n_hours_with_obs']}")