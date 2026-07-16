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
                 sim_ws:      Optional[np.ndarray] = None,
                 sensor_locs: Optional[List[Dict]] = None,   # reserved, unused
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
# 6. IDW Spatial Bias Projector (GIS Graph ML calibration)
# ════════════════════════════════════════════════════════════════

def _latlon_to_scene_xy(lats: np.ndarray,
                         lons: np.ndarray,
                         origin_lat: float,
                         origin_lon: float) -> np.ndarray:
    """
    Convert (lat, lon) arrays to local scene coordinates (x, y) in metres.
    Uses equirectangular projection centred on origin.

    Returns
    -------
    coords : (N, 2) float32 — (x_east, y_north) in metres
    """
    R = 6371000.0
    x = R * np.radians(lons - origin_lon) * np.cos(np.radians(origin_lat))
    y = R * np.radians(lats - origin_lat)
    return np.column_stack([x, y]).astype(np.float32)


class IDWSpatialBiasProjector:
    """
    Proper GIS spatial bias field for per-air-node IoT calibration.

    Unlike the global mean bias in SensorToGraphProjector.project(), this
    class computes a spatially-varying bias surface using Inverse Distance
    Weighting (IDW), so each air node gets its own correction derived from
    the distance-weighted contributions of nearby real sensor stations.

    Method
    ------
    For each air node n at time step t:

        bias_ta[t, n] = Σ_s  w(n,s) · Δta_s[t]
                        ─────────────────────────
                             Σ_s  w(n,s)

        w(n,s) = 1 / max(d(n, s), ε)^p          (p=2, ε=1 m)

    where Δta_s[t] = obs_ta_s[t] − sim_ta_at_s[t] is the per-station bias
    computed at the air node nearest to station s.

    Parameters
    ----------
    air_node_coords : (N_nodes, 2) float — scene (x, y) in metres
    sim_hours       : list[int] — hours simulated [8, 9, …, 18]
    origin_lat      : float — scene origin latitude (for geo→scene conversion)
    origin_lon      : float — scene origin longitude
    idw_power       : float — IDW exponent p (default 2)
    max_search_km   : float — only stations within this distance contribute
    min_stations    : int   — fallback to global mean if fewer stations found
    """

    def __init__(self,
                 air_node_coords: np.ndarray,
                 sim_hours:       List[int],
                 origin_lat:      float = 24.80,
                 origin_lon:      float = 120.97,
                 idw_power:       float = 2.0,
                 max_search_km:   float = 10.0,
                 min_stations:    int   = 1):
        self.nodes          = air_node_coords          # (N, 2) metres
        self.sim_hours      = sim_hours
        self.origin_lat     = origin_lat
        self.origin_lon     = origin_lon
        self.idw_power      = idw_power
        self.max_search_m   = max_search_km * 1000.0
        self.min_stations   = min_stations

    # ── public API ────────────────────────────────────────────────────────

    def compute_spatial_bias(
            self,
            iot_df:          Optional[pd.DataFrame],
            station_meta:    List[Dict],
            sim_ta:          np.ndarray,
            sim_rh:          np.ndarray,
    ) -> Dict:
        """
        Compute per-node IDW bias correction field.

        Parameters
        ----------
        iot_df       : hourly IoT DataFrame (index=DatetimeIndex, cols: ta, rh,
                       station_id) — output of IotSensorLoader.load_and_clean()
                       with per-station records preserved (not yet aggregated).
        station_meta : list of dicts {id, name, lat, lon} from station_metadata.json
        sim_ta       : (T, N_nodes) float — simulated temperature field
        sim_rh       : (T, N_nodes) float — simulated relative humidity field

        Returns
        -------
        dict:
          bias_ta_spatial  : (T, N_nodes) — per-node temperature bias [°C]
          bias_rh_spatial  : (T, N_nodes) — per-node RH bias [%]
          bias_ta_global   : (T,)         — fallback global mean bias
          bias_rh_global   : (T,)         — fallback global mean RH bias
          n_stations_used  : int
          quality_weights  : (T,) in [0, 1]
          method           : "IDW_spatial" | "global_mean" | "no_data"
        """
        T = len(self.sim_hours)
        N = len(self.nodes)

        bias_ta_spatial  = np.zeros((T, N), dtype=np.float32)
        bias_rh_spatial  = np.zeros((T, N), dtype=np.float32)
        bias_ta_global   = np.zeros(T,      dtype=np.float32)
        bias_rh_global   = np.zeros(T,      dtype=np.float32)
        quality_weights  = np.zeros(T,      dtype=np.float32)

        if iot_df is None or iot_df.empty or not station_meta:
            return {
                "bias_ta_spatial":  bias_ta_spatial,
                "bias_rh_spatial":  bias_rh_spatial,
                "bias_ta_global":   bias_ta_global,
                "bias_rh_global":   bias_rh_global,
                "n_stations_used":  0,
                "quality_weights":  quality_weights,
                "method":           "no_data",
            }

        # ── 1. Build per-station scene coordinates ─────────────────────
        meta_by_id = {str(s["id"]): s for s in station_meta}

        # If iot_df has per-station records, group by station
        if "station_id" in iot_df.columns:
            station_ids = iot_df["station_id"].astype(str).unique().tolist()
        else:
            # Aggregated dataframe — fall back to global mean
            return self._global_mean_fallback(
                iot_df, sim_ta, sim_rh, T,
                bias_ta_spatial, bias_rh_spatial, quality_weights
            )

        valid_stations = []   # (scene_xy, obs_ta_hourly, obs_rh_hourly)
        for sid in station_ids:
            # Try exact match first; then strip common prefixes (STA_/IOT_/DEMO_)
            meta = meta_by_id.get(sid)
            if meta is None:
                stripped = sid
                for pfx in ("STA_", "sta_", "IOT_", "DEMO_"):
                    if sid.startswith(pfx):
                        stripped = sid[len(pfx):]
                        break
                meta = meta_by_id.get(stripped)
            if meta is None:
                # Final fallback: name-fragment match
                for m in station_meta:
                    if sid in str(m.get("name", "")):
                        meta = m
                        break
            if meta is None:
                continue

            lat, lon = float(meta["lat"]), float(meta["lon"])
            xy = _latlon_to_scene_xy(
                np.array([lat]), np.array([lon]),
                self.origin_lat, self.origin_lon
            )[0]

            sdf = iot_df[iot_df["station_id"].astype(str) == sid]
            obs_ta = self._extract_hourly_series(sdf, "ta")
            obs_rh = self._extract_hourly_series(sdf, "rh")

            if len(obs_ta) < 2:
                continue

            valid_stations.append({
                "sid":    sid,
                "xy":     xy,          # (2,) metres
                "obs_ta": obs_ta,      # dict {hour: float}
                "obs_rh": obs_rh,
            })

        if len(valid_stations) < self.min_stations:
            return self._global_mean_fallback(
                iot_df, sim_ta, sim_rh, T,
                bias_ta_spatial, bias_rh_spatial, quality_weights
            )

        # ── 2. For each station, find nearest air node → compute Δ ────
        station_xys = np.array([s["xy"] for s in valid_stations], dtype=np.float32)

        if _SCIPY:
            tree = cKDTree(self.nodes)
            _, nearest_node_idxs = tree.query(station_xys, k=1, workers=-1)
        else:
            nearest_node_idxs = []
            for xy in station_xys:
                d = np.linalg.norm(self.nodes - xy, axis=1)
                nearest_node_idxs.append(int(np.argmin(d)))
            nearest_node_idxs = np.array(nearest_node_idxs)

        # Per-station hourly bias arrays: shape (T,) with NaN if no obs
        sta_delta_ta = np.full((len(valid_stations), T), np.nan, dtype=np.float32)
        sta_delta_rh = np.full((len(valid_stations), T), np.nan, dtype=np.float32)

        for s_idx, sta in enumerate(valid_stations):
            n_idx = nearest_node_idxs[s_idx]
            for t_idx, hr in enumerate(self.sim_hours):
                obs_ta_val = sta["obs_ta"].get(hr, np.nan)
                obs_rh_val = sta["obs_rh"].get(hr, np.nan)
                sim_ta_val = float(sim_ta[t_idx, n_idx]) if sim_ta.ndim == 2 else float(sim_ta[t_idx])
                sim_rh_val = float(sim_rh[t_idx, n_idx]) if sim_rh.ndim == 2 else float(sim_rh[t_idx])
                if not np.isnan(obs_ta_val):
                    sta_delta_ta[s_idx, t_idx] = obs_ta_val - sim_ta_val
                if not np.isnan(obs_rh_val):
                    sta_delta_rh[s_idx, t_idx] = obs_rh_val - sim_rh_val

        # ── 3. IDW interpolation to all air nodes ─────────────────────
        # Pre-compute distance matrix: (N_stations, N_nodes)
        # For large N, use batched computation to avoid OOM
        BATCH = 500
        dist_sq = np.zeros((len(valid_stations), N), dtype=np.float32)
        for i in range(0, N, BATCH):
            chunk = self.nodes[i:i + BATCH]                     # (batch, 2)
            for s_idx, sta in enumerate(valid_stations):
                diff = chunk - sta["xy"]                        # (batch, 2)
                dist_sq[s_idx, i:i + BATCH] = (diff ** 2).sum(axis=1)

        # Mask stations beyond max_search_m
        max_sq = self.max_search_m ** 2
        within = dist_sq <= max_sq                              # (S, N) bool

        eps = 1.0  # minimum distance in metres
        weights = 1.0 / (np.maximum(dist_sq, eps ** 2) ** (self.idw_power / 2))
        weights[~within] = 0.0                                  # zero out far stations

        for t_idx in range(T):
            delta_ta_t = sta_delta_ta[:, t_idx]                # (S,)
            delta_rh_t = sta_delta_rh[:, t_idx]
            has_ta = ~np.isnan(delta_ta_t)
            has_rh = ~np.isnan(delta_rh_t)

            if has_ta.any():
                w_ta = weights.copy()
                w_ta[~has_ta, :] = 0.0
                w_ta_sum = w_ta.sum(axis=0)
                w_ta_sum_safe = np.where(w_ta_sum > 0, w_ta_sum, 1.0)
                numerator_ta = (w_ta * np.nan_to_num(delta_ta_t)[:, None]).sum(axis=0)
                bias_ta_spatial[t_idx] = np.where(
                    w_ta_sum > 0, numerator_ta / w_ta_sum_safe, 0.0
                ).astype(np.float32)
                bias_ta_global[t_idx] = float(np.nanmean(delta_ta_t[has_ta]))
                quality_weights[t_idx] = 1.0

            if has_rh.any():
                w_rh = weights.copy()
                w_rh[~has_rh, :] = 0.0
                w_rh_sum = w_rh.sum(axis=0)
                w_rh_sum_safe = np.where(w_rh_sum > 0, w_rh_sum, 1.0)
                numerator_rh = (w_rh * np.nan_to_num(delta_rh_t)[:, None]).sum(axis=0)
                bias_rh_spatial[t_idx] = np.where(
                    w_rh_sum > 0, numerator_rh / w_rh_sum_safe, 0.0
                ).astype(np.float32)
                bias_rh_global[t_idx] = float(np.nanmean(delta_rh_t[has_rh]))

        n_used = int(has_ta.sum()) if T > 0 else 0  # stations with any obs
        n_used = sum(
            int(not np.all(np.isnan(sta_delta_ta[i])))
            for i in range(len(valid_stations))
        )

        print(f"  [IDW] Spatial bias computed: {len(valid_stations)} stations, "
              f"{n_used} with obs, {T} time steps, {N} air nodes")

        return {
            "bias_ta_spatial":  bias_ta_spatial,
            "bias_rh_spatial":  bias_rh_spatial,
            "bias_ta_global":   bias_ta_global,
            "bias_rh_global":   bias_rh_global,
            "n_stations_used":  n_used,
            "quality_weights":  quality_weights,
            "method":           "IDW_spatial",
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _extract_hourly_series(self,
                                sdf: pd.DataFrame,
                                col: str) -> Dict[int, float]:
        """Return {hour: median_value} dict from a per-station DataFrame."""
        result = {}
        if col not in sdf.columns:
            return result
        for hr in self.sim_hours:
            subset = sdf[sdf.index.hour == hr] if isinstance(
                sdf.index, pd.DatetimeIndex) else pd.DataFrame()
            if subset.empty and "timestamp" in sdf.columns:
                sdf_ts = sdf.set_index(pd.to_datetime(
                    sdf["timestamp"], errors="coerce"))
                subset = sdf_ts[sdf_ts.index.hour == hr]
            if not subset.empty:
                vals = subset[col].dropna()
                if len(vals):
                    result[hr] = float(vals.median())
        return result

    def _global_mean_fallback(self,
                               iot_df, sim_ta, sim_rh, T,
                               bias_ta_spatial, bias_rh_spatial,
                               quality_weights) -> Dict:
        """Fallback: compute global mean bias and broadcast to all nodes."""
        bias_ta_g = np.zeros(T, dtype=np.float32)
        bias_rh_g = np.zeros(T, dtype=np.float32)

        for t_idx, hr in enumerate(self.sim_hours):
            subset = (iot_df[iot_df.index.hour == hr]
                      if isinstance(iot_df.index, pd.DatetimeIndex)
                      else pd.DataFrame())
            if not subset.empty:
                obs_ta = subset["ta"].median() if "ta" in subset.columns else np.nan
                obs_rh = subset["rh"].median() if "rh" in subset.columns else np.nan
                sim_ta_mean = float(np.nanmean(sim_ta[t_idx]))
                sim_rh_mean = float(np.nanmean(sim_rh[t_idx]))
                if not np.isnan(obs_ta):
                    bias_ta_g[t_idx] = obs_ta - sim_ta_mean
                    quality_weights[t_idx] = 1.0
                if not np.isnan(obs_rh):
                    bias_rh_g[t_idx] = obs_rh - sim_rh_mean

        for t_idx in range(T):
            bias_ta_spatial[t_idx] = bias_ta_g[t_idx]
            bias_rh_spatial[t_idx] = bias_rh_g[t_idx]

        return {
            "bias_ta_spatial":  bias_ta_spatial,
            "bias_rh_spatial":  bias_rh_spatial,
            "bias_ta_global":   bias_ta_g,
            "bias_rh_global":   bias_rh_g,
            "n_stations_used":  0,
            "quality_weights":  quality_weights,
            "method":           "global_mean",
        }


def save_spatial_bias_correction(bias_dict: Dict, out_path: str) -> None:
    """
    Save IDW spatial bias field to JSON.

    Keys saved:
      bias_ta_spatial  : list[list[float]]  — (T, N_nodes)
      bias_rh_spatial  : list[list[float]]
      bias_ta_global   : list[float]        — (T,) global mean fallback
      bias_rh_global   : list[float]
      quality_weights  : list[float]        — (T,)
      n_stations_used  : int
      method           : str
    """
    payload = {
        "bias_ta_spatial":  bias_dict["bias_ta_spatial"].tolist(),
        "bias_rh_spatial":  bias_dict["bias_rh_spatial"].tolist(),
        "bias_ta_global":   bias_dict["bias_ta_global"].tolist(),
        "bias_rh_global":   bias_dict["bias_rh_global"].tolist(),
        "quality_weights":  bias_dict["quality_weights"].tolist(),
        "n_stations_used":  bias_dict.get("n_stations_used", 0),
        "method":           bias_dict.get("method", "unknown"),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [IDW Projector] ✓ Spatial bias saved: {out_path}")


def load_spatial_bias_correction(json_path: str) -> Optional[Dict]:
    """
    Load a previously saved spatial bias field from JSON.
    Returns None if file not found.

    Converts lists back to numpy arrays.
    """
    p = Path(json_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("bias_ta_spatial", "bias_rh_spatial"):
        if key in data:
            data[key] = np.array(data[key], dtype=np.float32)
    for key in ("bias_ta_global", "bias_rh_global", "quality_weights"):
        if key in data:
            data[key] = np.array(data[key], dtype=np.float32)
    return data


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

    # IDW spatial bias test
    print("\n--- IDW Spatial Bias Projector ---")
    idw = IDWSpatialBiasProjector(
        air_node_coords = nodes,
        sim_hours       = sim_hours,
        origin_lat      = 24.80,
        origin_lon      = 120.97,
    )
    # Minimal station_meta and iot_df test (no-data path)
    idw_result = idw.compute_spatial_bias(None, [], sim_ta, sim_rh)
    print(f"Method: {idw_result['method']}")
    print(f"bias_ta_spatial shape: {idw_result['bias_ta_spatial'].shape}")