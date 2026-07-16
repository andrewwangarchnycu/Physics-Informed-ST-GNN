"""
04_sensing_calibration.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/
[REMOVED_ZH:2] : [REMOVED_ZH:2] 04_envimet_calibration.py。
       (A) [REMOVED_ZH:9]（IoT）and[REMOVED_ZH:5]（CWB）[REMOVED_ZH:2]
       (B) [REMOVED_ZH:7]
       (C) and LBT Batch Simulation[REMOVED_ZH:4]，compute[REMOVED_ZH:4]
       (D) [REMOVED_ZH:3] RMSE，[REMOVED_ZH:10]
       (E) [REMOVED_ZH:2] calibrated_params.json [REMOVED_ZH:1] 03_lbt_batch_runner.py [REMOVED_ZH:2]

[REMOVED_ZH:4] ([REMOVED_ZH:2] abstract "calibration-and-generation workflow"):
  [REMOVED_ZH:6]（[REMOVED_ZH:4]）[REMOVED_ZH:6]，
  [REMOVED_ZH:1] scipy [REMOVED_ZH:5] LBT [REMOVED_ZH:13]：
    roughness_length z₀  ← [REMOVED_ZH:2] ASHRAE Wind Speed[REMOVED_ZH:2]
    albedo_road α        ← [REMOVED_ZH:8] → MRT
    ta_bias_offset β     ← [REMOVED_ZH:6]（[REMOVED_ZH:2] vs [REMOVED_ZH:6]）

Run :
  python 04_sensing_calibration.py
  python 04_sensing_calibration.py \
      --iot_csv path/iot.csv \
      --cwb_csv path/cwb.csv \
      --sim_dir ../outputs/raw_simulations \
      --month 7
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path (works whether run from scripts/ or root)
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]   # urban-thermal-gnn/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from shared import EPWData, HourlyClimate, solar_position
from sensing_integration.loader_iot import IotSensorLoader
from sensing_integration.loader_cwb import CWBStationLoader
from sensing_integration.sensor_to_graph_features import (
    SensorToGraphProjector, save_bias_correction,
    IDWSpatialBiasProjector, save_spatial_bias_correction,
)

try:
    from scipy.optimize import minimize, differential_evolution
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("[04] scipy [REMOVED_ZH:3]，[REMOVED_ZH:10]。")

try:
    from pythermalcomfort.models import utci as calc_utci
    _PTC = True
except ImportError:
    _PTC = False


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:6]
# ════════════════════════════════════════════════════════════════
PARAM_BOUNDS = {
    "roughness_length": (0.01, 2.5,  0.5,  "[REMOVED_ZH:6] z₀ [m]"),
    "albedo_road":      (0.05, 0.45, 0.15, "[REMOVED_ZH:5]"),
    "ta_bias_offset":   (-3.0, 3.0,  0.0,  "[REMOVED_ZH:6] [°C]"),
}


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:6]（[REMOVED_ZH:8]）
# ════════════════════════════════════════════════════════════════
def proxy_ta(ta_epw: np.ndarray,
              roughness: float,
              ta_bias: float) -> np.ndarray:
    """
    [REMOVED_ZH:2] Ta [REMOVED_ZH:2]：EPW + [REMOVED_ZH:6]（[REMOVED_ZH:10]）
    [REMOVED_ZH:6] → [REMOVED_ZH:7] → Ta [REMOVED_ZH:3] 0.3°C/（z₀=1m）
    """
    uhi_effect = (roughness - 0.5) * 0.3
    return ta_epw + uhi_effect + ta_bias


def proxy_mrt_simplified(ta: np.ndarray,
                           ghi: np.ndarray,
                           sol_alt: np.ndarray,
                           albedo: float,
                           svf: float = 0.7) -> np.ndarray:
    """
    [REMOVED_ZH:2] MRT，[REMOVED_ZH:3] albedo [REMOVED_ZH:8]。
    """
    SB   = 5.67e-8
    eps  = 0.97
    a_sw = 0.70

    t_sky  = ta - 20 * (1 - svf)
    lw_sky = eps * SB * (t_sky + 273.15)**4
    lw_gnd = eps * SB * (ta + 3 + 273.15)**4

    alt_r  = np.radians(np.maximum(sol_alt, 0.5))
    fp_fac = 0.308 * np.cos(alt_r * (1 - alt_r**2 / 48))
    sw_dir = np.where(sol_alt > 2, a_sw * ghi * 0.75 * fp_fac, 0.0)
    sw_dif = a_sw * (svf * 0.5 * ghi + (1 - svf) * 0.5 * ghi * albedo)

    total = lw_sky * svf + lw_gnd * (1 - svf) + sw_dir + sw_dif
    mrt   = (total / (eps * SB))**0.25 - 273.15
    return np.clip(mrt, ta - 5, ta + 65)


# ════════════════════════════════════════════════════════════════
# 3. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def run_calibration(obs_ta:    np.ndarray,
                     obs_rh:    np.ndarray,
                     epw_ta:    np.ndarray,
                     epw_ghi:   np.ndarray,
                     sol_alts:  np.ndarray,
                     method:    str = "differential_evolution"
                     ) -> Tuple[Dict[str, float], float]:
    """
    [REMOVED_ZH:5] RMSE [REMOVED_ZH:6]。

    Loss = w_ta * RMSE(ta) + w_mrt_proxy * RMSE(mrt_proxy)
    """
    if not _SCIPY:
        # [REMOVED_ZH:8] fallback
        ta_bias = float(np.nanmean(obs_ta - epw_ta))
        return {
            "roughness_length": 0.5,
            "albedo_road": 0.15,
            "ta_bias_offset": ta_bias,
        }, float(np.nanstd(obs_ta - epw_ta))

    def objective(x: np.ndarray) -> float:
        roughness, albedo, ta_bias = x
        ta_pred  = proxy_ta(epw_ta, roughness, ta_bias)
        # [REMOVED_ZH:2] NaN
        valid    = ~(np.isnan(obs_ta) | np.isnan(ta_pred))
        if valid.sum() < 3:
            return 1e6
        rmse_ta  = float(np.sqrt(np.mean((ta_pred[valid] - obs_ta[valid])**2)))

        mrt_pred = proxy_mrt_simplified(ta_pred, epw_ghi, sol_alts, albedo)
        rmse_mrt = float(np.nanstd(mrt_pred))   # [REMOVED_ZH:1] MRT [REMOVED_ZH:10]
        return 2.0 * rmse_ta + 0.5 * rmse_mrt

    bounds = [
        (PARAM_BOUNDS["roughness_length"][0], PARAM_BOUNDS["roughness_length"][1]),
        (PARAM_BOUNDS["albedo_road"][0],      PARAM_BOUNDS["albedo_road"][1]),
        (PARAM_BOUNDS["ta_bias_offset"][0],   PARAM_BOUNDS["ta_bias_offset"][1]),
    ]

    print(f"  [Calibrate] [REMOVED_ZH:4]: {method}")
    if method == "differential_evolution":
        result = differential_evolution(
            objective, bounds, seed=42, maxiter=200,
            tol=1e-5, workers=1, disp=False, polish=True
        )
    else:
        x0     = [v[2] for v in PARAM_BOUNDS.values()]
        result = minimize(objective, x0[:3], method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 300})

    best = {
        "roughness_length": float(result.x[0]),
        "albedo_road":      float(result.x[1]),
        "ta_bias_offset":   float(result.x[2]),
    }
    rmse_final = float(result.fun)
    return best, rmse_final


# ════════════════════════════════════════════════════════════════
# 4. [REMOVED_ZH:1] sim .npz [REMOVED_ZH:9]
# ════════════════════════════════════════════════════════════════
def load_sim_mean_fields(sim_dir: str,
                          n_samples: int = 20) -> Dict[str, np.ndarray]:
    """
    [REMOVED_ZH:2] n_samples [REMOVED_ZH:1] sim_XXXX.npz [REMOVED_ZH:3]compute[REMOVED_ZH:4]，
    [REMOVED_ZH:7]「[REMOVED_ZH:5]」[REMOVED_ZH:5]。
    """
    sim_path = Path(sim_dir)
    npz_files = sorted(sim_path.glob("sim_????.npz"))[:n_samples]
    if not npz_files:
        warnings.warn(f"[04] [REMOVED_ZH:3] sim_XXXX.npz: {sim_dir}")
        return {}

    ta_list, rh_list = [], []
    sim_hours = None
    for fp in npz_files:
        try:
            d = np.load(fp, allow_pickle=False)
            ta_list.append(np.nanmean(d["ta"], axis=1))   # (T,)
            rh_list.append(np.nanmean(d["rh"], axis=1))
            if sim_hours is None:
                sim_hours = d["sim_hours"].tolist()
        except Exception as e:
            warnings.warn(f"  [REMOVED_ZH:2] {fp.name}: {e}")

    if not ta_list:
        return {}
    return {
        "ta":        np.nanmean(ta_list, axis=0),   # (T,)
        "rh":        np.nanmean(rh_list, axis=0),
        "sim_hours": sim_hours or list(range(8, 19)),
    }


# ════════════════════════════════════════════════════════════════
# 5. Multi-month calibration (v3)
# ════════════════════════════════════════════════════════════════
def run_multi_month_calibration(months:   List[int],
                                 iot_csv:  str,
                                 cwb_csv:  str,
                                 epw_pkl:  str,
                                 sim_dir:  str,
                                 out_dir:  str,
                                 site_lat: float,
                                 site_lon: float,
                                 method:   str,
                                 cwb_dir:  str = "",
                                 verbose:  bool = True) -> Dict:
    """
    Calibrate for each month independently and save seasonal params.

    Returns
    -------
    Dict with per-month results + envelope statistics.
    """
    seasonal: Dict[str, Dict] = {}
    print(f"\n[04] Multi-month calibration: months = {months}")

    for m in months:
        print(f"\n{'─'*60}")
        print(f"  [04] Calibrating month {m} …")
        # Resolve per-month CWB file from directory if provided
        month_cwb = cwb_csv
        if cwb_dir:
            candidate = Path(cwb_dir) / f"cwb_data_{m}.csv"
            if candidate.exists():
                month_cwb = str(candidate)
            else:
                warnings.warn(f"[04] CWB file not found: {candidate}, "
                              f"falling back to cwb_csv={cwb_csv!r}")
        try:
            result = main(
                iot_csv  = iot_csv,
                cwb_csv  = month_cwb,
                epw_pkl  = epw_pkl,
                sim_dir  = sim_dir,
                out_dir  = out_dir,
                month    = m,
                site_lat = site_lat,
                site_lon = site_lon,
                method   = method,
                verbose  = verbose,
            )
            seasonal[str(m)] = result
        except Exception as exc:
            warnings.warn(f"[04] Month {m} calibration failed: {exc}")
            seasonal[str(m)] = {"error": str(exc)}

    # Aggregate: mean params across successful months
    param_keys = ["roughness_length", "albedo_road", "ta_bias_offset"]
    agg_params: Dict[str, List[float]] = {k: [] for k in param_keys}
    for m_result in seasonal.values():
        p = m_result.get("params", {})
        for k in param_keys:
            if k in p:
                agg_params[k].append(float(p[k]))

    seasonal["_seasonal_mean"] = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in agg_params.items()
    }
    seasonal["_months_calibrated"] = [
        int(m) for m in seasonal
        if str(m).lstrip("-").isdigit() and "error" not in seasonal[str(m)]
    ]

    out_json = Path(out_dir) / "calibrated_params_seasonal.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(seasonal, f, indent=2, ensure_ascii=False)
    print(f"\n  [04] ✓ Seasonal calibration saved: {out_json}")

    # Print summary table
    print(f"\n  {'Month':>6}  {'z₀':>6}  {'albedo':>8}  {'ta_bias':>9}  "
          f"{'RMSE':>8}  {'n_obs':>6}")
    print("  " + "─" * 50)
    for m in months:
        r = seasonal.get(str(m), {})
        if "error" in r:
            print(f"  {m:>6}  FAILED: {r['error'][:30]}")
            continue
        p = r.get("params", {})
        print(f"  {m:>6}  "
              f"{p.get('roughness_length', 0):.3f}  "
              f"{p.get('albedo_road', 0):.4f}    "
              f"{p.get('ta_bias_offset', 0):+.3f}     "
              f"{r.get('final_rmse', 0):.3f}    "
              f"{r.get('n_obs_hours', 0):>5}")

    return seasonal


# ════════════════════════════════════════════════════════════════
# 6. Main Program (single month — original behaviour)
# ════════════════════════════════════════════════════════════════
def main(iot_csv:   str = "",
          cwb_csv:   str = "",
          epw_pkl:   str = "../outputs/raw_simulations/epw_data.pkl",
          sim_dir:   str = "../outputs/raw_simulations",
          out_dir:   str = "../outputs/raw_simulations",
          month:     int = 7,
          site_lat:  float = 24.80,   # Hsinchu
          site_lon:  float = 120.97,  # Hsinchu
          method:    str = "differential_evolution",
          verbose:   bool = True) -> Dict:

    print("\n[04_sensing_calibration] ── [REMOVED_ZH:9] ──")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: [REMOVED_ZH:2] EPW ─────────────────────────────────────
    epw_pkl_path = Path(epw_pkl)
    if epw_pkl_path.exists():
        with open(epw_pkl_path, "rb") as f:
            epw: EPWData = pickle.load(f)
        print(f"  EPW: {epw.city} (lat={epw.latitude}, lon={epw.longitude})")
    else:
        warnings.warn(f"[04] EPW pickle [REMOVED_ZH:3]: {epw_pkl}，"
                      "[REMOVED_ZH:2]Run 01_epw_to_forcing.py")
        epw = EPWData(city="Demo", latitude=site_lat, longitude=site_lon,
                      timezone=8.0)

    # ── Step 2: [REMOVED_ZH:6] ─────────────────────────────────
    print(f"\n  Step 2: [REMOVED_ZH:6]...")
    iot_loader = IotSensorLoader(
        csv_path=iot_csv or "nonexistent_iot.csv",
        site_lat=site_lat, site_lon=site_lon
    )
    iot_hourly = iot_loader.load_and_clean(month=month, verbose=verbose)

    cwb_loader = CWBStationLoader(
        csv_path=cwb_csv or "nonexistent_cwb.csv",
        site_lat=site_lat, site_lon=site_lon
    )
    cwb_hourly = cwb_loader.load_and_clean(month=month, verbose=verbose)

    # EPW [REMOVED_ZH:2]
    if not cwb_hourly.empty and hasattr(epw, "hours"):
        typical = epw.get_typical_day(month=month, stat="hottest")
        epw_df  = cwb_loader.extract_epw_comparison(cwb_hourly, typical)
        if not epw_df.empty:
            print(f"\n  EPW vs CWB [REMOVED_ZH:2]（[REMOVED_ZH:3]）:")
            print(f"    ΔTa = {epw_df['delta_ta'].mean():.2f}°C  "
                  f"ΔRH = {epw_df['delta_rh'].mean():.1f}%  "
                  f"ΔWS = {epw_df['delta_ws'].mean():.2f} m/s")

    # ── Step 3: [REMOVED_ZH:7] ──────────────────────────────
    print(f"\n  Step 3: EPW [REMOVED_ZH:5]...")
    try:
        typical_day = epw.get_typical_day(month=month, stat="hottest")
    except Exception:
        # EPW [REMOVED_ZH:9]
        from shared import HourlyClimate
        typical_day = [
            HourlyClimate(month=month, day=15, hour=h,
                          ta=28+3*np.sin(np.pi*(h-6)/12),
                          rh=72, wind_speed=2.5, wind_dir=180,
                          ghi=max(0,800*np.sin(np.pi*(h-6)/12)),
                          dni=max(0,600*np.sin(np.pi*(h-6)/12)),
                          dhi=max(0,200*np.sin(np.pi*(h-6)/12)))
            for h in range(8, 19)
        ]

    sim_hours = [h.hour for h in typical_day if 8 <= h.hour <= 18]
    epw_ta    = np.array([h.ta         for h in typical_day if h.hour in sim_hours])
    epw_ghi   = np.array([h.ghi        for h in typical_day if h.hour in sim_hours])
    sol_alts  = np.array([
        solar_position(epw.latitude, epw.longitude, epw.timezone,
                        h.month, h.day if h.day else 15, h.hour)[0]
        for h in typical_day if h.hour in sim_hours
    ])

    # ── Step 4: [REMOVED_ZH:6] ────────────────────────────────
    print(f"\n  Step 4: [REMOVED_ZH:6]...")
    obs_ta_hourly = np.full(len(sim_hours), np.nan)
    obs_rh_hourly = np.full(len(sim_hours), np.nan)

    for i, hr in enumerate(sim_hours):
        # IoT [REMOVED_ZH:2]，CWB [REMOVED_ZH:2]
        if not iot_hourly.empty and "ta" in iot_hourly.columns:
            subset = iot_hourly[iot_hourly.index.hour == hr]
            if not subset.empty:
                obs_ta_hourly[i] = subset["ta"].median()
                obs_rh_hourly[i] = subset["rh"].median()

        if np.isnan(obs_ta_hourly[i]) and not cwb_hourly.empty:
            subset = cwb_hourly[cwb_hourly.index.hour == hr]
            if not subset.empty:
                obs_ta_hourly[i] = subset["ta"].median()
                obs_rh_hourly[i] = subset["rh"].median()

    n_obs = (~np.isnan(obs_ta_hourly)).sum()
    print(f"    [REMOVED_ZH:8]: {n_obs}/{len(sim_hours)}")

    # ── Step 5: [REMOVED_ZH:6]（[REMOVED_ZH:2] sim_XXXX.npz）──────────
    print(f"\n  Step 5: [REMOVED_ZH:8]...")
    sim_fields = load_sim_mean_fields(sim_dir, n_samples=30)
    if sim_fields:
        # [REMOVED_ZH:6] EPW [REMOVED_ZH:6]（[REMOVED_ZH:3]）
        sim_ta = sim_fields["ta"][:len(sim_hours)]
        sim_rh = sim_fields["rh"][:len(sim_hours)]
        calib_base_ta = sim_ta
        print(f"    [REMOVED_ZH:2] {len(sim_fields.get('sim_hours', []))} [REMOVED_ZH:8]")
    else:
        calib_base_ta = epw_ta
        print("    [REMOVED_ZH:5]，[REMOVED_ZH:2] EPW [REMOVED_ZH:7]")

    # ── Step 6: [REMOVED_ZH:4] ────────────────────────────────────
    print(f"\n  Step 6: Run[REMOVED_ZH:4]...")
    best_params, final_rmse = run_calibration(
        obs_ta    = obs_ta_hourly,
        obs_rh    = obs_rh_hourly,
        epw_ta    = calib_base_ta,
        epw_ghi   = epw_ghi,
        sol_alts  = sol_alts,
        method    = method,
    )

    print(f"\n  [REMOVED_ZH:4] (RMSE = {final_rmse:.3f}°C):")
    for k, v in best_params.items():
        default = PARAM_BOUNDS[k][2]
        unit    = PARAM_BOUNDS[k][3]
        print(f"    {k:22s}: {default:.3f} → {v:.3f}  ({unit})")

    # ── Step 7: [REMOVED_ZH:6] ────────────────────────────────
    result = {
        "month":       month,
        "n_obs_hours": int(n_obs),
        "final_rmse":  float(final_rmse),
        "params":      best_params,
        "bias_ta_mean": float(np.nanmean(obs_ta_hourly - calib_base_ta))
                        if n_obs > 0 else 0.0,
        "bias_rh_mean": float(np.nanmean(
                         obs_rh_hourly[~np.isnan(obs_rh_hourly)] -
                         (sim_fields.get("rh", epw_ta * 0 + 70)[:len(sim_hours)]
                          [~np.isnan(obs_rh_hourly)])
                        )) if n_obs > 0 else 0.0,
        "param_bounds": {
            k: {"min": v[0], "max": v[1], "default": v[2], "unit": v[3]}
            for k, v in PARAM_BOUNDS.items()
        },
    }

    out_json = out / "calibrated_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  [04] ✓ [REMOVED_ZH:6]: {out_json}")

    # ── Step 8: Hourly bias summary (temporal, global mean) ──────────────
    bias_out = out / "hourly_bias.json"
    bias_payload = {
        "sim_hours":       sim_hours,
        "obs_ta":          obs_ta_hourly.tolist(),
        "epw_ta":          epw_ta.tolist(),
        "delta_ta":        (obs_ta_hourly - calib_base_ta).tolist(),
        "obs_rh":          obs_rh_hourly.tolist(),
    }
    with open(bias_out, "w", encoding="utf-8") as f:
        json.dump(bias_payload, f, indent=2)
    print(f"  [04] ✓ Hourly bias saved: {bias_out}")

    # ── Step 9: IDW Spatial Bias Field (GIS Graph ML calibration) ────────
    #
    # Requires:
    #   • outputs/iot_data/station_metadata.json  (from 00_fetch_colife_iot.py)
    #   • iot_csv / iot_dir pointing to real Colife daily CSVs
    #   • sim_ta / sim_rh fields from sim_XXXX.npz to compute per-node bias
    #
    # If prerequisites are absent, this step is silently skipped and the
    # global hourly_bias.json produced in Step 8 is used instead.
    print(f"\n  Step 9: IDW Spatial Bias Field...")

    # Locate station metadata saved by 00_fetch_colife_iot.py
    # iot_csv holds the --iot_dir path when folder mode is used
    if iot_csv and Path(iot_csv).is_dir():
        iot_data_dir = Path(iot_csv)
    else:
        iot_data_dir = _SCRIPT_DIR.parent / "outputs" / "iot_data"
    meta_json       = iot_data_dir / "station_metadata.json"
    spatial_out     = out / "spatial_bias.json"

    station_meta: List = []
    if meta_json.exists():
        with open(meta_json, encoding="utf-8") as f:
            station_meta = json.load(f)
        print(f"    Station metadata: {len(station_meta)} stations from {meta_json}")
    else:
        print(f"    [SKIP] No station_metadata.json at {meta_json}")
        print(f"           Run 00_fetch_colife_iot.py first to enable spatial IDW.")

    # Load per-station IoT DataFrame (folder mode preserves station_id column)
    iot_folder = Path(iot_csv) if (iot_csv and Path(iot_csv).is_dir()) else iot_data_dir
    iot_loader_spatial = IotSensorLoader(
        csv_path=str(iot_folder),
        site_lat=site_lat, site_lon=site_lon
    )

    if station_meta and iot_folder.is_dir():
        # Load full per-station hourly data (not aggregated across stations)
        try:
            iot_folder_raw = iot_loader_spatial._read_folder(
                month=month, radius_km=15.0
            )
            if not iot_folder_raw.empty and "station_id" in iot_folder_raw.columns:
                # Parse timestamps and set index for per-station projection
                iot_folder_raw["timestamp"] = pd.to_datetime(
                    iot_folder_raw["timestamp"], errors="coerce"
                )
                iot_with_stations = iot_folder_raw.dropna(subset=["timestamp"])
                iot_with_stations = iot_with_stations.set_index("timestamp")
            else:
                iot_with_stations = None
        except Exception as exc:
            warnings.warn(f"[04] Could not load IoT folder data: {exc}")
            iot_with_stations = None

        if iot_with_stations is not None and len(iot_with_stations):
            # Build air node coordinate grid from sim_fields
            # Reconstruct a representative (N_nodes, 2) grid from sim_XXXX.npz
            sim_path   = Path(sim_dir)
            npz_files  = sorted(sim_path.glob("sim_????.npz"))
            air_coords = None
            sim_ta_field = None
            sim_rh_field = None

            for fp in npz_files[:5]:
                try:
                    d = np.load(fp, allow_pickle=False)
                    if "air_xy" in d:
                        air_coords   = d["air_xy"]          # (N_nodes, 2) metres
                        sim_ta_field = d["ta"][:, :len(air_coords)]   # (T, N)
                        sim_rh_field = d["rh"][:, :len(air_coords)]
                        break
                    elif "ta" in d:
                        _,   N_f = d["ta"].shape
                        # Synthesise a flat grid (placeholder when xy not stored)
                        side = int(N_f ** 0.5)
                        xs   = np.tile(np.arange(side) * 1.0, side)[:N_f]
                        ys   = np.repeat(np.arange(side) * 1.0, side)[:N_f]
                        air_coords   = np.column_stack([xs, ys]).astype(np.float32)
                        sim_ta_field = d["ta"]              # (T, N)
                        sim_rh_field = d["rh"]
                        break
                except Exception:
                    continue

            if air_coords is not None and sim_ta_field is not None:
                idw = IDWSpatialBiasProjector(
                    air_node_coords = air_coords,
                    sim_hours       = sim_hours,
                    origin_lat      = site_lat,
                    origin_lon      = site_lon,
                    idw_power       = 2.0,
                    max_search_km   = 10.0,
                    min_stations    = 1,
                )
                spatial_bias = idw.compute_spatial_bias(
                    iot_df       = iot_with_stations,
                    station_meta = station_meta,
                    sim_ta       = sim_ta_field,
                    sim_rh       = sim_rh_field,
                )
                save_spatial_bias_correction(spatial_bias, str(spatial_out))
                result["spatial_bias_method"]   = spatial_bias["method"]
                result["spatial_bias_stations"] = spatial_bias["n_stations_used"]
                print(f"  [04] ✓ Spatial bias ({spatial_bias['method']}): "
                      f"{spatial_bias['n_stations_used']} stations, "
                      f"shape {spatial_bias['bias_ta_spatial'].shape}")
            else:
                print("    [SKIP] Could not extract air_xy coordinates from npz files.")
        else:
            print("    [SKIP] IoT folder data empty or missing station_id column.")
    else:
        print("    [SKIP] Prerequisites missing — using global hourly bias only.")
        result["spatial_bias_method"] = "global_mean_fallback"

    # Persist updated result with spatial bias info
    with open(out / "calibrated_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[04_sensing_calibration] Complete.\n")
    return result


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iot_csv",  default="",
                    help="Single merged IoT CSV (optional; overridden by --iot_dir)")
    ap.add_argument("--iot_dir",  default="",
                    help="Folder containing moenviot_temperature/ and "
                         "moenviot_humidity/ sub-directories of daily CSVs")
    ap.add_argument("--cwb_csv",  default="",
                    help="CWB station CSV path (optional; overridden by --cwb_dir)")
    ap.add_argument("--cwb_dir",  default="",
                    help="Folder containing cwb_data_<month>.csv files "
                         "(used in multi-month mode)")
    ap.add_argument("--epw",      default="../outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--sim_dir",  default="../outputs/raw_simulations")
    ap.add_argument("--out",      default="../outputs/raw_simulations")
    ap.add_argument("--month",    type=int,   default=7,
                    help="Single month (ignored when --months is set)")
    ap.add_argument("--months",   default="",
                    help="Comma-separated list of months for multi-month "
                         "calibration, e.g. 6,7,8,9 (summer season)")
    # Hsinchu site (default)
    ap.add_argument("--lat",      type=float, default=24.80)
    ap.add_argument("--lon",      type=float, default=120.97)
    ap.add_argument("--method",   default="differential_evolution",
                    choices=["differential_evolution", "L-BFGS-B"])
    args = ap.parse_args()

    # --iot_dir takes precedence over --iot_csv
    iot_source = args.iot_dir if args.iot_dir else args.iot_csv

    if args.months:
        # Multi-month calibration (v3)
        month_list = [int(m.strip()) for m in args.months.split(",")]
        run_multi_month_calibration(
            months   = month_list,
            iot_csv  = iot_source,
            cwb_csv  = args.cwb_csv,
            cwb_dir  = args.cwb_dir,
            epw_pkl  = args.epw,
            sim_dir  = args.sim_dir,
            out_dir  = args.out,
            site_lat = args.lat,
            site_lon = args.lon,
            method   = args.method,
        )
    else:
        # Single-month calibration (original behaviour)
        cwb_src = args.cwb_csv
        if args.cwb_dir:
            candidate = Path(args.cwb_dir) / f"cwb_data_{args.month}.csv"
            if candidate.exists():
                cwb_src = str(candidate)
        main(
            iot_csv  = iot_source,
            cwb_csv  = cwb_src,
            epw_pkl  = args.epw,
            sim_dir  = args.sim_dir,
            out_dir  = args.out,
            month    = args.month,
            site_lat = args.lat,
            site_lon = args.lon,
            method   = args.method,
        )