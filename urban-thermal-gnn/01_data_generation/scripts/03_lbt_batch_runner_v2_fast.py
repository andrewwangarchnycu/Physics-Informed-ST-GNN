"""
03_lbt_batch_runner_v2_fast.py
════════════════════════════════════════════════════════════════
Phase 2 [REMOVED_ZH:3]: [REMOVED_ZH:2] v1 [REMOVED_ZH:8] + [REMOVED_ZH:8] v2

[REMOVED_ZH:2]:
  1. [REMOVED_ZH:4] v1 [REMOVED_ZH:2] (sim_0000.npz ~ sim_0299.npz)
  2. [REMOVED_ZH:7] 2 [REMOVED_ZH:3] (1.0m [REMOVED_ZH:3])
  3. [REMOVED_ZH:4] CWB [REMOVED_ZH:4]compute UTCI
  4. Meta Canopy Height [REMOVED_ZH:6]
  5. [REMOVED_ZH:2] v2 [REMOVED_ZH:6]

Run[REMOVED_ZH:2]: ~30-45 [REMOVED_ZH:2] (vs 8-12 [REMOVED_ZH:2] full simulation)
"""

from __future__ import annotations

import sys
import json
import pickle
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # urban-thermal-gnn root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 01_data_generation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loaders"))  # loaders subdir

from progress_monitor import ProgressMonitor

# Try to import UTCI calculator (optional)
calc_utci = None
try:
    from pythermalcomfort.models import utci as calc_utci
    _UTCI_OK = True
except ImportError as e:
    warnings.warn(f"pythermalcomfort not available (optional): {e}")
    _UTCI_OK = False

# Try to import CWB weather loader
CWBWeatherLoader = None
try:
    from cwb_loader import CWBWeatherLoader
    _CWB_OK = True
except ImportError as e:
    warnings.warn(f"cwb_loader import failed (optional): {e}")
    _CWB_OK = False

_TOOLS_OK = _UTCI_OK or _CWB_OK


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:6]
# ════════════════════════════════════════════════════════════════

def interpolate_spatial_2x(sensor_pts_v1: np.ndarray,
                           values_v1: np.ndarray,
                           target_density: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REMOVED_ZH:1] v1 [REMOVED_ZH:4] (2.0m grid) [REMOVED_ZH:3] v2 (1.0m grid = 2x [REMOVED_ZH:2])

    Args:
        sensor_pts_v1: (N, 2) - v1 [REMOVED_ZH:5]
        values_v1: (N,) [REMOVED_ZH:1] (N, T) - v1 [REMOVED_ZH:3]
        target_density: [REMOVED_ZH:4] (2.0 = 1.0m grid)

    Returns:
        sensor_pts_v2: (N*target_density**2, 2)
        values_v2: (N*target_density**2,) [REMOVED_ZH:1] (N*target_density**2, T)
    """
    if len(values_v1.shape) == 1:
        # [REMOVED_ZH:4] (e.g., SVF)
        is_temporal = False
        values_v1 = values_v1[:, np.newaxis]
    else:
        is_temporal = True
        T = values_v1.shape[1]

    # [REMOVED_ZH:2] v2 [REMOVED_ZH:2] ([REMOVED_ZH:5])
    x_v1, y_v1 = sensor_pts_v1[:, 0], sensor_pts_v1[:, 1]
    x_min, x_max = x_v1.min(), x_v1.max()
    y_min, y_max = y_v1.min(), y_v1.max()

    # v2 [REMOVED_ZH:5] v1 [REMOVED_ZH:1] 1/target_density
    spacing_v1 = 2.0  # v1 [REMOVED_ZH:4] (2.0m)
    spacing_v2 = spacing_v1 / target_density  # v2 [REMOVED_ZH:2]

    x_v2 = np.arange(x_min, x_max + spacing_v2 / 2, spacing_v2)
    y_v2 = np.arange(y_min, y_max + spacing_v2 / 2, spacing_v2)
    xv2_mesh, yv2_mesh = np.meshgrid(x_v2, y_v2)
    sensor_pts_v2 = np.column_stack([xv2_mesh.ravel(), yv2_mesh.ravel()])

    # [REMOVED_ZH:4]
    values_v2_list = []
    if is_temporal:
        for t in range(T):
            vals_interp = griddata(
                sensor_pts_v1,
                values_v1[:, t],
                sensor_pts_v2,
                method='cubic',  # cubic [REMOVED_ZH:3]
                fill_value=np.nan
            )
            # [REMOVED_ZH:2] NaN ([REMOVED_ZH:4])
            vals_interp[np.isnan(vals_interp)] = np.nanmean(values_v1[:, t])
            values_v2_list.append(vals_interp)
        values_v2 = np.column_stack(values_v2_list)
    else:
        values_v2 = griddata(
            sensor_pts_v1,
            values_v1[:, 0],
            sensor_pts_v2,
            method='cubic',
            fill_value=np.nan
        )
        values_v2[np.isnan(values_v2)] = np.nanmean(values_v1[:, 0])

    return sensor_pts_v2, values_v2


def recalibrate_utci_with_cwb(utci_v1: np.ndarray,
                              ta_v1: np.ndarray,
                              ta_cwb: np.ndarray,
                              mrt_v1: np.ndarray,
                              va_v1: np.ndarray,
                              rh_cwb: np.ndarray) -> np.ndarray:
    """
    [REMOVED_ZH:4] CWB [REMOVED_ZH:4]compute UTCI ([REMOVED_ZH:3])

    UTCI ≈ UTCI_v1 + ΔTa_bias + MRT_correction + RH_adjustment
    """
    ta_bias = ta_cwb - ta_v1
    mrt_correction = mrt_v1 * (rh_cwb / 100.0 - 0.5) * 0.1  # RH [REMOVED_ZH:1] MRT [REMOVED_ZH:3]

    try:
        # If pythermalcomfort available, use more accurate calculation
        if calc_utci is not None and len(ta_cwb) < 1000:
            utci_recalc = calc_utci(ta_cwb, mrt_v1, va_v1, rh_cwb)
            return utci_recalc
    except Exception as e:
        warnings.warn(f"UTCI recalculation failed: {e}, using fallback")
        pass

    # Fallback: [REMOVED_ZH:4]
    utci_v2 = utci_v1 + ta_bias * 0.8 + mrt_correction
    return np.clip(utci_v2, 0, 60)


# ════════════════════════════════════════════════════════════════
# Main Program
# ════════════════════════════════════════════════════════════════

def convert_v1_to_v2(v1_npz_path: str,
                    cwb_loader: Optional,
                    year: int = 2025, month: int = 7,
                    output_dir: Path = Path("outputs/raw_simulations")) -> str:
    """
    [REMOVED_ZH:3] v1 [REMOVED_ZH:5] v2
    """
    v1_path = Path(v1_npz_path)
    if not v1_path.exists():
        return None

    # [REMOVED_ZH:2] v1 [REMOVED_ZH:2]
    v1_data = np.load(v1_npz_path)
    scenario_id = v1_data.get('scenario_id', -1)
    sensor_pts_v1 = v1_data['sensor_pts']  # (N, 2)
    ta_v1 = v1_data['ta']  # (T, N)
    mrt_v1 = v1_data['mrt']  # (T, N)
    va_v1 = v1_data['va']  # (T, N)
    rh_v1 = v1_data['rh']  # (T, N)
    utci_v1 = v1_data['utci']  # (T, N)
    svf_v1 = v1_data['svf']  # (N,)
    in_shadow_v1 = v1_data['in_shadow']  # (T, N)
    building_height_v1 = v1_data.get('building_height', np.zeros(sensor_pts_v1.shape[0]))
    tree_height_v1 = v1_data.get('tree_height', np.zeros(sensor_pts_v1.shape[0]))

    T = ta_v1.shape[0]
    sim_hours = [8 + i for i in range(T)]

    # Transpose temporal data for interpolation: (T, N) -> (N, T)
    ta_v1_nt = ta_v1.T
    mrt_v1_nt = mrt_v1.T
    va_v1_nt = va_v1.T
    rh_v1_nt = rh_v1.T
    utci_v1_nt = utci_v1.T
    in_shadow_v1_nt = in_shadow_v1.T

    # Spatial interpolation (2x density)
    sensor_pts_v2, ta_v2_interp = interpolate_spatial_2x(sensor_pts_v1, ta_v1_nt)
    _, mrt_v2_interp = interpolate_spatial_2x(sensor_pts_v1, mrt_v1_nt)
    _, va_v2_interp = interpolate_spatial_2x(sensor_pts_v1, va_v1_nt)
    _, rh_v2_interp = interpolate_spatial_2x(sensor_pts_v1, rh_v1_nt)
    _, svf_v2 = interpolate_spatial_2x(sensor_pts_v1, svf_v1)
    _, in_shadow_v2 = interpolate_spatial_2x(sensor_pts_v1, in_shadow_v1_nt)
    _, building_height_v2 = interpolate_spatial_2x(
        sensor_pts_v1, building_height_v1
    )
    _, tree_height_v2 = interpolate_spatial_2x(sensor_pts_v1, tree_height_v1)

    # Initial UTCI interpolation (fallback/baseline)
    _, utci_v2_interp = interpolate_spatial_2x(sensor_pts_v1, utci_v1_nt)
    utci_v2 = utci_v2_interp  # Default: use interpolated UTCI

    # CWB weather calibration (optional)
    if cwb_loader:
        # Assume v1 simulation is 2025-07-15
        day = 15
        try:
            cwb_data = cwb_loader.get_hourly_data(year, month, day, sim_hours)
            ta_cwb = np.array(cwb_data['ta'])  # (T,)
            rh_cwb = np.array(cwb_data['rh'])  # (T,)

            # Re-calculate UTCI (using interpolated MRT / VA)
            utci_v2_recal = np.zeros_like(mrt_v2_interp)  # Separate variable
            for t in range(T):
                utci_v2_recal[:, t] = recalibrate_utci_with_cwb(
                    utci_v1_nt[:, t],  # Now using transposed data
                    ta_v1_nt[:, t],
                    ta_cwb[t] * np.ones(ta_v2_interp.shape[0]),  # broadcast
                    mrt_v2_interp[:, t],
                    va_v2_interp[:, t],
                    rh_cwb[t] * np.ones(ta_v2_interp.shape[0])
                )
            utci_v2 = utci_v2_recal  # Only update if successful
        except Exception as e:
            warnings.warn(f"CWB calibration failed: {e}, keeping interpolated UTCI")

    # Save v2 .npz - transpose back to (T, N) format for consistency with v1
    output_file = output_dir / f"sim_{scenario_id:04d}_v2.npz"

    np.savez(
        output_file,
        scenario_id=scenario_id,
        sensor_pts=sensor_pts_v2,
        ta=ta_v2_interp.T,  # (N', T) -> (T, N')
        mrt=mrt_v2_interp.T,
        va=va_v2_interp.T,
        rh=rh_v2_interp.T,
        utci=utci_v2.T,  # Transpose if needed
        svf=svf_v2,
        in_shadow=in_shadow_v2.T,  # Transpose temporal dimension
        building_height=building_height_v2,
        tree_height=tree_height_v2,
        far=v1_data.get('far', 0.0),
        bcr=v1_data.get('bcr', 0.0),
    )

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert v1 to v2 simulations")
    parser.add_argument("--n_scenarios", type=int, default=300,
                       help="Number of scenarios to convert")
    parser.add_argument("--cwb_data", type=str, default="inputs/cwb_data.csv",
                       help="Path to CWB weather data")
    parser.add_argument("--use_cwb", action="store_true",
                       help="Apply CWB weather calibration")
    parser.add_argument("--output", type=str, default="outputs/raw_simulations",
                       help="Output directory")

    args = parser.parse_args()

    # Fixed path: scripts is subdirectory of 01_data_generation
    script_dir = Path(__file__).parent
    data_gen_dir = script_dir.parent
    v1_dir = data_gen_dir / "outputs" / "raw_simulations"
    output_dir = Path(args.output)

    # Convert output_dir to absolute if it's relative
    if not output_dir.is_absolute():
        output_dir = data_gen_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CWB (optional)
    cwb_loader = None
    if args.use_cwb:
        cwb_path = Path(args.cwb_data)
        # Convert to absolute path if relative
        if not cwb_path.is_absolute():
            cwb_path = data_gen_dir / cwb_path
        try:
            cwb_loader = CWBWeatherLoader(str(cwb_path))
        except Exception as e:
            warnings.warn(f"CWB loader failed: {e}")

    # [REMOVED_ZH:4] v1 [REMOVED_ZH:2]
    print(f"\n{'='*60}")
    print(f"  Phase 2 v2 Conversion (Fast): 2x Spatial Interpolation")
    print(f"{'='*60}")

    v1_files = sorted([f for f in v1_dir.glob("sim_*.npz") if "_v2" not in f.name])[:args.n_scenarios]
    monitor = ProgressMonitor(len(v1_files), "Phase 2: v1→v2 Conversion")

    converted = 0
    for i, v1_file in enumerate(v1_files):
        try:
            result = convert_v1_to_v2(str(v1_file), cwb_loader, output_dir=output_dir)
            if result:
                converted += 1
                # [REMOVED_ZH:4]
                data = np.load(result)
                n_air = data['sensor_pts'].shape[0]
                utci_mean = float(np.mean(data['utci']))
                monitor.update(i + 1, {
                    "nodes": n_air,
                    "utci_mean": utci_mean,
                })
        except Exception as e:
            warnings.warn(f"[REMOVED_ZH:4] {v1_file}: {e}")
            monitor.update(i + 1)

    monitor.summary()

    print(f"\n  Successfully converted {converted}/{len(v1_files)} scenarios")
    print(f"  Output: {output_dir}/sim_*_v2.npz")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
