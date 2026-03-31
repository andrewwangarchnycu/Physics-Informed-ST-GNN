"""
03_lbt_batch_runner.py  (v3 — [REMOVED_ZH:6] bug + [REMOVED_ZH:4])
════════════════════════════════════════════════════════════════
[REMOVED_ZH:4]:
  Bug1: _shelter_coeff — [REMOVED_ZH:8] 3 [REMOVED_ZH:1] → [REMOVED_ZH:6] (break)
  Bug2: _outdoor_mrt   — lw_gnd [REMOVED_ZH:3] ta+3 → [REMOVED_ZH:1] GHI [REMOVED_ZH:2]compute[REMOVED_ZH:4]
  Bug3: _veg_cooling   — [REMOVED_ZH:6] MRT → [REMOVED_ZH:5] 0
  [REMOVED_ZH:2]: scenario [REMOVED_ZH:6] (trees)，[REMOVED_ZH:1] tree_height [REMOVED_ZH:4]
"""
from __future__ import annotations
import sys, json, math, pickle, time, warnings, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import translate as sh_translate

# ── sys.path [REMOVED_ZH:2]：[REMOVED_ZH:1] shared [REMOVED_ZH:4] ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from pythermalcomfort.models import utci as calc_utci
    _PTC = True
except ImportError:
    warnings.warn("pythermalcomfort [REMOVED_ZH:3]，[REMOVED_ZH:6]。")
    _PTC = False

from shared import EPWData, HourlyClimate, solar_position

# ── [REMOVED_ZH:2] ───────────────────────────────────────────────────────
GRID_SPACING    = 2.0
SENSOR_Z        = 1.5
WIND_REF_Z      = 10.0
WIND_ALPHA_BASE = 0.25
N_SVF_RAYS      = 36

# UrbanGraph [REMOVED_ZH:2] Table A3 [REMOVED_ZH:5]
LAND_COVER_ALBEDO = {0: 0.30, 1: 0.30, 2: 0.40, 3: 0.00, 4: 0.20, 5: 0.00}
VEG_COOLING_BASE  = 1.5   # [°C]
VEG_RADIUS_BASE   = 20.0  # [m]


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def load_calibration(calib_path: str) -> Dict:
    defaults = {"roughness_length": 0.5, "albedo_road": 0.15, "ta_bias_offset": 0.0}
    if calib_path and Path(calib_path).exists():
        with open(calib_path, "r") as f:
            data = json.load(f)
        defaults.update(data.get("params", {}))
        print(f"  [Runner] ✓ [REMOVED_ZH:4]: "
              f"z0={defaults['roughness_length']:.3f}  "
              f"α={defaults['albedo_road']:.3f}  "
              f"Δta={defaults['ta_bias_offset']:.2f}°C")
    else:
        print("  [Runner] [REMOVED_ZH:8]")
    return defaults


def _v_at_sensor(v10: float, roughness: float) -> float:
    """ASHRAE [REMOVED_ZH:3]，[REMOVED_ZH:8]。"""
    alpha = min(WIND_ALPHA_BASE + roughness * 0.02, 0.40)
    return v10 * (SENSOR_Z / WIND_REF_Z) ** alpha


def _compute_svf(pt: Point, buildings: List[Dict],
                  trees: List[Dict] = None,
                  n_rays: int = N_SVF_RAYS) -> float:
    """[REMOVED_ZH:2] SVF [REMOVED_ZH:2]，[REMOVED_ZH:6]and[REMOVED_ZH:4]。"""
    trees   = trees or []
    blocked = 0.0
    for i in range(n_rays):
        angle  = math.radians(i * 360 / n_rays)
        dx, dy = math.cos(angle), math.sin(angle)
        for dist in np.linspace(1.0, 35.0, 10):
            cp  = Point(pt.x + dx * dist, pt.y + dy * dist)
            hit = False
            for b in buildings:
                if b["footprint"].contains(cp):
                    blocked += (math.atan2(b["height"], dist) / (math.pi/2)) / n_rays
                    hit = True
                    break
            if not hit:
                for t in trees:
                    td = math.dist((pt.x, pt.y), t["pos"])
                    if td <= t.get("radius", 3.0):
                        blocked += (math.atan2(t["height"], max(td, 0.5)) / (math.pi/2)) / n_rays
                        hit = True
                        break
            if hit:
                break
    return float(np.clip(1.0 - blocked, 0.05, 0.99))


def _shelter_coeff(pt: Point, buildings: List[Dict],
                    wind_dir: float, roughness: float) -> float:
    """
    ★ Bug1 [REMOVED_ZH:2]：[REMOVED_ZH:5]compute[REMOVED_ZH:4] ([REMOVED_ZH:1] break)。
    [REMOVED_ZH:11] 8/14/20m [REMOVED_ZH:7]，[REMOVED_ZH:2] 3 [REMOVED_ZH:3]。
    """
    rad = math.radians(wind_dir)
    dx  = -math.sin(rad)
    dy  = -math.cos(rad)
    blocked = 0.0
    for b in buildings:
        for dist in (8.0, 14.0, 20.0):
            cp = Point(pt.x + dx * dist, pt.y + dy * dist)
            if b["footprint"].contains(cp):
                h_ratio  = min(1.0, b["height"] / 10.0)
                blocked += h_ratio * (10.0 / dist) * 0.25
                break    # ← [REMOVED_ZH:11]
    rough_factor = 1.0 - min(0.3, roughness * 0.08)
    return float(np.clip((1.0 - blocked) * rough_factor, 0.15, 1.0))


def _in_shadow(pt: Point, buildings: List[Dict], trees: List[Dict],
                sol_alt: float, sol_az: float) -> bool:
    """Shadow Determination（[REMOVED_ZH:2] + [REMOVED_ZH:2]）。"""
    if sol_alt <= 2.0:
        return True
    alt_r = math.radians(sol_alt)
    az_r  = math.radians(sol_az)
    for b in buildings:
        sl  = b["height"] / math.tan(alt_r)
        sdx = -math.sin(az_r) * sl
        sdy = -math.cos(az_r) * sl
        shadow = b["footprint"].union(
            sh_translate(b["footprint"], sdx, sdy)
        ).convex_hull
        if shadow.contains(pt):
            return True
    for t in trees:
        td = math.dist((pt.x, pt.y), t["pos"])
        cr = t.get("radius", t["height"] * 0.4)
        if td <= cr:
            return True
    return False


def _outdoor_mrt(ta: float, ghi: float, dni: float,
                  sol_alt: float, svf: float,
                  in_shad: bool, albedo: float) -> float:
    """
    ★ Bug2 [REMOVED_ZH:2]：Ground Temperature Offset[REMOVED_ZH:1] GHI [REMOVED_ZH:2]compute，[REMOVED_ZH:4] ta+3。
    [REMOVED_ZH:1] Oke (1987) [REMOVED_ZH:15] 15–25°C。
    """
    SB   = 5.67e-8
    eps  = 0.97
    a_sw = 0.70

    # [REMOVED_ZH:2]Ground Temperature Offset
    if in_shad or sol_alt <= 2.0:
        t_gnd_offset = 2.0                                 # Shadow：[REMOVED_ZH:4]
    else:
        t_gnd_offset = 2.0 + (1.0 - albedo) * ghi * 0.012  # [REMOVED_ZH:2]：[REMOVED_ZH:2] ~+15°C

    t_sky  = ta - 20.0 * (1.0 - svf)
    lw_sky = eps * SB * (t_sky + 273.15) ** 4
    lw_gnd = eps * SB * (ta + t_gnd_offset + 273.15) ** 4   # ← [REMOVED_ZH:4]

    sw_dir = 0.0
    if not in_shad and sol_alt > 0:
        alt_r  = math.radians(max(0.5, sol_alt))
        fp_fac = 0.308 * math.cos(alt_r * (1.0 - alt_r**2 / 48.0))
        sw_dir = a_sw * dni * fp_fac

    sw_dif = a_sw * (svf * 0.5 * ghi + (1.0 - svf) * 0.5 * ghi * albedo)
    total  = lw_sky * svf + lw_gnd * (1.0 - svf) + sw_dir + sw_dif
    return float(np.clip((total / (eps * SB)) ** 0.25 - 273.15, ta - 10, ta + 60))


def _veg_cooling(pt: Point, trees: List[Dict], ghi: float) -> float:
    """[REMOVED_ZH:7] [°C]，[REMOVED_ZH:3]and GHI [REMOVED_ZH:2]compute。"""
    if not trees or ghi < 50:
        return 0.0
    cool   = 0.0
    gf     = min(1.0, ghi / 800.0)
    for t in trees:
        d = math.dist((pt.x, pt.y), t["pos"])
        r = t.get("radius", VEG_RADIUS_BASE * (t["height"] / 10.0))
        if d < r:
            cool += VEG_COOLING_BASE * min(1.0, t["height"] / 10.0) * (1.0 - d/r) * gf
    return float(np.clip(cool, 0.0, 4.0))


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def build_sensor_grid(scenario: Dict, spacing: float = GRID_SPACING) -> np.ndarray:
    site  = scenario["site_polygon"]
    bldgs = scenario["buildings"]
    bz    = site.bounds
    bu    = unary_union([b["footprint"] for b in bldgs])

    xs = np.arange(bz[0] + spacing/2, bz[2], spacing)
    ys = np.arange(bz[1] + spacing/2, bz[3], spacing)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    valid = [p for p in pts
             if site.contains(Point(p[0], p[1]))
             and not bu.contains(Point(p[0], p[1]))]
    return np.array(valid, dtype=np.float32)


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:5]
# ════════════════════════════════════════════════════════════════
def simulate_one(scenario: Dict,
                  climate_by_hour: Dict[int, HourlyClimate],
                  epw: EPWData,
                  sim_hours: List[int],
                  calib_params: Dict,
                  spacing: float = GRID_SPACING) -> Optional[Dict]:
    """[REMOVED_ZH:3]，[REMOVED_ZH:5]。"""
    sid            = scenario["scenario_id"]
    site           = scenario["site_polygon"]
    buildings      = scenario["buildings"]
    trees          = scenario.get("trees", [])
    land_cover_map = scenario.get("land_cover_map", None)

    roughness = calib_params.get("roughness_length", 0.5)
    albedo_rd = calib_params.get("albedo_road",      0.15)
    ta_bias   = calib_params.get("ta_bias_offset",   0.0)

    sensor_pts = build_sensor_grid(scenario, spacing)
    N = len(sensor_pts)
    T = len(sim_hours)
    if N < 4:
        warnings.warn(f"[REMOVED_ZH:2] {sid}: [REMOVED_ZH:3] {N} < 4，[REMOVED_ZH:2]")
        return None

    # ── static[REMOVED_ZH:2] ─────────────────────────────────
    svf_arr = np.array(
        [_compute_svf(Point(px, py), buildings, trees) for px, py in sensor_pts],
        dtype=np.float32
    )
    bh_arr = np.zeros(N, dtype=np.float32)
    th_arr = np.zeros(N, dtype=np.float32)
    lc_arr = np.zeros(N, dtype=np.int32)

    for i, (px, py) in enumerate(sensor_pts):
        pt  = Point(px, py)
        # [REMOVED_ZH:6]
        md = np.inf
        for b in buildings:
            d = b["footprint"].distance(pt)
            if d < md:
                md = d
                bh_arr[i] = b["height"]
        # [REMOVED_ZH:4]
        if land_cover_map:
            lc_arr[i] = land_cover_map.get((round(px), round(py)), 0)
        # [REMOVED_ZH:4]
        for t in trees:
            if math.dist((px, py), t["pos"]) <= t.get("radius", VEG_RADIUS_BASE):
                th_arr[i] = max(th_arr[i], t["height"])

    # ── [REMOVED_ZH:3] ───────────────────────────────────
    ta_arr     = np.zeros((T, N), dtype=np.float32)
    mrt_arr    = np.zeros((T, N), dtype=np.float32)
    va_arr     = np.zeros((T, N), dtype=np.float32)
    rh_arr     = np.zeros((T, N), dtype=np.float32)
    shadow_arr = np.zeros((T, N), dtype=bool)

    for t_idx, hr in enumerate(sim_hours):
        clim = climate_by_hour.get(hr)
        if clim is None:
            continue

        sol_alt, sol_az = solar_position(
            epw.latitude, epw.longitude, epw.timezone,
            clim.month, clim.day if clim.day else 15, hr
        )
        v_base = _v_at_sensor(clim.wind_speed, roughness)

        for i, (px, py) in enumerate(sensor_pts):
            pt      = Point(px, py)
            in_shad = _in_shadow(pt, buildings, trees, sol_alt, sol_az)
            sc_coef = _shelter_coeff(pt, buildings, clim.wind_dir, roughness)
            albedo  = LAND_COVER_ALBEDO.get(int(lc_arr[i]), albedo_rd)

            ta_adj  = clim.ta + ta_bias
            mrt_val = _outdoor_mrt(ta_adj, clim.ghi, clim.dni,
                                    sol_alt, svf_arr[i], in_shad, albedo)

            # ★ Bug3 [REMOVED_ZH:2]：[REMOVED_ZH:4] MRT [REMOVED_ZH:2]Shadow[REMOVED_ZH:3]
            vc = _veg_cooling(pt, trees, clim.ghi)
            ta_arr[t_idx, i]     = ta_adj - vc * 0.3
            mrt_arr[t_idx, i]    = mrt_val - (vc * 1.0 if in_shad else 0.0)
            va_arr[t_idx, i]     = max(0.5, v_base * sc_coef)
            rh_arr[t_idx, i]     = clim.rh
            shadow_arr[t_idx, i] = in_shad

    # ── [REMOVED_ZH:3] UTCI ──────────────────────────────
    if _PTC:
        res  = calc_utci(tdb=ta_arr.ravel(), tr=mrt_arr.ravel(),
                          v=va_arr.ravel(), rh=rh_arr.ravel(), units="SI")
        utci_arr = np.array(res["utci"], dtype=np.float32).reshape(T, N)
    else:
        utci_arr = (ta_arr + 0.33 * (mrt_arr - ta_arr)
                    - 0.7 * va_arr - 4.0).astype(np.float32)

    return {
        "scenario_id":     sid,
        "sensor_pts":      sensor_pts,
        "ta":              ta_arr,
        "mrt":             mrt_arr,
        "va":              va_arr,
        "rh":              rh_arr,
        "utci":            utci_arr,
        "svf":             svf_arr,
        "in_shadow":       shadow_arr,
        "building_height": bh_arr,
        "tree_height":     th_arr,
        "land_cover":      lc_arr,
        "sim_hours":       sim_hours,
        "far":             scenario["far_actual"],
        "bcr":             scenario["bcr_actual"],
        "n_buildings":     len(buildings),
    }


def save_npz(result: Dict, out_dir: Path) -> Path:
    sid  = result["scenario_id"]
    path = out_dir / f"sim_{sid:04d}.npz"
    np.savez_compressed(
        path,
        sensor_pts      = result["sensor_pts"],
        ta              = result["ta"],
        mrt             = result["mrt"],
        va              = result["va"],
        rh              = result["rh"],
        utci            = result["utci"],
        svf             = result["svf"],
        in_shadow       = result["in_shadow"].astype(np.uint8),
        building_height = result["building_height"],
        tree_height     = result["tree_height"],
        land_cover      = result["land_cover"],
        sim_hours       = np.array(result["sim_hours"]),
        scenario_id     = np.array(result["scenario_id"]),
        far             = np.array(result["far"]),
        bcr             = np.array(result["bcr"]),
        n_buildings     = np.array(result["n_buildings"]),
    )
    return path


# ════════════════════════════════════════════════════════════════
# Main Program
# ════════════════════════════════════════════════════════════════
def main(epw_pkl:       str   = "../outputs/raw_simulations/epw_data.pkl",
          scenario_pkl: str   = "../outputs/raw_simulations/scenarios.pkl",
          calib_json:   str   = "../outputs/raw_simulations/calibrated_params.json",
          out_dir:      str   = "../outputs/raw_simulations",
          month:        int   = 7,
          hour_start:   int   = 8,
          hour_end:     int   = 18,
          grid_spacing: float = GRID_SPACING,
          workers:      int   = 1,
          skip_existing: bool = True) -> None:

    print("\n[03_lbt_batch_runner v3] ── Batch Simulation（[REMOVED_ZH:1]Physical Correction）──")

    with open(epw_pkl,      "rb") as f: epw       = pickle.load(f)
    with open(scenario_pkl, "rb") as f: scenarios = pickle.load(f)

    calib_params = load_calibration(calib_json)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    typical   = epw.get_typical_day(month=month, stat="hottest")
    clim_map  = {h.hour: h for h in typical}
    sim_hours = list(range(hour_start, hour_end + 1))

    print(f"  {epw.city} | {month}[REMOVED_ZH:1] | {hour_start}–{hour_end}[REMOVED_ZH:1] | "
          f"{len(scenarios)} [REMOVED_ZH:2] | workers={workers}")
    t0   = time.time()
    done = skip = 0

    def _run(sc):
        sid  = sc["scenario_id"]
        path = out / f"sim_{sid:04d}.npz"
        if skip_existing and path.exists():
            return None, sid
        res = simulate_one(sc, clim_map, epw, sim_hours, calib_params, grid_spacing)
        return res, sid

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, sc): sc["scenario_id"] for sc in scenarios}
            for fut in as_completed(futs):
                res, sid = fut.result()
                if res is None: skip += 1
                else: save_npz(res, out); done += 1
                tot = done + skip
                if tot % 50 == 0:
                    print(f"    {tot}/{len(scenarios)}  [REMOVED_ZH:1]:{done}  [REMOVED_ZH:2]:{skip}")
    else:
        for sc in scenarios:
            res, sid = _run(sc)
            if res is None: skip += 1
            else: save_npz(res, out); done += 1
            tot = done + skip
            if tot % 50 == 0:
                print(f"    {tot}/{len(scenarios)}  [REMOVED_ZH:1]:{done}  [REMOVED_ZH:2]:{skip}")

    elapsed = time.time() - t0
    print(f"\n  ✓ [REMOVED_ZH:2] {done} [REMOVED_ZH:1]  [REMOVED_ZH:2] {skip} [REMOVED_ZH:1]  [REMOVED_ZH:2] {elapsed:.1f}s")
    print("[03_lbt_batch_runner] [REMOVED_ZH:2]。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw",      default="../outputs/raw_simulations/epw_data.pkl")
    ap.add_argument("--scenarios",default="../outputs/raw_simulations/scenarios.pkl")
    ap.add_argument("--calib",    default="../outputs/raw_simulations/calibrated_params.json")
    ap.add_argument("--out",      default="../outputs/raw_simulations")
    ap.add_argument("--month",    type=int,   default=7)
    ap.add_argument("--h_start",  type=int,   default=8)
    ap.add_argument("--h_end",    type=int,   default=18)
    ap.add_argument("--spacing",  type=float, default=GRID_SPACING)
    ap.add_argument("--workers",  type=int,   default=1)
    ap.add_argument("--no_skip",  action="store_true")
    args = ap.parse_args()
    main(args.epw, args.scenarios, args.calib, args.out,
         args.month, args.h_start, args.h_end,
         args.spacing, args.workers, not args.no_skip)