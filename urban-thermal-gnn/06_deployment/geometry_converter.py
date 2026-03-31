"""
06_deployment/geometry_converter.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:1] Rhino/GH [REMOVED_ZH:2] JSON [REMOVED_ZH:3] GNN [REMOVED_ZH:4]

[REMOVED_ZH:2] JSON [REMOVED_ZH:2]（[REMOVED_ZH:2] GhPython）：
{
  "site_boundary": [[x,y], ...],         # [REMOVED_ZH:4]
  "buildings": [
    { "footprint": [[x,y],...],          # [REMOVED_ZH:4]（[REMOVED_ZH:6]）
      "height": 18.0,                    # [REMOVED_ZH:2] (m)
      "floor_count": 4 }
  ],
  "trees": [
    { "x":10,"y":20,"radius":3,"height":5 }
  ],
  "sensor_resolution": 2.0              # [REMOVED_ZH:4] (m)
}

[REMOVED_ZH:2]：
{
  "sensor_pts":   (N, 2) float32         # [REMOVED_ZH:5]
  "obj_feat":     (N_obj, 7) float32     # [REMOVED_ZH:6]
  "air_feat":     (N_air, T, 8) float32  # [REMOVED_ZH:6]
  "static_edges": {"contiguity": (2,E),  # KNN air-air [REMOVED_ZH:1]（[REMOVED_ZH:1] N_obj offset）
                   "semantic":  (2,E2)}  # fully-connected obj-obj [REMOVED_ZH:1]
}

[REMOVED_ZH:2]compute（[REMOVED_ZH:3]，[REMOVED_ZH:2] EnergyPlus [REMOVED_ZH:2]）：
  SVF       : [REMOVED_ZH:10]
  in_shadow : [REMOVED_ZH:8]
  MRT       : Tmrt ≈ Ta + f_solar(SVF, shadow, GHI) + f_IR
  ta/rh/va  : [REMOVED_ZH:4] EPW [REMOVED_ZH:3] + [REMOVED_ZH:8]
"""
from __future__ import annotations
import sys, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "02_graph_construction"))
sys.path.insert(0, str(_HERE.parent / "04_training"))

try:
    from shapely.geometry import Polygon, Point, MultiPolygon
    from shapely.ops import unary_union
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:7]
# ════════════════════════════════════════════════════════════════

def generate_sensor_grid(site_pts: list,
                          buildings: list,
                          resolution: float = 2.0) -> np.ndarray:
    """
    [REMOVED_ZH:4]（[REMOVED_ZH:8]）[REMOVED_ZH:1] resolution [REMOVED_ZH:7]。

    Returns
    -------
    pts : (N, 2) float32
    """
    if _SHAPELY:
        return _grid_shapely(site_pts, buildings, resolution)
    else:
        return _grid_fallback(site_pts, buildings, resolution)


def _grid_shapely(site_pts, buildings, resolution):
    site_poly  = Polygon(site_pts)
    bldg_polys = []
    for b in buildings:
        try:
            poly = Polygon(b["footprint"])
            if poly.is_valid and poly.area > 0:
                bldg_polys.append(poly.buffer(0.1))  # slight inset avoid edge
        except Exception:
            pass

    occupied = unary_union(bldg_polys) if bldg_polys else None
    minx, miny, maxx, maxy = site_poly.bounds

    xs = np.arange(minx + resolution/2, maxx, resolution)
    ys = np.arange(miny + resolution/2, maxy, resolution)

    pts = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if site_poly.contains(pt):
                if occupied is None or not occupied.contains(pt):
                    pts.append([x, y])

    if len(pts) == 0:
        pts = [[minx + (maxx-minx)*0.5, miny + (maxy-miny)*0.5]]

    return np.array(pts, dtype=np.float32)


def _grid_fallback(site_pts, buildings, resolution):
    """Shapely [REMOVED_ZH:9]（[REMOVED_ZH:8]）"""
    pts_arr = np.array(site_pts)
    minx, miny = pts_arr.min(axis=0)
    maxx, maxy = pts_arr.max(axis=0)

    xs = np.arange(minx + resolution/2, maxx, resolution)
    ys = np.arange(miny + resolution/2, maxy, resolution)

    # [REMOVED_ZH:2] AABB
    bldg_aabb = []
    for b in buildings:
        fp = np.array(b["footprint"])
        bldg_aabb.append((fp[:,0].min(), fp[:,1].min(),
                          fp[:,0].max(), fp[:,1].max()))

    pts = []
    for x in xs:
        for y in ys:
            if not _point_in_polygon(x, y, site_pts):
                continue
            in_bldg = False
            for (bx0, by0, bx1, by1) in bldg_aabb:
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    in_bldg = True
                    break
            if not in_bldg:
                pts.append([x, y])

    return np.array(pts, dtype=np.float32) if pts else \
           np.array([[(minx+maxx)/2, (miny+maxy)/2]], dtype=np.float32)


def _point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon test"""
    n, inside = len(poly), False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py) and
                px < (xj-xi)*(py-yi)/(yj-yi+1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:4]compute
# ════════════════════════════════════════════════════════════════

def compute_svf(sensor_pts: np.ndarray,
                buildings:  list,
                n_dirs:     int = 16) -> np.ndarray:
    """
    [REMOVED_ZH:8] (SVF)。

    [REMOVED_ZH:6]，[REMOVED_ZH:1] n_dirs [REMOVED_ZH:3]compute[REMOVED_ZH:7]，
    SVF = 1 - mean(sin(max_elevation_per_direction))

    Returns
    -------
    svf : (N,) float32, values in [0.05, 1.0]
    """
    N   = len(sensor_pts)
    svf = np.ones(N, dtype=np.float32)
    if not buildings:
        return svf

    angles = np.linspace(0, 2*np.pi, n_dirs, endpoint=False)

    for b in buildings:
        h  = float(b.get("height", 10.0))
        fp = np.array(b["footprint"], dtype=float)
        cx, cy = fp.mean(axis=0)
        # [REMOVED_ZH:2]「[REMOVED_ZH:2]」（[REMOVED_ZH:5]）
        r_bldg = np.sqrt(((fp - fp.mean(axis=0))**2).sum(axis=1)).max() + 0.5

        for i, pt in enumerate(sensor_pts):
            dx, dy = cx - pt[0], cy - pt[1]
            dist   = math.hypot(dx, dy)
            if dist < 0.5:
                svf[i] = 0.1
                continue

            bldg_dir    = math.atan2(dy, dx)
            half_angle  = math.atan2(r_bldg, dist)
            max_el      = math.atan2(h, max(dist - r_bldg, 0.5))

            blocked_frac = 0.0
            for angle in angles:
                diff = abs(((angle - bldg_dir + math.pi) % (2*math.pi)) - math.pi)
                if diff < half_angle:
                    blocked_frac += math.sin(max_el)

            svf[i] -= blocked_frac / n_dirs

    return np.clip(svf, 0.05, 1.0).astype(np.float32)


def compute_in_shadow(sensor_pts: np.ndarray,
                       buildings:  list,
                       sol_alt_deg: float,
                       sol_az_deg:  float) -> np.ndarray:
    """
    [REMOVED_ZH:2]Shadowcompute：[REMOVED_ZH:12]Shadow。

    Returns
    -------
    in_shadow : (N,) float32, 0.0 [REMOVED_ZH:1] 1.0
    """
    N         = len(sensor_pts)
    in_shadow = np.zeros(N, dtype=np.float32)

    if sol_alt_deg < 3.0 or not buildings:
        return in_shadow

    sol_alt = math.radians(sol_alt_deg)
    sol_az  = math.radians(sol_az_deg)

    # [REMOVED_ZH:5]（Shadow[REMOVED_ZH:4]）
    shade_dx = -math.cos(sol_alt) * math.sin(sol_az)
    shade_dy = -math.cos(sol_alt) * math.cos(sol_az)

    for b in buildings:
        h  = float(b.get("height", 10.0))
        fp = np.array(b["footprint"], dtype=float)
        cx, cy = fp.mean(axis=0)
        r_bldg  = np.sqrt(((fp - fp.mean(axis=0))**2).sum(axis=1)).max()

        # Shadow[REMOVED_ZH:2]
        shadow_len = h / math.tan(sol_alt)

        # Shadow[REMOVED_ZH:2]
        tip_x = cx + shade_dx * shadow_len
        tip_y = cy + shade_dy * shadow_len

        for i, pt in enumerate(sensor_pts):
            # [REMOVED_ZH:2]：[REMOVED_ZH:4] → [REMOVED_ZH:3]
            vx, vy = pt[0] - cx, pt[1] - cy
            # [REMOVED_ZH:3]Shadow[REMOVED_ZH:2]
            proj = vx * shade_dx + vy * shade_dy
            if proj <= 0 or proj > shadow_len:
                continue
            # [REMOVED_ZH:4]
            lat = math.hypot(vx - proj*shade_dx, vy - proj*shade_dy)
            # Shadow[REMOVED_ZH:12]
            width = r_bldg * (1.0 - proj / (shadow_len + 1e-6)) + 0.5
            if lat < width:
                in_shadow[i] = 1.0

    return in_shadow


def compute_nearest_building_height(sensor_pts: np.ndarray,
                                     buildings:  list) -> np.ndarray:
    """[REMOVED_ZH:12] (m)"""
    N   = len(sensor_pts)
    bh  = np.zeros(N, dtype=np.float32)
    if not buildings:
        return bh

    for i, pt in enumerate(sensor_pts):
        min_d, nearest_h = np.inf, 0.0
        for b in buildings:
            fp = np.array(b["footprint"], dtype=float)
            cx, cy = fp.mean(axis=0)
            d = math.hypot(pt[0]-cx, pt[1]-cy)
            if d < min_d:
                min_d, nearest_h = d, float(b.get("height", 0))
        bh[i] = nearest_h

    return bh


def compute_nearest_tree_height(sensor_pts: np.ndarray,
                                 trees:      list) -> np.ndarray:
    """[REMOVED_ZH:12] (m)；[REMOVED_ZH:6] 0"""
    N  = len(sensor_pts)
    th = np.zeros(N, dtype=np.float32)
    if not trees:
        return th

    for i, pt in enumerate(sensor_pts):
        min_d, nearest_h = np.inf, 0.0
        for t in trees:
            d = math.hypot(pt[0]-t["x"], pt[1]-t["y"])
            if d < min_d:
                min_d, nearest_h = d, float(t.get("height", 0))
        th[i] = nearest_h

    return th


def estimate_mrt(ta_vals: np.ndarray,
                  ghi:       float,
                  svf:       np.ndarray,
                  in_shadow: np.ndarray,
                  sol_alt_deg: float) -> np.ndarray:
    """
    [REMOVED_ZH:8] (MRT) [REMOVED_ZH:2]。
    Tmrt ≈ Ta + ΔT_direct + ΔT_diffuse

    MRT [REMOVED_ZH:7]，[REMOVED_ZH:6]。
    """
    sol_alt = math.radians(max(sol_alt_deg, 5.0))
    # [REMOVED_ZH:4]
    I_direct   = ghi * math.sin(sol_alt) * (1.0 - in_shadow)
    # [REMOVED_ZH:6]（and SVF [REMOVED_ZH:3]）
    I_diffuse  = ghi * 0.15 * svf
    # [REMOVED_ZH:6]
    I_reflect  = ghi * 0.10 * (1 - svf)

    delta_mrt  = (I_direct + I_diffuse + I_reflect) / 100.0  # ~°C
    return (ta_vals + delta_mrt).astype(np.float32)


# ════════════════════════════════════════════════════════════════
# 3. GNN [REMOVED_ZH:5]
# ════════════════════════════════════════════════════════════════

class GNNInputBuilder:
    """
    [REMOVED_ZH:4]：
        builder = GNNInputBuilder(norm_stats, epw_data, dim_air=9)
        result  = builder.build(payload_dict)

    result keys:
        sensor_pts   : (N, 2)
        obj_feat     : (N_obj, 7)
        air_feat     : (N_air, T, dim_air)
        static_edges : {"contiguity": (2,E), "semantic": (2,E2)}
    """

    SIM_HOURS = list(range(8, 19))   # 8:00 – 18:00, T=11

    def __init__(self, norm_stats: dict, epw_data, dim_air: int = 8):
        self.norm_stats = norm_stats
        self.epw_data   = epw_data
        self.dim_air    = dim_air

        # [REMOVED_ZH:9]（[REMOVED_ZH:5]）
        self._clim_map = self._build_clim_map()

    def _build_clim_map(self) -> dict:
        typical = self.epw_data.get_typical_day(month=7, stat="hottest")
        return {h.hour: h for h in typical}

    def build(self, payload: dict) -> dict | None:
        site_pts       = payload["site_boundary"]
        buildings      = payload.get("buildings", [])
        trees          = payload.get("trees", [])
        resolution     = float(payload.get("sensor_resolution", 2.0))
        material_zones = payload.get("material_zones", None)

        # 1. [REMOVED_ZH:5]
        sensor_pts = generate_sensor_grid(site_pts, buildings, resolution)
        N_air = len(sensor_pts)
        T     = len(self.SIM_HOURS)

        if N_air < 3:
            return None

        # 2. [REMOVED_ZH:6] (N_obj, 7)
        obj_feat = self._build_obj_feat(buildings, site_pts)
        N_obj    = obj_feat.shape[0]

        # 3. [REMOVED_ZH:6] (N_air, T, dim_air)
        air_feat = self._build_air_feat(sensor_pts, buildings, trees, T,
                                         material_zones=material_zones)

        # 4. Static Edges
        static_edges = self._build_static_edges(sensor_pts, N_obj, N_air, N_obj)

        return {
            "sensor_pts":   sensor_pts,
            "obj_feat":     obj_feat,
            "air_feat":     air_feat,
            "static_edges": static_edges,
        }

    # ── [REMOVED_ZH:4] ──────────────────────────────────────────────

    def _build_obj_feat(self, buildings: list, site_pts: list) -> np.ndarray:
        site_arr = np.array(site_pts)
        site_cx  = site_arr[:, 0].mean()
        site_cy  = site_arr[:, 1].mean()

        if not buildings:
            return np.zeros((1, 7), dtype=np.float32)

        rows = []
        for b in buildings:
            h      = float(b.get("height",      10.0))
            fl     = int(b.get("floor_count",    round(h / 4.5)))
            fp     = np.array(b["footprint"])
            area   = abs(_shoelace(fp))
            cx, cy = fp.mean(axis=0)
            gfa    = area * fl
            # is_L_shape：[REMOVED_ZH:3] > 4 → [REMOVED_ZH:3]
            is_L   = 1.0 if len(fp) > 4 else 0.0

            rows.append([
                h        / 50.0,    # 0: height
                fl       / 12.0,    # 1: floors
                area     / 2000.0,  # 2: footprint area
                (cx - site_cx) / 80.0,  # 3: centroid_x (site-relative)
                (cy - site_cy) / 80.0,  # 4: centroid_y
                gfa      / 20000.0, # 5: GFA
                is_L,               # 6: is_L_shape
            ])

        return np.array(rows, dtype=np.float32)

    # ── [REMOVED_ZH:6] ─────────────────────────────────────────

    def _build_air_feat(self, sensor_pts, buildings, trees, T,
                         material_zones=None) -> np.ndarray:
        N  = len(sensor_pts)
        ns = self.norm_stats
        dim = self.dim_air

        # static[REMOVED_ZH:4]（[REMOVED_ZH:6]）
        svf = compute_svf(sensor_pts, buildings)
        bh  = compute_nearest_building_height(sensor_pts, buildings)
        th  = compute_nearest_tree_height(sensor_pts, trees)
        bh_n = (bh / 50.0).astype(np.float32)
        th_n = (th / 12.0).astype(np.float32)

        # Resolve per-sensor material types from zones (for dim_air >= 9)
        sensor_materials = None
        if dim >= 9 and material_zones:
            try:
                from shared.surface_materials import assign_materials_to_sensors
                sensor_materials = assign_materials_to_sensors(
                    sensor_pts.tolist(), material_zones)
            except Exception:
                pass

        feat = np.zeros((N, T, dim), dtype=np.float32)

        for t_idx, hr in enumerate(self.SIM_HOURS):
            clim = self._clim_map.get(hr)
            if clim is None:
                continue

            # [REMOVED_ZH:4]（[REMOVED_ZH:1] shared.solar_position）
            try:
                from shared import solar_position
                sol_alt, sol_az = solar_position(
                    self.epw_data.latitude,
                    self.epw_data.longitude,
                    self.epw_data.timezone,
                    7, 15, hr)
            except Exception:
                sol_alt = max(0.0, 90.0 - abs(hr - 12) * 15)
                sol_az  = 180.0

            # Shadow（[REMOVED_ZH:2]）
            in_shadow = compute_in_shadow(sensor_pts, buildings,
                                          sol_alt, sol_az)

            # [REMOVED_ZH:3]
            ta  = np.full(N, clim.ta,         dtype=np.float32)
            rh  = np.full(N, clim.rh,         dtype=np.float32)
            va  = np.full(N, clim.wind_speed,  dtype=np.float32)
            ghi = float(clim.ghi)

            # [REMOVED_ZH:2]Wind Speed[REMOVED_ZH:2]（[REMOVED_ZH:4]）
            va *= (0.5 + 0.5 * svf)  # [REMOVED_ZH:9]

            # MRT [REMOVED_ZH:2]
            mrt = estimate_mrt(ta, ghi, svf, in_shadow, sol_alt)

            # [REMOVED_ZH:3]
            ta_n  = _norm(ta,  ns, "ta")
            mrt_n = _norm(mrt, ns, "mrt")
            va_n  = _norm(va,  ns, "va")
            rh_n  = _norm(rh,  ns, "rh")

            feat[:, t_idx, 0] = ta_n
            feat[:, t_idx, 1] = mrt_n
            feat[:, t_idx, 2] = va_n
            feat[:, t_idx, 3] = rh_n
            feat[:, t_idx, 4] = svf           # not normalized, [0,1]
            feat[:, t_idx, 5] = in_shadow     # 0/1
            feat[:, t_idx, 6] = bh_n
            feat[:, t_idx, 7] = th_n

            # ── V2: surface temperature (9th feature) ─────────────
            if dim >= 9:
                rh_frac = clim.rh / 100.0  # EPW stores 0-100

                try:
                    from shared.surface_materials import compute_surface_temp_scalar_batch

                    if sensor_materials:
                        # Per-sensor material: group by material, compute batch
                        ts_raw = np.zeros(N, dtype=np.float32)
                        # Collect unique materials
                        mat_groups: dict = {}
                        for i, pt in enumerate(sensor_pts):
                            mat = sensor_materials.get(
                                (float(pt[0]), float(pt[1])), "concrete")
                            mat_groups.setdefault(mat, []).append(i)

                        for mat_name, indices in mat_groups.items():
                            idx = np.array(indices)
                            ts_raw[idx] = compute_surface_temp_scalar_batch(
                                mat_name, ta[idx], ghi, va[idx], rh_frac)
                    else:
                        # Default: all concrete
                        ts_raw = compute_surface_temp_scalar_batch(
                            "concrete", ta, ghi, va, rh_frac)

                except ImportError:
                    # Graceful fallback: surface temp ≈ air temp + solar offset
                    ts_raw = ta + ghi * (1.0 - 0.35) / (5.7 + 3.8 * va + 5.0)

                # Normalize using ts stats or fallback
                ts_stats = ns.get("ts", {"mean": ns["ta"]["mean"] + 5.0,
                                          "std": ns["ta"]["std"] * 1.2})
                ts_n = ((ts_raw - ts_stats["mean"]) /
                        (ts_stats["std"] + 1e-8)).astype(np.float32)
                feat[:, t_idx, 8] = ts_n

        return feat

    # ── Static Edges ───────────────────────────────────────────────

    def _build_static_edges(self, sensor_pts: np.ndarray,
                              N_obj: int, N_air: int,
                              n_obj: int) -> dict:
        edges = {}

        # Contiguity (KNN air-air)，[REMOVED_ZH:1] N_obj [REMOVED_ZH:2]
        knn_ei = _knn_edges(sensor_pts, k=8) + N_obj
        edges["contiguity"] = knn_ei.astype(np.int64)

        # Semantic (fully-connected obj-obj)
        if N_obj >= 2:
            src, dst = [], []
            for i in range(N_obj):
                for j in range(N_obj):
                    if i != j:
                        src.append(i); dst.append(j)
            edges["semantic"] = np.array([src, dst], dtype=np.int64)

        return edges


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════

def _norm(arr: np.ndarray, ns: dict, key: str) -> np.ndarray:
    mu  = ns[key]["mean"]
    std = ns[key]["std"]
    return ((arr - mu) / (std + 1e-8)).astype(np.float32)


def _shoelace(pts: np.ndarray) -> float:
    xs, ys = pts[:, 0], pts[:, 1]
    return 0.5 * (np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _knn_edges(pts: np.ndarray, k: int = 8) -> np.ndarray:
    """Build KNN [REMOVED_ZH:3] (2, E)，0-indexed"""
    try:
        from scipy.spatial import cKDTree
        k_eff  = min(k, len(pts) - 1)
        tree   = cKDTree(pts)
        _, idx = tree.query(pts, k=k_eff + 1)
        src, dst = [], []
        for i, nbrs in enumerate(idx):
            for j in nbrs[1:]:
                src.append(i); dst.append(int(j))
        return np.array([src, dst], dtype=np.int64)
    except ImportError:
        # [REMOVED_ZH:3] fallback
        N  = len(pts)
        kk = min(k, N - 1)
        src, dst = [], []
        for i in range(N):
            dists = np.linalg.norm(pts - pts[i], axis=1)
            dists[i] = np.inf
            nbrs = np.argsort(dists)[:kk]
            for j in nbrs:
                src.append(i); dst.append(int(j))
        return np.array([src, dst], dtype=np.int64)
