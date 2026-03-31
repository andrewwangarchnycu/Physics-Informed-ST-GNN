"""
02_geometry_sampler.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:2] : 01_data_generation/scripts/
[REMOVED_ZH:2] : [REMOVED_ZH:2] site_constraints.yaml，[REMOVED_ZH:11] (FAR)
       and[REMOVED_ZH:3] (BCR) [REMOVED_ZH:2]，[REMOVED_ZH:18]。
       [REMOVED_ZH:7] pickle，[REMOVED_ZH:1] 03_lbt_batch_runner.py [REMOVED_ZH:2]。

[REMOVED_ZH:4]:
  {
    "scenario_id": int,
    "site_polygon": Shapely Polygon,
    "buildings": [
        {
          "footprint": Shapely Polygon,
          "floors":    int,
          "height":    float,
          "centroid":  (x, y),
          "gfa":       float,
          "coverage":  float,
        }, ...
    ],
    "open_space":  Shapely Geometry,
    "far_actual":  float,
    "bcr_actual":  float,
    "total_gfa":   float,
  }

Run :
  python 02_geometry_sampler.py
  python 02_geometry_sampler.py --config ../config/site_constraints.yaml \
                                 --n 500 --out ../outputs/raw_simulations
"""

import math
import random
import pickle
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import yaml
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from shapely.affinity import rotate
from shapely.validation import make_valid


# ════════════════════════════════════════════════════════════════
# 1. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def site_polygon_from_cfg(cfg: Dict) -> Polygon:
    """[REMOVED_ZH:1] config Build[REMOVED_ZH:7]。"""
    s  = cfg["site"]
    ox, oy = s.get("origin", [0.0, 0.0])
    w, d   = float(s["width"]), float(s["depth"])
    return box(ox, oy, ox + w, oy + d)


# ════════════════════════════════════════════════════════════════
# 2. [REMOVED_ZH:8]
# ════════════════════════════════════════════════════════════════
def _make_rect(cx: float, cy: float, w: float, d: float,
               angle: float = 0.0) -> Polygon:
    r = box(cx - w/2, cy - d/2, cx + w/2, cy + d/2)
    return rotate(r, angle, origin=(cx, cy)) if angle else r


def _make_lshape(cx: float, cy: float, w: float, d: float,
                 angle: float = 0.0, rng: random.Random = None) -> Polygon:
    """L [REMOVED_ZH:1]：[REMOVED_ZH:7]，[REMOVED_ZH:10]。"""
    rng = rng or random
    big   = box(cx - w/2, cy - d/2, cx + w/2, cy + d/2)
    cw    = w * rng.uniform(0.30, 0.50)
    cd    = d * rng.uniform(0.30, 0.50)
    notch = box(cx + w/2 - cw, cy + d/2 - cd,
                cx + w/2 + 1,  cy + d/2 + 1)
    shape = make_valid(big.difference(notch))
    return rotate(shape, angle, origin=(cx, cy)) if angle else shape


def make_footprint(cx: float, cy: float, w: float, d: float,
                   shape: str = "rect", angle: float = 0.0,
                   rng: random.Random = None) -> Polygon:
    if shape == "L":
        return _make_lshape(cx, cy, w, d, angle, rng)
    return _make_rect(cx, cy, w, d, angle)


# ════════════════════════════════════════════════════════════════
# 3. [REMOVED_ZH:5]
# ════════════════════════════════════════════════════════════════
class GeometrySampler:
    """
    [REMOVED_ZH:8] FAR / BCR [REMOVED_ZH:10]。

    Parameters
    ----------
    site_polygon       : Shapely Polygon
    far_target         : [REMOVED_ZH:5]
    bcr_target         : [REMOVED_ZH:5]
    far_tolerance      : [REMOVED_ZH:6] ([REMOVED_ZH:2] ±2%)
    floors_range       : (min_floors, max_floors)
    n_buildings_range  : (min_n, max_n)
    min_footprint      : [REMOVED_ZH:8] [m²]
    floor_height       : [REMOVED_ZH:4] [m]
    setback_min        : [REMOVED_ZH:4] [m]
    allow_lshape       : [REMOVED_ZH:2] L [REMOVED_ZH:3]
    seed               : [REMOVED_ZH:4]
    """

    def __init__(self,
                 site_polygon:       Polygon,
                 far_target:         float = 2.5,
                 bcr_target:         float = 0.60,
                 far_tolerance:      float = 0.02,
                 floors_range:       Tuple[int,int] = (3, 12),
                 n_buildings_range:  Tuple[int,int] = (2, 5),
                 min_footprint:      float = 80.0,
                 floor_height:       float = 3.6,
                 setback_min:        float = 3.0,
                 allow_lshape:       bool  = True,
                 seed:               int   = 42):

        self.site        = make_valid(site_polygon)
        self.site_area   = self.site.area
        self.far         = far_target
        self.bcr         = bcr_target
        self.far_tol     = far_tolerance
        self.floors_rng  = floors_range
        self.n_bldg_rng  = n_buildings_range
        self.min_fp      = min_footprint
        self.fh          = floor_height
        self.allow_l     = allow_lshape
        self.rng         = random.Random(seed)
        np.random.seed(seed)

        self.bz = self.site.buffer(-setback_min)
        if self.bz.is_empty:
            raise ValueError("[REMOVED_ZH:7]，[REMOVED_ZH:3] setback_min [REMOVED_ZH:5]。")

        self.target_coverage = bcr_target * self.site_area
        self.target_gfa      = far_target * self.site_area

    # ── [REMOVED_ZH:6] ─────────────────────────────
    def _try_place(self,
                   placed:      List[Polygon],
                   rem_cov:     float,
                   rem_gfa:     float) -> Optional[Dict]:
        if rem_cov < self.min_fp:
            return None

        bnd = self.bz.bounds   # (minx, miny, maxx, maxy)

        for _ in range(50):
            # [REMOVED_ZH:3]
            max_f = max(self.floors_rng[0],
                        min(self.floors_rng[1],
                            int(rem_gfa / max(self.min_fp, 1))))
            floors = self.rng.randint(self.floors_rng[0], max_f)

            # [REMOVED_ZH:2] footprint [REMOVED_ZH:2]
            max_fp = min(rem_cov,
                         rem_gfa / floors,
                         self.bz.area * 0.6)
            if max_fp < self.min_fp:
                continue

            fp_area = self.rng.uniform(self.min_fp, min(max_fp, rem_cov * 0.8))
            aspect  = self.rng.uniform(1.0, 3.0)
            w = math.sqrt(fp_area * aspect)
            d = math.sqrt(fp_area / aspect)

            cx = self.rng.uniform(bnd[0]+w/2, bnd[2]-w/2)
            cy = self.rng.uniform(bnd[1]+d/2, bnd[3]-d/2)
            ang = self.rng.uniform(-45, 45)

            shape = ("L" if self.allow_l
                         and self.rng.random() < 0.30
                         and fp_area > 200
                         else "rect")
            fp = make_footprint(cx, cy, w, d, shape, ang, self.rng)

            if not fp.is_valid:
                fp = make_valid(fp)
            if fp.area < self.min_fp:
                continue
            if not self.bz.contains(fp):
                fp = fp.intersection(self.bz)
                if fp.is_empty or fp.area < self.min_fp:
                    continue
            if any(fp.intersects(p) for p in placed):
                continue

            return {
                "footprint": fp,
                "floors":    floors,
                "height":    floors * self.fh,
                "centroid":  (fp.centroid.x, fp.centroid.y),
                "gfa":       fp.area * floors,
                "coverage":  fp.area,
                "shape_type": shape,
            }
        return None

    # ── [REMOVED_ZH:6] ─────────────────────────────
    def sample_scenario(self, scenario_id: int) -> Optional[Dict[str, Any]]:
        n_bldg = self.rng.randint(*self.n_bldg_rng)
        placed: List[Polygon] = []
        buildings: List[Dict] = []
        rem_cov = self.target_coverage
        rem_gfa = self.target_gfa

        for _ in range(n_bldg):
            b = self._try_place(placed, rem_cov, rem_gfa)
            if b is None:
                break
            placed.append(b["footprint"])
            buildings.append(b)
            rem_cov -= b["coverage"]
            rem_gfa -= b["gfa"]
            if rem_cov < self.min_fp:
                break

        if not buildings:
            return None

        total_gfa  = sum(b["gfa"]      for b in buildings)
        total_cov  = sum(b["coverage"] for b in buildings)
        far_actual = total_gfa  / self.site_area
        bcr_actual = total_cov  / self.site_area
        open_space = self.site.difference(unary_union(placed))
        # [REMOVED_ZH:1] sample_scenario() return [REMOVED_ZH:1]，open_space [REMOVED_ZH:5]
        import random as _rnd
        _rng_tree = _rnd.Random(scenario_id + 9999)
        trees = []
        # [REMOVED_ZH:8] 2–5 [REMOVED_ZH:2]
        n_trees = _rng_tree.randint(2, 5)
        os_bounds = open_space.bounds  # (minx, miny, maxx, maxy)
        for _ in range(n_trees * 5):   # [REMOVED_ZH:4]
            if len(trees) >= n_trees:
                break
            tx = _rng_tree.uniform(os_bounds[0], os_bounds[2])
            ty = _rng_tree.uniform(os_bounds[1], os_bounds[3])
            tp = Point(tx, ty)
            if open_space.contains(tp):
                h  = _rng_tree.uniform(4.0, 12.0)
                trees.append({"pos": (tx, ty), "height": h,
                            "radius": h * 0.4})

        return {
            "scenario_id":  scenario_id,
            "site_polygon": self.site,
            "buildings":    buildings,
            "open_space":   open_space,
            "trees":        trees,          # ← [REMOVED_ZH:2]
            "land_cover_map": None,         # ← [REMOVED_ZH:2] None，[REMOVED_ZH:3] GIS
            "far_actual":   far_actual,
            "bcr_actual":   bcr_actual,
            "total_gfa":    total_gfa,
        }

        return {
            "scenario_id":  scenario_id,
            "site_polygon": self.site,
            "buildings":    buildings,
            "open_space":   open_space,
            "far_actual":   far_actual,
            "bcr_actual":   bcr_actual,
            "total_gfa":    total_gfa,
        }

    # ── [REMOVED_ZH:4] ─────────────────────────────────
    def generate_batch(self,
                        n: int,
                        verbose: bool = True) -> List[Dict[str, Any]]:
        results:  List[Dict] = []
        attempts: int        = 0
        max_att = n * 12

        while len(results) < n and attempts < max_att:
            sc = self.sample_scenario(len(results))
            attempts += 1
            if sc is None:
                continue

            far_ok = abs(sc["far_actual"] - self.far) / self.far <= self.far_tol
            bcr_ok = sc["bcr_actual"] <= self.bcr * (1 + self.far_tol)

            if far_ok and bcr_ok:
                results.append(sc)
                if verbose and len(results) % 100 == 0:
                    print(f"    {len(results)}/{n}  ([REMOVED_ZH:2] {attempts})")

        sr = len(results) / max(attempts, 1) * 100
        if verbose:
            print(f"  [Sampler] ✓ {len(results)} [REMOVED_ZH:2]  [REMOVED_ZH:3] {sr:.1f}%  "
                  f"([REMOVED_ZH:2] {attempts})")
        return results


# ════════════════════════════════════════════════════════════════
# 4. [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════
def save_scenarios(scenarios: List[Dict], out_dir: Path) -> Path:
    """Sequence[REMOVED_ZH:6] scenarios.pkl。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "scenarios.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"  [Sampler] ✓ [REMOVED_ZH:6]: {pkl}")
    return pkl


def print_summary(scenarios: List[Dict]) -> None:
    if not scenarios:
        return
    n_bldg  = [len(s["buildings"]) for s in scenarios]
    far_arr = [s["far_actual"]     for s in scenarios]
    bcr_arr = [s["bcr_actual"]     for s in scenarios]
    open_arr= [s["open_space"].area for s in scenarios]
    print(f"\n  ── [REMOVED_ZH:4] ──────────────────────────────")
    print(f"  [REMOVED_ZH:3]:      {len(scenarios)}")
    print(f"  [REMOVED_ZH:2]:        {min(n_bldg)}–{max(n_bldg)}  "
          f"([REMOVED_ZH:2] {np.mean(n_bldg):.1f})")
    print(f"  FAR:         {np.min(far_arr):.2f}–{np.max(far_arr):.2f}  "
          f"(μ={np.mean(far_arr):.2f})")
    print(f"  BCR:         {np.min(bcr_arr):.2f}–{np.max(bcr_arr):.2f}  "
          f"(μ={np.mean(bcr_arr):.2f})")
    print(f"  [REMOVED_ZH:4]:    {np.min(open_arr):.0f}–{np.max(open_arr):.0f} m²  "
          f"(μ={np.mean(open_arr):.0f})")
    print(f"  ──────────────────────────────────────────\n")


# ════════════════════════════════════════════════════════════════
# 5. Main Program
# ════════════════════════════════════════════════════════════════
def main(config_path: str = "../config/site_constraints.yaml",
         n_override:  int  = 0,
         out_dir_override: str = ""):

    print("\n[02_geometry_sampler] ── [REMOVED_ZH:8] ──")
    cfg = load_config(config_path)

    site  = site_polygon_from_cfg(cfg)
    reg   = cfg["regulations"]
    samp  = cfg["sampling"]

    n_scenarios = n_override or samp["n_scenarios"]
    out_dir = Path(out_dir_override or "../outputs/raw_simulations")

    sampler = GeometrySampler(
        site_polygon      = site,
        far_target        = float(reg["far"]),
        bcr_target        = float(reg["bcr"]),
        far_tolerance     = float(reg.get("far_tolerance", 0.02)),
        floors_range      = (int(reg["floors_min"]), int(reg["floors_max"])),
        n_buildings_range = (int(samp["n_buildings_min"]),
                              int(samp["n_buildings_max"])),
        min_footprint     = float(reg.get("min_footprint", 80.0)),
        floor_height      = float(reg.get("floor_height", 3.6)),
        setback_min       = float(reg.get("setback_min", 3.0)),
        allow_lshape      = bool(samp.get("allow_lshape", True)),
        seed              = int(samp.get("random_seed", 42)),
    )

    scenarios = sampler.generate_batch(n=n_scenarios, verbose=True)
    print_summary(scenarios)
    save_scenarios(scenarios, out_dir)

    print("[02_geometry_sampler] [REMOVED_ZH:2]。\n")
    return scenarios


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../config/site_constraints.yaml")
    ap.add_argument("--n",     type=int, default=0,
                    help="[REMOVED_ZH:2] config [REMOVED_ZH:2] n_scenarios")
    ap.add_argument("--out",   default="",
                    help="[REMOVED_ZH:6]")
    args = ap.parse_args()
    main(args.config, args.n, args.out)