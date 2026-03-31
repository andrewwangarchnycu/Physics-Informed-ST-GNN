"""
07_optimization/chromosome.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:5]：[REMOVED_ZH:2] / [REMOVED_ZH:2] [REMOVED_ZH:2] + [REMOVED_ZH:2] [REMOVED_ZH:2]

[REMOVED_ZH:3] = flat float array，[REMOVED_ZH:7] [0, 1]
  [b0_cx, b0_cy, b0_w, b0_d, b0_rot, b0_floors,   ← building 0
   b1_cx, …                                         ← building 1
   t0_x,  t0_y, t0_r, t0_h,                        ← tree 0
   t1_x,  …                                         ← tree 1 ]

[REMOVED_ZH:4]：
  SBX crossover (Deb & Agrawal 1995)
  Polynomial mutation (Deb 1996)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

GENES_PER_BUILDING = 6   # cx, cy, w, d, rot, floors
GENES_PER_TREE     = 4   # x, y, radius, height


# ── [REMOVED_ZH:8] ──────────────────────────────────────────

@dataclass
class BuildingGene:
    cx:     float          # footprint centroid x (m, [REMOVED_ZH:5])
    cy:     float
    w:      float          # footprint width  (m)
    d:      float          # footprint depth  (m)
    rot:    float          # rotation angle   (degrees, 0–180)
    floors: int            # number of floors

    @property
    def height(self, floor_h: float = 4.5) -> float:
        return self.floors * floor_h

    def footprint_polygon(self) -> np.ndarray:
        """[REMOVED_ZH:10] (4, 2)"""
        hw, hd = self.w / 2, self.d / 2
        corners = np.array([[-hw, -hd], [hw, -hd],
                             [hw,  hd], [-hw,  hd]], dtype=float)
        theta = math.radians(self.rot)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (corners @ R.T) + np.array([self.cx, self.cy])


@dataclass
class TreeGene:
    x:      float
    y:      float
    radius: float
    height: float


@dataclass
class Design:
    buildings: List[BuildingGene] = field(default_factory=list)
    trees:     List[TreeGene]     = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "buildings": [
                {"cx": b.cx, "cy": b.cy, "w": b.w, "d": b.d,
                 "rot": b.rot, "floors": b.floors}
                for b in self.buildings
            ],
            "trees": [
                {"x": t.x, "y": t.y, "radius": t.radius, "height": t.height}
                for t in self.trees
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Design":
        bldgs = [BuildingGene(**b) for b in d.get("buildings", [])]
        trees = [TreeGene(**t)     for t in d.get("trees", [])]
        for b in bldgs:
            b.floors = int(round(b.floors))
        return Design(buildings=bldgs, trees=trees)


# ── [REMOVED_ZH:6] ──────────────────────────────────────────────

@dataclass
class ChromosomeConfig:
    """
    [REMOVED_ZH:8]。[REMOVED_ZH:1] Grasshopper [REMOVED_ZH:7]Build[REMOVED_ZH:1]
    [REMOVED_ZH:2] JSON [REMOVED_ZH:2] FastAPI [REMOVED_ZH:3]。
    """
    site_bbox:    Tuple[float, float, float, float] = (0.0, 0.0, 80.0, 80.0)
    n_buildings:  int   = 2
    n_trees:      int   = 4
    floor_range:  Tuple[int, int]     = (2, 8)
    bldg_w_range: Tuple[float, float] = (8.0, 30.0)
    bldg_d_range: Tuple[float, float] = (8.0, 30.0)
    tree_r_range: Tuple[float, float] = (1.5, 5.0)
    tree_h_range: Tuple[float, float] = (3.0, 10.0)
    floor_height: float = 4.5         # m/floor

    @property
    def n_genes(self) -> int:
        return (self.n_buildings * GENES_PER_BUILDING
                + self.n_trees     * GENES_PER_TREE)

    @property
    def site_w(self) -> float:
        return self.site_bbox[2] - self.site_bbox[0]

    @property
    def site_h(self) -> float:
        return self.site_bbox[3] - self.site_bbox[1]

    def to_dict(self) -> dict:
        return {
            "site_bbox":    list(self.site_bbox),
            "n_buildings":  self.n_buildings,
            "n_trees":      self.n_trees,
            "floor_range":  list(self.floor_range),
            "bldg_w_range": list(self.bldg_w_range),
            "bldg_d_range": list(self.bldg_d_range),
            "tree_r_range": list(self.tree_r_range),
            "tree_h_range": list(self.tree_h_range),
            "floor_height": self.floor_height,
        }

    @staticmethod
    def from_dict(d: dict) -> "ChromosomeConfig":
        return ChromosomeConfig(
            site_bbox    = tuple(d["site_bbox"]),
            n_buildings  = d["n_buildings"],
            n_trees      = d["n_trees"],
            floor_range  = tuple(d["floor_range"]),
            bldg_w_range = tuple(d["bldg_w_range"]),
            bldg_d_range = tuple(d["bldg_d_range"]),
            tree_r_range = tuple(d["tree_r_range"]),
            tree_h_range = tuple(d["tree_h_range"]),
            floor_height = d.get("floor_height", 4.5),
        )


# ── [REMOVED_ZH:2] / [REMOVED_ZH:2] ───────────────────────────────────────────────

def decode(genes: np.ndarray, cfg: ChromosomeConfig) -> Design:
    """[0,1]^n → Design"""
    genes = np.clip(genes, 0.0, 1.0)
    xmin, ymin = cfg.site_bbox[0], cfg.site_bbox[1]
    sw, sh = cfg.site_w, cfg.site_h

    buildings: List[BuildingGene] = []
    trees:     List[TreeGene]     = []
    idx = 0

    for _ in range(cfg.n_buildings):
        g = genes[idx:idx + GENES_PER_BUILDING]; idx += GENES_PER_BUILDING
        w   = _lerp(g[2], *cfg.bldg_w_range)
        d   = _lerp(g[3], *cfg.bldg_d_range)
        mg  = max(w, d) / 2.0 + 0.5          # centroid margin
        cx  = xmin + _lerp(g[0], mg, sw - mg)
        cy  = ymin + _lerp(g[1], mg, sh - mg)
        rot = g[4] * 180.0
        fl  = int(round(_lerp(g[5], *cfg.floor_range)))
        fl  = int(np.clip(fl, *cfg.floor_range))
        buildings.append(BuildingGene(cx=cx, cy=cy, w=w, d=d, rot=rot, floors=fl))

    for _ in range(cfg.n_trees):
        g = genes[idx:idx + GENES_PER_TREE]; idx += GENES_PER_TREE
        r  = _lerp(g[2], *cfg.tree_r_range)
        tx = xmin + _lerp(g[0], r, sw - r)
        ty = ymin + _lerp(g[1], r, sh - r)
        th = _lerp(g[3], *cfg.tree_h_range)
        trees.append(TreeGene(x=tx, y=ty, radius=r, height=th))

    return Design(buildings=buildings, trees=trees)


def encode(design: Design, cfg: ChromosomeConfig) -> np.ndarray:
    """Design → [0,1]^n（[REMOVED_ZH:10]）"""
    genes = np.zeros(cfg.n_genes)
    xmin, ymin = cfg.site_bbox[0], cfg.site_bbox[1]
    sw, sh = cfg.site_w, cfg.site_h
    idx = 0

    for b in design.buildings:
        mg = max(b.w, b.d) / 2.0 + 0.5
        genes[idx+0] = _inv_lerp(b.cx - xmin, mg, sw - mg)
        genes[idx+1] = _inv_lerp(b.cy - ymin, mg, sh - mg)
        genes[idx+2] = _inv_lerp(b.w, *cfg.bldg_w_range)
        genes[idx+3] = _inv_lerp(b.d, *cfg.bldg_d_range)
        genes[idx+4] = b.rot / 180.0
        genes[idx+5] = _inv_lerp(b.floors, *cfg.floor_range)
        idx += GENES_PER_BUILDING

    for t in design.trees:
        genes[idx+0] = _inv_lerp(t.x - xmin, t.radius, sw - t.radius)
        genes[idx+1] = _inv_lerp(t.y - ymin, t.radius, sh - t.radius)
        genes[idx+2] = _inv_lerp(t.radius, *cfg.tree_r_range)
        genes[idx+3] = _inv_lerp(t.height, *cfg.tree_h_range)
        idx += GENES_PER_TREE

    return np.clip(genes, 0.0, 1.0)


# ── [REMOVED_ZH:4] ──────────────────────────────────────────────────

def sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                   eta: float = 20.0,
                   prob: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover"""
    if np.random.rand() > prob:
        return p1.copy(), p2.copy()

    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() < 0.5 or abs(p1[i] - p2[i]) < 1e-10:
            continue
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        dy = y2 - y1 + 1e-12
        u  = np.random.rand()

        for sign, ya in ((+1, y1), (-1, y2)):
            if sign == +1:
                beta = 1.0 + 2.0 * ya / dy
            else:
                beta = 1.0 + 2.0 * (1.0 - ya) / dy
            alpha = 2.0 - beta ** (-(eta + 1))
            if u <= 1.0 / alpha:
                bq = (u * alpha) ** (1.0 / (eta + 1))
            else:
                bq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))
            if sign == +1:
                c1[i] = np.clip(0.5 * ((y1 + y2) - bq * dy), 0, 1)
            else:
                c2[i] = np.clip(0.5 * ((y1 + y2) + bq * dy), 0, 1)

    return c1, c2


def polynomial_mutation(x: np.ndarray,
                         eta: float = 20.0,
                         prob: float | None = None) -> np.ndarray:
    """Polynomial Mutation"""
    n    = len(x)
    prob = prob if prob is not None else 1.0 / n
    y = x.copy()
    for i in range(n):
        if np.random.rand() < prob:
            u = np.random.rand()
            delta = ((2*u)**(1/(eta+1)) - 1.0) if u < 0.5 \
                else (1.0 - (2*(1-u))**(1/(eta+1)))
            y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y


def random_individual(cfg: ChromosomeConfig) -> np.ndarray:
    return np.random.rand(cfg.n_genes)


# ── [REMOVED_ZH:4] ───────────────────────────────────────────────────
def _lerp(t, lo, hi):
    return lo + t * (hi - lo)

def _inv_lerp(v, lo, hi):
    r = hi - lo
    return np.clip((v - lo) / r, 0.0, 1.0) if abs(r) > 1e-12 else 0.5
