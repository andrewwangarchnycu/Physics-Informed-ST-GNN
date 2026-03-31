"""
07_optimization/constraints.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:4]and[REMOVED_ZH:6]

[REMOVED_ZH:4]：
  C1  FAR   — [REMOVED_ZH:3] ≤ far_max
  C2  BCR   — [REMOVED_ZH:3] ≤ bcr_max
  C3  [REMOVED_ZH:2]  — [REMOVED_ZH:7] ([REMOVED_ZH:2] – setback) [REMOVED_ZH:3]
  C4  [REMOVED_ZH:2]  — [REMOVED_ZH:8]
  C5  [REMOVED_ZH:3] — [REMOVED_ZH:8]

[REMOVED_ZH:2]「[REMOVED_ZH:3]」(violation ≥ 0)，0 [REMOVED_ZH:6]。
NSGA-II [REMOVED_ZH:2] total_violation [REMOVED_ZH:9]。
"""
from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np

from chromosome import Design, BuildingGene


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:4]
# ════════════════════════════════════════════════════════════════

def _rect_polygon(b: BuildingGene) -> np.ndarray:
    """[REMOVED_ZH:7] 4 [REMOVED_ZH:3] (4, 2)"""
    hw, hd = b.w / 2.0, b.d / 2.0
    pts = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    theta = math.radians(b.rot)
    c, s  = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([b.cx, b.cy])


def _poly_area(pts: np.ndarray) -> float:
    """Shoelace [REMOVED_ZH:4]"""
    n = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _poly_contains_all(poly: np.ndarray, test_pts: np.ndarray) -> bool:
    """
    [REMOVED_ZH:2] test_pts [REMOVED_ZH:12] poly [REMOVED_ZH:1]。
    [REMOVED_ZH:7]（[REMOVED_ZH:2] poly [REMOVED_ZH:2]，[REMOVED_ZH:5]）。
    """
    n = len(poly)
    for pt in test_pts:
        for i in range(n):
            e = poly[(i+1) % n] - poly[i]
            v = pt - poly[i]
            if e[0]*v[1] - e[1]*v[0] < -1e-6:
                return False
    return True


def _polygon_inside_polygon_approx(inner: np.ndarray,
                                    outer: np.ndarray) -> float:
    """
    compute inner [REMOVED_ZH:8] outer [REMOVED_ZH:9]。
    [REMOVED_ZH:3] 0 → [REMOVED_ZH:5]。
    """
    violation = 0.0
    n_outer = len(outer)
    for pt in inner:
        # [REMOVED_ZH:7]
        max_pen = 0.0
        for i in range(n_outer):
            e  = outer[(i+1) % n_outer] - outer[i]
            v  = pt - outer[i]
            # cross product sign (+ = inside CCW polygon)
            cross = e[0]*v[1] - e[1]*v[0]
            edge_len = math.hypot(e[0], e[1]) + 1e-12
            pen = -cross / edge_len   # positive = outside this edge
            if pen > max_pen:
                max_pen = pen
        violation += max_pen
    return violation


def _rect_overlap_area(b1: BuildingGene, b2: BuildingGene) -> float:
    """
    [REMOVED_ZH:9]（[REMOVED_ZH:2] AABB，[REMOVED_ZH:10]）。
    [REMOVED_ZH:14]。
    """
    dx = abs(b1.cx - b2.cx)
    dy = abs(b1.cy - b2.cy)
    # [REMOVED_ZH:11]
    r1 = math.hypot(b1.w, b1.d) / 2.0
    r2 = math.hypot(b2.w, b2.d) / 2.0
    dist = math.hypot(dx, dy)
    # [REMOVED_ZH:7]，[REMOVED_ZH:10]
    overlap_est = max(0.0, r1 + r2 - dist)
    # [REMOVED_ZH:7]（[REMOVED_ZH:2]）
    return overlap_est ** 2


def _site_polygon_ccw(site_pts: list) -> np.ndarray:
    """[REMOVED_ZH:12]"""
    pts = np.array(site_pts, dtype=float)
    # Shoelace signed area
    n = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    signed = 0.5 * (np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    if signed < 0:
        pts = pts[::-1]
    return pts


def _offset_polygon(pts: np.ndarray, offset: float) -> np.ndarray:
    """
    [REMOVED_ZH:8] ([REMOVED_ZH:3])。
    [REMOVED_ZH:3]：[REMOVED_ZH:5]。[REMOVED_ZH:9]。
    """
    centroid = pts.mean(axis=0)
    vecs     = pts - centroid
    norms    = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    offsets  = vecs / norms * offset
    return pts - offsets


# ════════════════════════════════════════════════════════════════
# [REMOVED_ZH:2]compute
# ════════════════════════════════════════════════════════════════

class ConstraintChecker:
    """
    Parameters
    ----------
    site_pts   : list of [x,y] — [REMOVED_ZH:6]（[REMOVED_ZH:4]）
    setback    : float — [REMOVED_ZH:4] (m)
    far_max    : float — [REMOVED_ZH:5]
    bcr_max    : float — [REMOVED_ZH:5]
    floor_h    : float — [REMOVED_ZH:4] (m)
    """

    def __init__(self,
                 site_pts:  list,
                 setback:   float = 3.0,
                 far_max:   float = 3.0,
                 bcr_max:   float = 0.6,
                 floor_h:   float = 4.5):
        self.site_poly    = _site_polygon_ccw(site_pts)
        self.site_area    = _poly_area(self.site_poly)
        self.setback_poly = _offset_polygon(self.site_poly, setback)
        self.setback_area = _poly_area(self.setback_poly)
        self.far_max      = far_max
        self.bcr_max      = bcr_max
        self.floor_h      = floor_h

    # ── [REMOVED_ZH:4] ───────────────────────────────────────────────

    def c1_far(self, design: Design) -> float:
        """[REMOVED_ZH:5]：max(0, FAR - far_max)"""
        total_gfa = sum(b.w * b.d * b.floors for b in design.buildings)
        far       = total_gfa / (self.site_area + 1e-6)
        return max(0.0, far - self.far_max)

    def c2_bcr(self, design: Design) -> float:
        """[REMOVED_ZH:5]：max(0, BCR - bcr_max)"""
        total_fp = sum(b.w * b.d for b in design.buildings)
        bcr      = total_fp / (self.site_area + 1e-6)
        return max(0.0, bcr - self.bcr_max)

    def c3_setback(self, design: Design) -> float:
        """[REMOVED_ZH:4]：[REMOVED_ZH:15]"""
        total_v = 0.0
        for b in design.buildings:
            poly = _rect_polygon(b)
            total_v += _polygon_inside_polygon_approx(poly, self.setback_poly)
        return total_v

    def c4_containment(self, design: Design) -> float:
        """[REMOVED_ZH:6]：[REMOVED_ZH:10]"""
        total_v = 0.0
        for b in design.buildings:
            poly = _rect_polygon(b)
            total_v += _polygon_inside_polygon_approx(poly, self.site_poly)
        return total_v

    def c5_overlap(self, design: Design) -> float:
        """[REMOVED_ZH:13]"""
        buildings = design.buildings
        total_v   = 0.0
        for i in range(len(buildings)):
            for j in range(i + 1, len(buildings)):
                total_v += _rect_overlap_area(buildings[i], buildings[j])
        return total_v

    # ── [REMOVED_ZH:2] ───────────────────────────────────────────────────

    def check_all(self, design: Design) -> dict:
        """
        Returns
        -------
        dict with keys:
          c1_far, c2_bcr, c3_setback, c4_containment, c5_overlap,
          total_violation, feasible,
          far_actual, bcr_actual
        """
        c1 = self.c1_far(design)
        c2 = self.c2_bcr(design)
        c3 = self.c3_setback(design)
        c4 = self.c4_containment(design)
        c5 = self.c5_overlap(design)
        total = c1 + c2 + c3 * 0.1 + c4 * 0.1 + c5 * 0.01

        # compute[REMOVED_ZH:3]（[REMOVED_ZH:3]）
        total_gfa = sum(b.w * b.d * b.floors for b in design.buildings)
        total_fp  = sum(b.w * b.d            for b in design.buildings)
        far_act   = total_gfa / (self.site_area + 1e-6)
        bcr_act   = total_fp  / (self.site_area + 1e-6)

        return {
            "c1_far":          c1,
            "c2_bcr":          c2,
            "c3_setback":      c3,
            "c4_containment":  c4,
            "c5_overlap":      c5,
            "total_violation": total,
            "feasible":        total < 1e-4,
            "far_actual":      far_act,
            "bcr_actual":      bcr_act,
        }

    def violation_vector(self, design: Design) -> np.ndarray:
        """[REMOVED_ZH:2] (5,) [REMOVED_ZH:4]，[REMOVED_ZH:1] NSGA-II constraint-domination [REMOVED_ZH:2]"""
        return np.array([
            self.c1_far(design),
            self.c2_bcr(design),
            self.c3_setback(design),
            self.c4_containment(design),
            self.c5_overlap(design),
        ])
