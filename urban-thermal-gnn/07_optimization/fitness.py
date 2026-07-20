"""
07_optimization/fitness.py
════════════════════════════════════════════════════════════════
Fitness[REMOVED_ZH:3]：[REMOVED_ZH:7] GNN [REMOVED_ZH:2]，compute[REMOVED_ZH:4]

[REMOVED_ZH:2]：
  f1 = mean_utci     — [REMOVED_ZH:5]、[REMOVED_ZH:7] UTCI (°C)，minimize
  f2 = -green_ratio  — [REMOVED_ZH:3]（[REMOVED_ZH:4] / [REMOVED_ZH:4]），maximize（[REMOVED_ZH:2]）

[REMOVED_ZH:4]（[REMOVED_ZH:1] ConstraintChecker compute）[REMOVED_ZH:4] NSGA-II [REMOVED_ZH:2]
feasibility-first [REMOVED_ZH:2]，[REMOVED_ZH:8]。

[REMOVED_ZH:2]：FitnessEvaluator [REMOVED_ZH:5] GNN model [REMOVED_ZH:3]，
      [REMOVED_ZH:5]（app.py）[REMOVED_ZH:5] nsga2_engine [REMOVED_ZH:2]。
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

# ── [REMOVED_ZH:2] ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "02_graph_construction"))
sys.path.insert(0, str(_HERE.parent / "03_model"))
sys.path.insert(0, str(_HERE.parent / "06_deployment"))

from chromosome import Design, ChromosomeConfig, decode
from constraints import ConstraintChecker

# ── Walkway thermal-exposure penalty (ported from Thesis_GIA appendix
#    appx:walkway_derivation, eq. Phi(UTCI_j) = ReLU(UTCI_j - UTCI_comfort) /
#    (UTCI_max - UTCI_comfort)). UTCI_COMFORT is the fixed comfort threshold
#    used throughout the thesis (26 degC); UTCI_MAX follows the appendix's
#    own reported empirical ceiling (~55-56 degC peak across sampled scenes)
#    used to normalise the penalty to [0,1]. ──────────────────────────────
UTCI_COMFORT = 26.0
UTCI_MAX     = 56.0


def _phi_penalty(utci: np.ndarray) -> np.ndarray:
    """Elementwise dimensionless heat-stress penalty in [0, 1]."""
    return np.clip(utci - UTCI_COMFORT, 0.0, None) / (UTCI_MAX - UTCI_COMFORT)


def walkway_exposure(utci: np.ndarray, sensor_pts: np.ndarray,
                     route: List[List[float]]) -> float:
    """
    Simplified single-route implementation of the thesis's derived
    time-space-averaged walkway exposure metric $\\bar{\\Phi}_{walkway}$
    (Thesis_GIA appx:walkway_derivation). The route is a fixed ordered
    polyline of waypoints (site-local coordinates); each consecutive pair
    forms one walkway edge whose exposure is the real segment distance
    weighted by the heat-stress penalty at the destination waypoint's
    nearest sensor node, averaged over edges (space) then timesteps (time).

    NOTE (honest scope disclosure): this evaluates ONE fixed route per call,
    not the full multi-route walkway-graph $E_{walk}$ the appendix derivation
    is stated in terms of. It validates the already-derived penalty formula
    and three-objective NSGA-II formulation; a full A*-based multi-route
    walkway graph is left as future work (see thesis discussion).

    Parameters
    ----------
    utci       : (N_air, T) denormalised UTCI
    sensor_pts : (N_air, 2) sensor coordinates, same frame as `route`
    route      : ordered list of [x, y] waypoints, >= 2 points

    Returns
    -------
    float — time-averaged, distance-weighted mean penalty along the route
    """
    if route is None or len(route) < 2:
        return 0.0
    route = np.asarray(route, dtype=float)
    # nearest sensor index for every waypoint (single nearest-neighbour pass)
    idx = np.array([
        int(np.argmin(((sensor_pts - wp) ** 2).sum(axis=1)))
        for wp in route
    ])
    seg_dist = np.linalg.norm(route[1:] - route[:-1], axis=1)  # (n_seg,)
    dest_idx = idx[1:]                                          # (n_seg,)
    phi = _phi_penalty(utci[dest_idx, :])                       # (n_seg, T)
    spatial = (seg_dist[:, None] * phi).sum(axis=0) / seg_dist.sum()  # (T,)
    return float(spatial.mean())


class FitnessEvaluator:
    """
    Parameters
    ----------
    model        : [REMOVED_ZH:3] GPU/CPU [REMOVED_ZH:1] UrbanGraph [REMOVED_ZH:2]
    norm_stats   : {"ta":{"mean":…,"std":…}, "utci":…, …}
    epw_data     : EPWData [REMOVED_ZH:2]（[REMOVED_ZH:6]Sequence）
    site_pts     : list of [x,y] [REMOVED_ZH:6]
    cfg          : ChromosomeConfig
    checker      : ConstraintChecker
    device       : "cuda" | "cpu"
    sensor_res   : [REMOVED_ZH:5] (m)
    """

    def __init__(self,
                 model,
                 norm_stats:  dict,
                 epw_data,
                 site_pts:    list,
                 cfg:         ChromosomeConfig,
                 checker:     ConstraintChecker,
                 device:      str = "cuda",
                 sensor_res:  float = 2.0,
                 dim_air:     int = 9,
                 walkway_route: list | None = None):
        self.model       = model
        self.norm_stats  = norm_stats
        self.epw_data    = epw_data
        self.site_pts    = site_pts
        self.cfg         = cfg
        self.checker     = checker
        self.device      = device
        self.sensor_res  = sensor_res
        self.dim_air     = dim_air
        # Fixed pedestrian route (site-local [x,y] waypoints) used for the
        # simplified single-route walkway-exposure objective f2; None
        # disables f2 (falls back to the original 2-objective evaluator).
        self.walkway_route = walkway_route

        # [REMOVED_ZH:4]（[REMOVED_ZH:4] import）
        from geometry_converter import GNNInputBuilder
        from train import build_env_time_seq

        self._builder = GNNInputBuilder(norm_stats, epw_data, dim_air=dim_air)
        env_seq, time_seq = build_env_time_seq(epw_data,
                                               list(range(8, 19)),  # 8:00–18:00
                                               month=7)
        self.env_seq  = env_seq.to(device)
        self.time_seq = time_seq.to(device)

        # [REMOVED_ZH:4]（[REMOVED_ZH:2]compute[REMOVED_ZH:3]）
        from constraints import _poly_area, _site_polygon_ccw
        self._site_area = _poly_area(_site_polygon_ccw(site_pts))

    # ── [REMOVED_ZH:4] ─────────────────────────────────────────────

    @property
    def n_obj(self) -> int:
        return 3 if self.walkway_route is not None else 2

    def evaluate(self, genes: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        genes : (n_genes,) float array [0,1]

        Returns
        -------
        objectives  : (2,) [mean_utci, -green_ratio] if walkway_route is None,
                      else (3,) [mean_utci, walkway_exposure, -green_ratio]
        constraints : (5,) violation vector (≥0)
        """
        design = decode(genes, self.cfg)
        cv     = self.checker.violation_vector(design)
        # worst-case fallback objectives (infeasible / evaluation failure)
        bad = np.array([60.0, 1.0, 0.0]) if self.n_obj == 3 else np.array([60.0, 0.0])

        if cv.sum() > 50.0:
            return bad, cv

        try:
            result = self._run_gnn(design)
        except Exception:
            return bad, cv

        if self.n_obj == 3:
            mean_utci, walk_exp, green_ratio = result
            return np.array([mean_utci, walk_exp, -green_ratio]), cv
        else:
            mean_utci, green_ratio = result
            return np.array([mean_utci, -green_ratio]), cv

    def batch_evaluate(self, population: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        population : (pop_size, n_genes)

        Returns
        -------
        F  : (pop_size, n_obj)  -- n_obj is 2 or 3, see `n_obj` property
        CV : (pop_size, 5)
        """
        n = len(population)
        F  = np.empty((n, self.n_obj))
        CV = np.empty((n, 5))
        for i, genes in enumerate(population):
            F[i], CV[i] = self.evaluate(genes)
        return F, CV

    # ── [REMOVED_ZH:2] GNN [REMOVED_ZH:2] ────────────────────────────────────────

    def _run_gnn(self, design: Design):
        """
        [REMOVED_ZH:1] Design [REMOVED_ZH:2] GNN [REMOVED_ZH:2] → [REMOVED_ZH:2] → [REMOVED_ZH:2]
        (mean_utci, green_ratio), or (mean_utci, walkway_exposure, green_ratio)
        when self.walkway_route is set.
        """
        payload = {
            "site_boundary": self.site_pts,
            "buildings": [
                {"footprint": b.footprint_polygon().tolist(),
                 "height": b.floors * self.cfg.floor_height,
                 "floor_count": b.floors}
                for b in design.buildings
            ],
            "trees": [
                {"x": t.x, "y": t.y,
                 "radius": t.radius, "height": t.height}
                for t in design.trees
            ],
            "sensor_resolution": self.sensor_res,
        }

        gnn_inputs = self._builder.build(payload)
        if gnn_inputs is None or gnn_inputs["sensor_pts"].shape[0] < 5:
            raise ValueError("Too few sensor points")

        sensor_pts = gnn_inputs["sensor_pts"]
        obj_feat   = torch.from_numpy(gnn_inputs["obj_feat"]).to(self.device)
        air_feat   = torch.from_numpy(gnn_inputs["air_feat"]).to(self.device)
        static_edges = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in gnn_inputs["static_edges"].items()
        }
        dynamic_edges = [{}] * air_feat.shape[1]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                obj_feat      = obj_feat,
                air_feat      = air_feat,
                dynamic_edges = dynamic_edges,
                static_edges  = static_edges,
                env_seq       = self.env_seq,
                time_seq      = self.time_seq,
            )  # (N_air, T)

        # denormalize
        mu  = self.norm_stats["utci"]["mean"]
        std = self.norm_stats["utci"]["std"]
        utci = pred.cpu().numpy() * std + mu   # (N_air, T)

        mean_utci = float(utci.mean())

        # [REMOVED_ZH:3]：[REMOVED_ZH:5] / [REMOVED_ZH:4]
        tree_canopy = sum(math.pi * t.radius**2 for t in design.trees)
        green_ratio = min(1.0, tree_canopy / (self._site_area + 1e-6))

        if self.walkway_route is not None:
            walk_exp = walkway_exposure(utci, sensor_pts, self.walkway_route)
            return mean_utci, walk_exp, green_ratio

        return mean_utci, green_ratio
