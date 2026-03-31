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
                 sensor_res:  float = 2.0):
        self.model       = model
        self.norm_stats  = norm_stats
        self.epw_data    = epw_data
        self.site_pts    = site_pts
        self.cfg         = cfg
        self.checker     = checker
        self.device      = device
        self.sensor_res  = sensor_res

        # [REMOVED_ZH:4]（[REMOVED_ZH:4] import）
        from geometry_converter import GNNInputBuilder
        from train import build_env_time_seq

        self._builder = GNNInputBuilder(norm_stats, epw_data)
        env_seq, time_seq = build_env_time_seq(epw_data,
                                               list(range(8, 19)),  # 8:00–18:00
                                               month=7)
        self.env_seq  = env_seq.to(device)
        self.time_seq = time_seq.to(device)

        # [REMOVED_ZH:4]（[REMOVED_ZH:2]compute[REMOVED_ZH:3]）
        from constraints import _poly_area, _site_polygon_ccw
        self._site_area = _poly_area(_site_polygon_ccw(site_pts))

    # ── [REMOVED_ZH:4] ─────────────────────────────────────────────

    def evaluate(self, genes: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        genes : (n_genes,) float array [0,1]

        Returns
        -------
        objectives  : (2,) [mean_utci, -green_ratio]
        constraints : (5,) violation vector (≥0)
        """
        design = decode(genes, self.cfg)
        cv     = self.checker.violation_vector(design)

        # [REMOVED_ZH:11]（[REMOVED_ZH:2] GNN compute）
        if cv.sum() > 50.0:
            return np.array([60.0, 0.0]), cv

        try:
            mean_utci, green_ratio = self._run_gnn(design)
        except Exception:
            return np.array([60.0, 0.0]), cv

        return np.array([mean_utci, -green_ratio]), cv

    def batch_evaluate(self, population: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        population : (pop_size, n_genes)

        Returns
        -------
        F  : (pop_size, 2)
        CV : (pop_size, 5)
        """
        n = len(population)
        F  = np.empty((n, 2))
        CV = np.empty((n, 5))
        for i, genes in enumerate(population):
            F[i], CV[i] = self.evaluate(genes)
        return F, CV

    # ── [REMOVED_ZH:2] GNN [REMOVED_ZH:2] ────────────────────────────────────────

    def _run_gnn(self, design: Design) -> Tuple[float, float]:
        """
        [REMOVED_ZH:1] Design [REMOVED_ZH:2] GNN [REMOVED_ZH:2] → [REMOVED_ZH:2] → [REMOVED_ZH:2] (mean_utci, green_ratio)
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

        return mean_utci, green_ratio
