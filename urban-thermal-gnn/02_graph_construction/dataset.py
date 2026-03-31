"""
02_graph_construction/dataset.py
════════════════════════════════════════════════════════════════
UTCIGraphDataset  -  Physics-Informed ST-GNN dataset loader

Builds heterogeneous graph data from:
  - ground_truth.h5  (HDF5 with simulation results + normalization stats + splits)
  - scenarios.pkl    (building/tree geometry per scenario)
  - epw_data.pkl     (EPW weather data, used for surface temperature computation)

No torch_geometric dependency - uses lightweight custom containers.

Node types:
  "object" : building nodes  (N_obj, 7)
  "air"    : sensor point nodes  (N_air, T, dim_air)

Edge types:
  ("object", "semantic",    "object") : fully-connected between buildings
  ("air",    "contiguity",  "air")    : KNN (K=8) among air nodes

Air node feature order (DIM_AIR=8, V1):
  0: ta_norm      - normalized air temperature
  1: mrt_norm     - normalized mean radiant temperature
  2: va_norm      - normalized wind velocity
  3: rh_norm      - normalized relative humidity
  4: svf          - sky view factor (static, [0,1])
  5: in_shadow    - shadow flag (0.0 or 1.0, temporal)
  6: bldg_h_norm  - nearest building height / 50.0
  7: tree_h_norm  - nearest tree height / 12.0

Air node feature order (DIM_AIR=9, V2 — adds surface temperature):
  8: ts_norm      - normalized surface temperature

Object node feature order (DIM_OBJECT=7):
  0: height / 50.0
  1: floors / 12.0
  2: footprint_area / 2000.0
  3: centroid_x / 80.0
  4: centroid_y / 80.0
  5: gfa / 20000.0
  6: is_L_shape (0/1)

IMPORTANT - edge index offset:
  urbangraph.py forward() concatenates h_all = [h_obj; h_air].
  Contiguity edge indices (air-to-air) are stored 0-indexed in air space;
  get() adds N_obj offset so they correctly address the concatenated tensor.
  Semantic edge indices (object-to-object) need no offset (already 0..N_obj-1).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import h5py


# ════════════════════════════════════════════════════════════════
# 1. Lightweight graph containers (no torch_geometric dependency)
# ════════════════════════════════════════════════════════════════

class NodeStorage:
    """
    Attribute bag mirroring PyG NodeStorage.
    Holds arbitrary tensor attributes (.x, .y, .pos, ...).
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        parts = [
            f"{k}={tuple(v.shape)}"
            for k, v in vars(self).items()
            if hasattr(v, "shape")
        ]
        return f"NodeStorage({', '.join(parts)})"


class EdgeStorage:
    """
    Holds edge_index (2, E) LongTensor.
    The attribute is only set when edge_index is provided, so
    hasattr(edge_storage, 'edge_index') correctly returns False
    when no edges were registered.
    """
    def __init__(self, edge_index: Optional[torch.Tensor] = None):
        if edge_index is not None:
            self.edge_index = edge_index

    def __repr__(self) -> str:
        ei = getattr(self, "edge_index", None)
        if ei is not None:
            return f"EdgeStorage(edge_index={tuple(ei.shape)})"
        return "EdgeStorage(empty)"


class HeteroGraphData:
    """
    Lightweight heterogeneous graph container.

    Supports:
      data["node_type"]            -> NodeStorage
      data[("src", "rel", "dst")]  -> EdgeStorage
      data.dynamic_edges           -> list[T] of dicts
    """
    def __init__(self):
        self._node_stores: Dict[str, NodeStorage] = {}
        self._edge_stores: Dict[tuple, EdgeStorage] = {}
        self.dynamic_edges: List[dict] = []

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._node_stores:
                self._node_stores[key] = NodeStorage()
            return self._node_stores[key]
        if isinstance(key, tuple) and len(key) == 3:
            if key not in self._edge_stores:
                self._edge_stores[key] = EdgeStorage()
            return self._edge_stores[key]
        raise KeyError(f"Unsupported key: {key!r}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._node_stores[key] = value
        elif isinstance(key, tuple) and len(key) == 3:
            self._edge_stores[key] = value
        else:
            raise KeyError(f"Unsupported key: {key!r}")

    def __repr__(self) -> str:
        return (
            f"HeteroGraphData("
            f"nodes={list(self._node_stores.keys())}, "
            f"edges={list(self._edge_stores.keys())})"
        )


# ════════════════════════════════════════════════════════════════
# 2. UTCIGraphDataset
# ════════════════════════════════════════════════════════════════

class UTCIGraphDataset:
    """
    Dataset for the Physics-Informed ST-GNN model.

    Parameters
    ----------
    h5_path      : path to ground_truth.h5
    scenario_pkl : path to scenarios.pkl
    epw_pkl      : path to epw_data.pkl  (used for surface temperature computation)
    split        : "train" | "val" | "test"
    knn_k        : K for KNN contiguity edges among air nodes (default 8)
    dim_air      : air feature dimension (8=V1, 9=V2 with surface temperature)
    """

    def __init__(
        self,
        h5_path:      str,
        scenario_pkl: str,
        epw_pkl:      str,
        split:        str = "train",
        knn_k:        int = 8,
        dim_air:      int = 8,
    ):
        self._h5_path = str(h5_path)
        self._knn_k   = knn_k
        self._dim_air = dim_air

        # ── Load scenario geometry ────────────────────────────────
        with open(scenario_pkl, "rb") as f:
            all_scenarios: List[dict] = pickle.load(f)
        self._scenario_map: Dict[int, dict] = {
            int(s.get("scenario_id", i)): s
            for i, s in enumerate(all_scenarios)
        }

        # ── Load EPW data for surface temperature computation ─────
        self._epw_data = None
        self._clim_map: Dict[int, object] = {}
        if dim_air >= 9:
            try:
                import __main__
                from shared import HourlyClimate as _HC, EPWData as _ED
                __main__.HourlyClimate = _HC
                __main__.EPWData       = _ED
                with open(epw_pkl, "rb") as f:
                    self._epw_data = pickle.load(f)
                typical = self._epw_data.get_typical_day(month=7, stat="hottest")
                self._clim_map = {h.hour: h for h in typical}
            except Exception as e:
                warnings.warn(f"Could not load EPW for surface temp: {e}")
                self._epw_data = None

        # ── Read HDF5 metadata ────────────────────────────────────
        with h5py.File(self._h5_path, "r") as hf:
            # simulation hours
            self._sim_hours: List[int] = (
                hf["metadata/sim_hours"][()].tolist()
            )

            # normalization statistics
            self._norm_stats: Dict[str, Dict[str, float]] = {}
            for field in hf["normalization"].keys():
                grp = hf[f"normalization/{field}"]
                self._norm_stats[field] = {
                    "mean": float(grp.attrs["mean"]),
                    "std":  float(grp.attrs["std"]),
                }

            # split IDs
            split_key = {
                "train": "train_ids",
                "val":   "val_ids",
                "test":  "test_ids",
            }[split]
            self._ids: List[int] = (
                hf[f"splits/{split_key}"][()].tolist()
            )

        # ── KNN edge cache (0-indexed in air-node space) ──────────
        self._knn_cache: Dict[int, torch.Tensor] = {}

    # ── Properties ───────────────────────────────────────────────

    @property
    def sim_hours(self) -> List[int]:
        """Simulation hours, e.g. [8, 9, ..., 18]."""
        return self._sim_hours

    @property
    def norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Normalization statistics: {field: {"mean": float, "std": float}}."""
        return self._norm_stats

    # ── Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def get(self, idx: int) -> HeteroGraphData:
        """
        Build and return one heterogeneous graph for scenario at index idx.

        Returns
        -------
        HeteroGraphData with:
          data["object"].x                          (N_obj, 7)
          data["air"].x                             (N_air, T, dim_air)
          data["air"].y                             (N_air, T)  normalized UTCI
          data["air"].pos                           (N_air, 2)
          data[("object","semantic","object")].edge_index  (2, E_obj)
          data[("air","contiguity","air")].edge_index      (2, E_air)
          data.dynamic_edges                        [{}] * T
        """
        sid = int(self._ids[idx])
        data = HeteroGraphData()
        T = len(self._sim_hours)
        data.dynamic_edges = [{}] * T

        # ── Read simulation arrays from HDF5 ─────────────────────
        with h5py.File(self._h5_path, "r") as hf:
            grp = hf[f"scenarios/{sid}"]

            sensor_pts   = grp["sensor_pts"][()]              # (N, 2)
            ta           = grp["ta"][()]                      # (T, N)
            mrt          = grp["mrt"][()]                     # (T, N)
            va           = grp["va"][()]                      # (T, N)
            rh           = grp["rh"][()]                      # (T, N)
            utci         = grp["utci"][()]                    # (T, N)
            svf          = grp["svf"][()]                     # (N,)
            in_shadow    = grp["in_shadow"][()].astype(np.float32)  # (T, N)
            bldg_height  = grp["building_height"][()]         # (N,)
            tree_height  = grp["tree_height"][()]             # (N,)

        N = sensor_pts.shape[0]
        actual_T = ta.shape[0]

        # ── Normalization helper ──────────────────────────────────
        def norm(arr: np.ndarray, field: str) -> np.ndarray:
            mu  = self._norm_stats[field]["mean"]
            std = self._norm_stats[field]["std"]
            return ((arr - mu) / std).astype(np.float32)

        # ── Build air node features (N, T, 8) ────────────────────
        # HDF5 stores (T, N) -> transpose to (N, T)
        ta_n  = norm(ta,  "ta").T   # (N, T)
        mrt_n = norm(mrt, "mrt").T  # (N, T)
        va_n  = norm(va,  "va").T   # (N, T)
        rh_n  = norm(rh,  "rh").T   # (N, T)

        svf_nt    = np.repeat(svf[:, None],              actual_T, axis=1)  # (N, T)
        shadow_nt = in_shadow.T                                              # (N, T)
        bh_nt     = np.repeat((bldg_height / 50.0)[:, None], actual_T, axis=1)  # (N, T)
        th_nt     = np.repeat((tree_height  / 12.0)[:, None], actual_T, axis=1) # (N, T)

        # Base 8 features
        feat_list = [ta_n, mrt_n, va_n, rh_n, svf_nt, shadow_nt, bh_nt, th_nt]

        # ── V2: compute surface temperature as 9th feature ────────
        if self._dim_air >= 9:
            ts_nt = self._compute_surface_temp(
                ta.T, va.T, rh.T, svf, N, actual_T)  # (N, T) raw °C
            # Normalize: use 'ts' stats if available, otherwise derive from ta stats
            if "ts" in self._norm_stats:
                ts_n = norm(ts_nt.T, "ts").T  # norm expects (T, N)
            else:
                # Fallback: same distribution as ta but shifted ~+5°C
                mu  = self._norm_stats["ta"]["mean"] + 5.0
                std = self._norm_stats["ta"]["std"] * 1.2
                ts_n = ((ts_nt - mu) / (std + 1e-8)).astype(np.float32)
            feat_list.append(ts_n)

        # Stack along feature axis -> (N, T, dim_air)
        air_feat = np.stack(feat_list, axis=2).astype(np.float32)

        # ── Normalized UTCI target (N, T) ─────────────────────────
        utci_n = norm(utci, "utci").T  # (N, T)

        # ── air NodeStorage ───────────────────────────────────────
        data["air"] = NodeStorage(
            x   = torch.from_numpy(air_feat),        # (N, T, dim_air)
            y   = torch.from_numpy(utci_n),          # (N, T)
            pos = torch.from_numpy(sensor_pts.astype(np.float32)),  # (N, 2)
        )

        # ── object NodeStorage (building features) ────────────────
        scenario = self._scenario_map.get(sid)
        if scenario is not None and scenario.get("buildings"):
            obj_feat = self._extract_object_features(scenario)  # (N_obj, 7)
        else:
            warnings.warn(
                f"Scenario {sid} not found in scenarios.pkl; "
                "using single dummy object node."
            )
            obj_feat = torch.zeros(1, 7, dtype=torch.float32)

        data["object"] = NodeStorage(x=obj_feat)

        N_obj = obj_feat.shape[0]

        # ── KNN contiguity edges (air-to-air) ────────────────────
        # Raw 0-indexed in air-node space; offset by N_obj for concat tensor
        knn_raw = self._get_knn_edges(sid, sensor_pts)   # (2, E)
        knn_offset = knn_raw + N_obj                      # (2, E) offset
        data[("air", "contiguity", "air")] = EdgeStorage(
            edge_index=knn_offset
        )

        # ── Semantic edges (object-to-object, fully connected) ────
        # Object indices are 0..N_obj-1 in h_all, no offset needed
        if N_obj >= 2:
            src, dst = [], []
            for i in range(N_obj):
                for j in range(N_obj):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            sem_ei = torch.tensor([src, dst], dtype=torch.long)
            data[("object", "semantic", "object")] = EdgeStorage(
                edge_index=sem_ei
            )
        # else: hasattr check in train.py will skip missing edge_index

        return data

    # ── Internal helpers ──────────────────────────────────────────

    def _compute_surface_temp(self, ta_nt: np.ndarray, va_nt: np.ndarray,
                               rh_nt: np.ndarray, svf: np.ndarray,
                               N: int, T: int) -> np.ndarray:
        """
        Compute surface temperature (N, T) for the 9th air-node feature.

        Uses per-timestep EPW climate data (GHI) combined with per-sensor
        air temperature, wind speed, and SVF-adjusted wind.

        Falls back to ta + 5°C if EPW data is unavailable.
        """
        ts = np.zeros((N, T), dtype=np.float32)

        if self._epw_data is None or not self._clim_map:
            # Graceful fallback: approximate surface temp as air temp + offset
            ts[:] = ta_nt + 5.0
            return ts

        try:
            from shared.surface_materials import compute_surface_temp_scalar_batch
        except ImportError:
            ts[:] = ta_nt + 5.0
            return ts

        for t_idx, hr in enumerate(self._sim_hours):
            if t_idx >= T:
                break
            clim = self._clim_map.get(hr)
            if clim is None:
                ts[:, t_idx] = ta_nt[:, t_idx] + 5.0
                continue

            ghi = float(clim.ghi)
            rh_scalar = float(clim.rh) / 100.0  # EPW stores 0-100

            # Use per-sensor wind speed (already spatially varied in HDF5)
            ts[:, t_idx] = compute_surface_temp_scalar_batch(
                "concrete",  # default material for training data
                ta_nt[:, t_idx],
                ghi,
                va_nt[:, t_idx],
                rh_scalar,
            )

        return ts

    def _get_knn_edges(self, sid: int, pos: np.ndarray) -> torch.Tensor:
        """
        Build/retrieve cached KNN contiguity edge_index for this scenario.
        Returns LongTensor (2, E), 0-indexed in air-node space.
        """
        if sid in self._knn_cache:
            return self._knn_cache[sid]

        from scipy.spatial import cKDTree

        k = min(self._knn_k, len(pos) - 1)
        tree = cKDTree(pos)
        _, indices = tree.query(pos, k=k + 1)  # k+1: first result is self

        src_list: List[int] = []
        dst_list: List[int] = []
        for i, nbrs in enumerate(indices):
            for j in nbrs[1:]:          # skip self (index 0)
                src_list.append(i)
                dst_list.append(int(j))

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        self._knn_cache[sid] = edge_index
        return edge_index

    @staticmethod
    def _extract_object_features(scenario: dict) -> torch.Tensor:
        """
        Extract (N_obj, 7) float32 tensor from scenario["buildings"].

        Feature layout:
          0: height / 50.0
          1: floors / 12.0
          2: footprint_area / 2000.0  (Shapely .area; fallback to coverage)
          3: centroid_x / 80.0
          4: centroid_y / 80.0
          5: gfa / 20000.0
          6: is_L_shape (0.0 or 1.0)
        """
        buildings = scenario["buildings"]
        N = len(buildings)
        feats = np.zeros((N, 7), dtype=np.float32)

        for i, b in enumerate(buildings):
            feats[i, 0] = float(b.get("height", 0.0)) / 50.0
            feats[i, 1] = float(b.get("floors", 1.0)) / 12.0

            # Footprint area: prefer Shapely polygon, fallback to coverage field
            try:
                fp_area = float(b["footprint"].area)
            except (AttributeError, KeyError, TypeError):
                fp_area = float(
                    b.get("coverage",
                           b.get("gfa", 0.0) / max(float(b.get("floors", 1)), 1))
                )
            feats[i, 2] = fp_area / 2000.0

            cx, cy = b.get("centroid", (0.0, 0.0))
            feats[i, 3] = float(cx) / 80.0
            feats[i, 4] = float(cy) / 80.0
            feats[i, 5] = float(b.get("gfa", 0.0)) / 20000.0
            feats[i, 6] = 1.0 if b.get("shape_type", "rect") == "L" else 0.0

        return torch.from_numpy(feats)
