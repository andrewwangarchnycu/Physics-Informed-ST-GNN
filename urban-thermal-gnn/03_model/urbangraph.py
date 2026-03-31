"""
03_model/urbangraph.py
════════════════════════════════════════════════════════════════
PIN-ST-GNN [REMOVED_ZH:3] (UrbanGraph)
[REMOVED_ZH:1] UrbanGraph ICLR 2026 [REMOVED_ZH:4] (Figure 3)

Pipeline:
  Input MLP → RGCN × 3 → Fusion MLP + Global Context → LSTM → Output MLP
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional

from layers.input_mlp  import InputMLP, GlobalContextMLP
from layers.rgcn_block import RGCNEncoder
from layers.lstm_layer import TemporalLSTM
from layers.output_mlp import OutputMLP
from loss.data_loss     import data_loss, sensor_supervision_loss
from loss.physics_penalty import total_physics_loss

DIM_OBJECT = 7
DIM_AIR_V1 = 8   # V1: [Ta, MRT, Va, RH, SVF, shadow, Bh, Th]
DIM_AIR_V2 = 9   # V2: [Ta, MRT, Va, RH, SVF, shadow, Bh, Th, Ts]
DIM_AIR    = DIM_AIR_V2   # Default for new models
ENV_DIM    = 7   # [Ta, RH, WS, WDsin, WDcos, GHI, SolAlt]
TIME_DIM   = 2   # [sin_hour, cos_hour]


class UrbanGraph(nn.Module):
    """
    Physics-Informed Spatio-Temporal GNN for UTCI prediction.

    Parameters
    ----------
    hidden_dim    : [REMOVED_ZH:6] (default 128)
    n_rgcn_layers : RGCN [REMOVED_ZH:2] (default 3)
    lstm_hidden   : LSTM [REMOVED_ZH:4] (default 256)
    lstm_layers   : LSTM [REMOVED_ZH:2] (default 1)
    out_timesteps : [REMOVED_ZH:5] (default 11, 8:00–18:00)
    dropout       : dropout rate
    lambdas       : [REMOVED_ZH:6] dict
    dim_air       : air node feature dimension (8 for V1, 9 for V2 with surface temp)
    """

    def __init__(self,
                 hidden_dim:    int   = 128,
                 n_rgcn_layers: int   = 3,
                 lstm_hidden:   int   = 256,
                 lstm_layers:   int   = 1,
                 out_timesteps: int   = 11,
                 dropout:       float = 0.1,
                 lambdas:       dict  = None,
                 dim_air:       int   = DIM_AIR):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.out_timesteps = out_timesteps
        self.dim_air       = dim_air
        self.lambdas       = lambdas or {"lambda1": 0.1, "lambda2": 0.05, "lambda3": 0.05}

        # ── [REMOVED_ZH:4] ──────────────────────────────────
        self.obj_encoder = InputMLP(DIM_OBJECT, hidden_dim, dropout)
        self.air_encoder = InputMLP(dim_air,    hidden_dim, dropout)
        self.ctx_encoder = GlobalContextMLP(hidden_dim, dropout)

        # ── [REMOVED_ZH:5] ────────────────────────────────
        self.rgcn = RGCNEncoder(hidden_dim, hidden_dim, n_rgcn_layers, dropout)

        # ── [REMOVED_ZH:2] MLP ([REMOVED_ZH:2] Eq.8) ─────────────────────
        ctx_out_dim = hidden_dim + hidden_dim // 2
        fuse_in     = hidden_dim + ctx_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── h0 [REMOVED_ZH:2]：RGCN hidden_dim -> lstm_hidden ───
        self.h0_proj = nn.Linear(hidden_dim, lstm_hidden)

        # ── [REMOVED_ZH:2] LSTM ─────────────────────────────────
        self.lstm = TemporalLSTM(lstm_hidden, lstm_hidden, lstm_layers, dropout)

        # ── [REMOVED_ZH:3] ────────────────────────────────────
        self.head = OutputMLP(lstm_hidden, out_timesteps, n_layers=2, dropout=dropout)

    # ── Forward Propagation ─────────────────────────────────────────────────
    def forward(self,
                 obj_feat:      Tensor,
                 air_feat:      Tensor,
                 dynamic_edges: List[Dict],
                 static_edges:  Dict,
                 env_seq:       Tensor,
                 time_seq:      Tensor) -> Tensor:
        """
        Parameters
        ----------
        obj_feat      : (N_obj, DIM_OBJECT)
        air_feat      : (N_air, T, DIM_AIR)
        dynamic_edges : list[T] of {rel: (src, dst)} — [REMOVED_ZH:2]Dynamic Edges
        static_edges  : {rel: edge_index}             — Static Edges
        env_seq       : (T, ENV_DIM)                  — [REMOVED_ZH:4]Sequence
        time_seq      : (T, TIME_DIM)                 — [REMOVED_ZH:4]Sequence

        Returns
        -------
        (N_air, T) — [REMOVED_ZH:5] UTCI
        """
        N_air = air_feat.shape[0]
        T     = air_feat.shape[1]
        device = obj_feat.device

        # ── Object Node Initial Embedding ──────────────────────────
        h_obj_init = self.obj_encoder(obj_feat)   # (N_obj, d)

        # ── [REMOVED_ZH:7] air node [REMOVED_ZH:2] ─────────────
        air_hidden_list = []

        for t in range(T):
            # air node [REMOVED_ZH:6]
            h_air_t = self.air_encoder(air_feat[:, t, :])   # (N_air, d)

            # [REMOVED_ZH:4] + [REMOVED_ZH:6]
            h_all = torch.cat([h_obj_init, h_air_t], dim=0)  # (N_obj+N_air, d)
            N_all = h_all.shape[0]

            # [REMOVED_ZH:8] ([REMOVED_ZH:2] + static)
            edges_t = self._merge_edges(dynamic_edges[t] if t < len(dynamic_edges) else {},
                                         static_edges, N_all, device)

            # RGCN
            h_all = self.rgcn(h_all, edges_t)                # (N_all, d)
            h_air_t = h_all[h_obj_init.shape[0]:]            # (N_air, d)

            air_hidden_list.append(h_air_t.unsqueeze(1))     # (N_air, 1, d)

        # (N_air, T, d)
        air_hidden = torch.cat(air_hidden_list, dim=1)

        # ── [REMOVED_ZH:2] RGCN [REMOVED_ZH:4] LSTM warmup ────────────
        # [REMOVED_ZH:1] t=0 [REMOVED_ZH:1] RGCN [REMOVED_ZH:2]
        h0_all = self.rgcn(
            torch.cat([h_obj_init, self.air_encoder(air_feat[:, 0, :])], dim=0),
            self._merge_edges(dynamic_edges[0] if dynamic_edges else {},
                               static_edges, h_obj_init.shape[0] + N_air, device)
        )
        h0_air = h0_all[h_obj_init.shape[0]:]   # (N_air, d)

        # ── [REMOVED_ZH:5]Sequence ────────────────────────────
        ctx = self.ctx_encoder(env_seq, time_seq)   # (T, ctx_dim)
        ctx_expand = ctx.unsqueeze(0).expand(N_air, -1, -1)   # (N_air, T, ctx_dim)

        # ── [REMOVED_ZH:2] ──────────────────────────────────────
        fuse_input = torch.cat([air_hidden, ctx_expand], dim=-1)   # (N_air, T, d+ctx)
        x_lstm = self.fusion(fuse_input)                            # (N_air, T, lstm_d)

        # ── LSTM ──────────────────────────────────────
        lstm_out = self.lstm(x_lstm, self.h0_proj(h0_air))  # (N_air, T, lstm_d)

        # ── [REMOVED_ZH:3]：[REMOVED_ZH:5] hidden state [REMOVED_ZH:4] T ──
        h_last   = lstm_out[:, -1, :]              # (N_air, lstm_d)
        utci_pred = self.head(h_last)              # (N_air, T_pred)

        return utci_pred

    def compute_loss(self,
                      pred:         Tensor,
                      target:       Tensor,
                      svf:          Tensor,
                      in_shadow:    Tensor,
                      sol_alt_seq:  Tensor,
                      bldg_height:  Tensor,
                      quality_w:    Tensor = None,
                      sensor_utci:  Tensor = None,
                      sensor_mask:  Tensor = None,
                      lambda_sense: float  = 0.5) -> Dict[str, Tensor]:
        """
        [REMOVED_ZH:4] loss + [REMOVED_ZH:4]。

        Returns
        -------
        dict: loss_data, loss_physics, loss_sensor, loss_total
        """
        l_data  = data_loss(pred, target, quality_w)
        l_phys  = total_physics_loss(pred, svf, in_shadow.bool(),
                                      sol_alt_seq, bldg_height, self.lambdas)
        l_sense = (sensor_supervision_loss(pred, sensor_utci, sensor_mask)
                   if sensor_utci is not None else
                   torch.tensor(0.0, device=pred.device))

        l_total = l_data + l_phys + lambda_sense * l_sense
        return {
            "loss_data":    l_data,
            "loss_physics": l_phys,
            "loss_sensor":  l_sense,
            "loss_total":   l_total,
        }

    # ── [REMOVED_ZH:2]：Merge Dynamic and Static Edges ─────────────────────────────
    @staticmethod
    def _merge_edges(dyn_edges: Dict, static_edges: Dict,
                      n_nodes: int, device) -> Dict[str, Tensor]:
        """
        [REMOVED_ZH:2]Dynamic Edges（object→air, air→air）andStatic Edges，
        [REMOVED_ZH:2] {rel_name: edge_index_tensor}。
        """
        merged = {}

        # Static Edges
        for rel, ei in static_edges.items():
            if isinstance(ei, torch.Tensor):
                merged[rel] = ei.to(device)

        # Dynamic Edges
        for rel, (src, dst, attr) in dyn_edges.items():
            if isinstance(src, torch.Tensor) and src.numel() > 0:
                merged[rel] = torch.stack([src, dst]).to(device)
            elif hasattr(src, '__len__') and len(src) > 0:
                import numpy as np
                ei = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
                merged[rel] = ei

        return merged


# ── [REMOVED_ZH:4] ───────────────────────────────────────────────────
def build_model(cfg: dict) -> UrbanGraph:
    """[REMOVED_ZH:1] urbangraph_params.yaml Build[REMOVED_ZH:2]。"""
    m = cfg.get("model", {})
    return UrbanGraph(
        hidden_dim    = m.get("hidden_dim",    128),
        n_rgcn_layers = m.get("n_rgcn_layers", 3),
        lstm_hidden   = m.get("lstm_hidden",   256),
        lstm_layers   = m.get("lstm_layers",   1),
        out_timesteps = m.get("out_timesteps", 11),
        dropout       = m.get("dropout",       0.1),
        lambdas       = m.get("lambdas", {"lambda1":0.1,"lambda2":0.05,"lambda3":0.05}),
        dim_air       = m.get("dim_air",       DIM_AIR),
    )