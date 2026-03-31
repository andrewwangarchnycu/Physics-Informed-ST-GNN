"""
03_model/layers/rgcn_block.py
Relational Graph Convolutional Network Block
[REMOVED_ZH:2] UrbanGraph [REMOVED_ZH:2] Eq.(2)，5 [REMOVED_ZH:15]。
[REMOVED_ZH:2] residual connection + LayerNorm [REMOVED_ZH:6]。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RGCNBlock(nn.Module):
    """
    [REMOVED_ZH:2] RGCN [REMOVED_ZH:1]，[REMOVED_ZH:8]。

    Parameters
    ----------
    in_dim        : [REMOVED_ZH:6]
    out_dim       : [REMOVED_ZH:6]
    relation_types: [REMOVED_ZH:8] (e.g., ['shadow','veg_et','convective','semantic','contiguity'])
    use_self_loop : [REMOVED_ZH:7]
    dropout       : dropout rate
    """

    RELATION_TYPES = ["shadow", "veg_et", "convective", "semantic", "contiguity"]

    def __init__(self,
                 in_dim:         int,
                 out_dim:        int,
                 relation_types: list = None,
                 use_self_loop:  bool = True,
                 dropout:        float = 0.1):
        super().__init__()
        self.relations = relation_types or self.RELATION_TYPES
        self.out_dim   = out_dim

        # [REMOVED_ZH:7] W_r ([REMOVED_ZH:2] Eq.2)
        self.W_rel = nn.ModuleDict({
            r: nn.Linear(in_dim, out_dim, bias=False)
            for r in self.relations
        })
        # [REMOVED_ZH:5]
        self.W_self = nn.Linear(in_dim, out_dim, bias=True) if use_self_loop else None

        self.norm     = nn.LayerNorm(out_dim)
        self.act      = nn.PReLU()
        self.drop     = nn.Dropout(dropout)

        # Residual projection（in_dim ≠ out_dim [REMOVED_ZH:1]）
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) \
                        if in_dim != out_dim else nn.Identity()

    def forward(self,
                 x:           Tensor,
                 edge_indices: dict,
                 edge_attrs:   dict = None) -> Tensor:
        """
        x           : (N, in_dim)
        edge_indices: {rel_name: (2, E) LongTensor}
        edge_attrs  : {rel_name: (E, attr_dim)} ([REMOVED_ZH:3]，[REMOVED_ZH:2])
        return      : (N, out_dim)
        """
        N   = x.size(0)
        out = torch.zeros(N, self.out_dim, device=x.device)

        for rel, W in self.W_rel.items():
            ei = edge_indices.get(rel)
            if ei is None or ei.size(1) == 0:
                continue
            src, dst = ei[0], ei[1]

            # [REMOVED_ZH:4] (N_src, out_dim)
            msg = W(x[src])

            # [REMOVED_ZH:4] (c_{i,r} = |N_i^r|)
            deg = torch.zeros(N, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
            deg_inv = (1.0 / deg.clamp(min=1.0)).unsqueeze(1)

            # [REMOVED_ZH:2]
            agg = torch.zeros(N, self.out_dim, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            out = out + agg * deg_inv

        # [REMOVED_ZH:3]
        if self.W_self is not None:
            out = out + self.W_self(x)

        # Residual + Norm + Act
        out = self.norm(out + self.res_proj(x))
        out = self.act(out)
        return self.drop(out)


class RGCNEncoder(nn.Module):
    """
    [REMOVED_ZH:2] n_layers [REMOVED_ZH:1] RGCNBlock。
    """
    def __init__(self,
                 in_dim:   int,
                 hid_dim:  int,
                 n_layers: int = 3,
                 dropout:  float = 0.1):
        super().__init__()
        dims   = [in_dim] + [hid_dim] * n_layers
        self.layers = nn.ModuleList([
            RGCNBlock(dims[i], dims[i+1], dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: Tensor, edge_indices: dict) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_indices)
        return x