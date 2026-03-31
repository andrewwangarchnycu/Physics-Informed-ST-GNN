"""03_model/layers/output_mlp.py — [REMOVED_ZH:3]"""
import torch
import torch.nn as nn
from torch import Tensor


class OutputMLP(nn.Module):
    """
    Single-head [REMOVED_ZH:3] ([REMOVED_ZH:2] Table A8 [REMOVED_ZH:4] Multi-head)。
    [REMOVED_ZH:1] LSTM [REMOVED_ZH:12] T [REMOVED_ZH:3]。
    """
    def __init__(self, hidden_dim: int = 256, out_timesteps: int = 11,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, out_timesteps)]
        self.net = nn.Sequential(*layers)

    def forward(self, h_last: Tensor) -> Tensor:
        """
        h_last : (N, hidden_dim) LSTM [REMOVED_ZH:8]
        return : (N, T_pred) UTCI [REMOVED_ZH:2]
        """
        return self.net(h_last)