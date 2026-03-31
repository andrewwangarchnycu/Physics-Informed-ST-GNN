"""
03_model/layers/input_mlp.py  — [REMOVED_ZH:4] MLP
"""
import torch
import torch.nn as nn


class InputMLP(nn.Module):
    """
    [REMOVED_ZH:18]。
    Linear → LayerNorm → ReLU → Linear → ReLU
    [REMOVED_ZH:2] object node (static) [REMOVED_ZH:1] air node ([REMOVED_ZH:2]) [REMOVED_ZH:6]。
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) → (..., hidden_dim)"""
        return self.net(x)


class GlobalContextMLP(nn.Module):
    """
    [REMOVED_ZH:9]: [Ta, RH, WS, WD, GHI, SolarAlt, SolarAz] → d
    """
    ENV_DIM  = 7   # [REMOVED_ZH:6]
    TIME_DIM = 2   # [REMOVED_ZH:6] (sin, cos)

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.env_enc = nn.Sequential(
            nn.Linear(self.ENV_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )
        self.time_enc = nn.Sequential(
            nn.Linear(self.TIME_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, env_feat: torch.Tensor,
                 time_feat: torch.Tensor) -> torch.Tensor:
        """
        env_feat  : (B, ENV_DIM)  [REMOVED_ZH:1] (ENV_DIM,)
        time_feat : (B, TIME_DIM) [REMOVED_ZH:1] (TIME_DIM,)
        return    : (B, hidden_dim + hidden_dim//2)
        """
        e = self.env_enc(env_feat)
        t = self.time_enc(time_feat)
        return self.drop(torch.cat([e, t], dim=-1))