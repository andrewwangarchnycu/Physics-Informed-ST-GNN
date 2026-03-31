"""
03_model/layers/lstm_layer.py
[REMOVED_ZH:2] LSTM [REMOVED_ZH:2]
[REMOVED_ZH:1] UrbanGraph [REMOVED_ZH:4] LSTM [REMOVED_ZH:2] Transformer：
[REMOVED_ZH:4] Markovian [REMOVED_ZH:2]，LSTM [REMOVED_ZH:13]。
[REMOVED_ZH:2] warming-up [REMOVED_ZH:2] ([REMOVED_ZH:2] Eq.9)。
"""
import torch
import torch.nn as nn
from torch import Tensor


class TemporalLSTM(nn.Module):
    """
    [REMOVED_ZH:5]Sequence[REMOVED_ZH:2] LSTM [REMOVED_ZH:4]。

    Parameters
    ----------
    input_dim  : LSTM [REMOVED_ZH:4] (spatial + env + time [REMOVED_ZH:3])
    hidden_dim : LSTM [REMOVED_ZH:4]
    n_layers   : LSTM [REMOVED_ZH:2]
    dropout    : dropout rate
    """
    def __init__(self,
                 input_dim:  int,
                 hidden_dim: int = 256,
                 n_layers:   int = 1,
                 dropout:    float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        # Warming-up MLP: h₀ = MLP([REMOVED_ZH:2] RGCN [REMOVED_ZH:2])
        self.warmup_h = nn.Linear(hidden_dim, hidden_dim)
        self.warmup_c = nn.Linear(hidden_dim, hidden_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self,
                 x_seq:    Tensor,
                 h0_feat:  Tensor = None) -> Tensor:
        """
        x_seq   : (N_nodes, T, input_dim) [REMOVED_ZH:4]
        h0_feat : (N_nodes, hidden_dim) [REMOVED_ZH:3] RGCN [REMOVED_ZH:5] h₀
        return  : (N_nodes, T, hidden_dim)
        """
        N = x_seq.size(0)
        if h0_feat is not None:
            # [REMOVED_ZH:2] Eq.9: h₀ = MLP_h0(h^RGCN_{v,t0})
            h0 = self.warmup_h(h0_feat).tanh().unsqueeze(0).repeat(self.n_layers, 1, 1)
            c0 = self.warmup_c(h0_feat).tanh().unsqueeze(0).repeat(self.n_layers, 1, 1)
        else:
            h0 = torch.zeros(self.n_layers, N, self.hidden_dim, device=x_seq.device)
            c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x_seq, (h0, c0))
        return self.drop(out)