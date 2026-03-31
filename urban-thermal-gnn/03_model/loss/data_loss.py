"""03_model/loss/data_loss.py — MSE + [REMOVED_ZH:4] loss"""
import torch
import torch.nn.functional as F
from torch import Tensor


def data_loss(pred: Tensor, target: Tensor,
               quality_weights: Tensor = None) -> Tensor:
    """
    [REMOVED_ZH:2] MSE。
    pred/target : (N_air, T)
    quality_weights : (T,) [REMOVED_ZH:4] [0,1]，None=[REMOVED_ZH:3]
    """
    loss = F.mse_loss(pred, target, reduction="none")   # (N, T)
    if quality_weights is not None:
        w = quality_weights.view(1, -1).to(pred.device)
        loss = (loss * w).sum() / (w.sum() * loss.shape[0] + 1e-9)
    else:
        loss = loss.mean()
    return loss


def sensor_supervision_loss(pred:    Tensor,
                              sensor_utci: Tensor,
                              mask:    Tensor) -> Tensor:
    """
    [REMOVED_ZH:5] UTCI [REMOVED_ZH:2]。
    mask: (N_air, T) bool，True=[REMOVED_ZH:7]
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], sensor_utci[mask])