"""
03_model/loss/physics_penalty.py
[REMOVED_ZH:5] (Physics-Informed Loss)
════════════════════════════════════════════════════════════════
[REMOVED_ZH:4]Constraint Penalty，[REMOVED_ZH:13]：

L_total = L_data + λ₁·L_radiation + λ₂·L_wind + λ₃·L_temporal

1. L_radiation : Solar Radiation Node MRT ≥ Ta + δ_min (ISO 7726)
                 Shadow Node MRT ≤ Ta + δ_max
2. L_wind      : [REMOVED_ZH:5] leeward [REMOVED_ZH:2]Wind Speed[REMOVED_ZH:5]
3. L_temporal  : [REMOVED_ZH:4] UTCI [REMOVED_ZH:6] (≤ 5°C/hr)
"""
import torch
from torch import Tensor


def radiation_penalty(utci_pred: Tensor,
                        svf:       Tensor,
                        in_shadow: Tensor,
                        sol_alt:   Tensor,
                        lambda1:   float = 0.1) -> Tensor:
    """
    L_radiation: [REMOVED_ZH:4] vs Shadow[REMOVED_ZH:1] UTCI [REMOVED_ZH:8]。

    utci_pred : (N, T) [REMOVED_ZH:5] UTCI
    svf       : (N,)   [REMOVED_ZH:6]
    in_shadow : (N, T) bool
    sol_alt   : (T,)   [REMOVED_ZH:5] [deg]

    [REMOVED_ZH:4]: [REMOVED_ZH:4]，Solar Radiation Node UTCI [REMOVED_ZH:1] > Shadow Node UTCI
    [REMOVED_ZH:2]: max(0, UTCI_shadow - UTCI_sun + margin)^2
    """
    if utci_pred.shape[1] == 0:
        return torch.tensor(0.0, device=utci_pred.device)

    # [REMOVED_ZH:6] (sol_alt > 10°)
    sun_mask = (sol_alt > 10.0).to(utci_pred.device)   # (T,)
    if not sun_mask.any():
        return torch.tensor(0.0, device=utci_pred.device)

    penalty_sum = torch.tensor(0.0, device=utci_pred.device)
    n_valid = 0

    for t_idx in range(utci_pred.shape[1]):
        if not sun_mask[t_idx]:
            continue
        shd  = in_shadow[:, t_idx]            # (N,) bool
        utci = utci_pred[:, t_idx]             # (N,)

        sun_nodes   = (~shd) & (svf > 0.5)
        shade_nodes = shd & (svf < 0.5)

        if sun_nodes.any() and shade_nodes.any():
            u_sun   = utci[sun_nodes].mean()
            u_shade = utci[shade_nodes].mean()
            # [REMOVED_ZH:4]Shadow[REMOVED_ZH:1] ([REMOVED_ZH:5] 0.5 [REMOVED_ZH:1] std [REMOVED_ZH:2])
            margin  = 0.5
            penalty = torch.relu(u_shade - u_sun + margin) ** 2
            penalty_sum = penalty_sum + penalty
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=utci_pred.device)
    return lambda1 * penalty_sum / n_valid


def temporal_smoothness_penalty(utci_pred: Tensor,
                                  lambda2:   float = 0.05,
                                  max_delta: float = 0.625) -> Tensor:
    """
    L_temporal: [REMOVED_ZH:4] UTCI [REMOVED_ZH:6] 5°C/hr。
    ([REMOVED_ZH:4] 5/8=0.625 [REMOVED_ZH:1] std)

    utci_pred : (N, T)
    """
    if utci_pred.shape[1] < 2:
        return torch.tensor(0.0, device=utci_pred.device)

    delta  = utci_pred[:, 1:] - utci_pred[:, :-1]   # (N, T-1)
    excess = torch.relu(delta.abs() - max_delta)
    return lambda2 * excess.mean()


def wind_obstruction_penalty(utci_pred: Tensor,
                               bldg_height: Tensor,
                               lambda3:    float = 0.05,
                               height_thresh: float = 0.4) -> Tensor:
    """
    L_wind: [REMOVED_ZH:8] UTCI [REMOVED_ZH:11]。
    [REMOVED_ZH:2] (bh > height_thresh * 50m = 20m) [REMOVED_ZH:5] UTCI
    [REMOVED_ZH:7] + 1σ [REMOVED_ZH:3]。

    utci_pred   : (N, T)
    bldg_height : (N,) [REMOVED_ZH:6] ([REMOVED_ZH:3]，0–1)
    """
    near_tall = bldg_height > height_thresh    # (N,)
    if not near_tall.any():
        return torch.tensor(0.0, device=utci_pred.device)

    utci_mean = utci_pred.mean()
    utci_std  = utci_pred.std().clamp(min=0.01)
    upper_bound = utci_mean + 1.5 * utci_std

    u_near = utci_pred[near_tall]              # (N_near, T)
    excess = torch.relu(u_near - upper_bound)
    return lambda3 * excess.mean()


def total_physics_loss(utci_pred:   Tensor,
                        svf:         Tensor,
                        in_shadow:   Tensor,
                        sol_alt:     Tensor,
                        bldg_height: Tensor,
                        lambdas:     dict = None) -> Tensor:
    """
    [REMOVED_ZH:8]。
    lambdas: {'lambda1':0.1, 'lambda2':0.05, 'lambda3':0.05}
    """
    lam = lambdas or {"lambda1": 0.1, "lambda2": 0.05, "lambda3": 0.05}

    l_rad  = radiation_penalty(utci_pred, svf, in_shadow, sol_alt,
                                 lam["lambda1"])
    l_temp = temporal_smoothness_penalty(utci_pred, lam["lambda2"])
    l_wind = wind_obstruction_penalty(utci_pred, bldg_height, lam["lambda3"])

    return l_rad + l_temp + l_wind