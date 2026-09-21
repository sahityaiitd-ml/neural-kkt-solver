"""
loss.py
=======
Powell-Hestenes-Rockafellar (PHR) Augmented Lagrangian & KKT Loss Engine for Iteration 5.

Key Mathematical Innovations:
1. Powell-Hestenes-Rockafellar (PHR) Augmented Lagrangian:
   - Replaces exterior quadratic penalty walls with the exact ALM penalty:
       g_i = (G x - h)_i
       v_i = relu(lambda_i + rho_i * g_i)
       L_ALM = sum_i (1 / (2 * rho_i)) * (v_i^2 - lambda_i^2)
   - Guarantees zero boundary penetration at finite penalty weights rho by using
     predicted dual multipliers to counteract the outward cost gradient.
2. Row-Scaled Adaptive Penalty (rho_i):
   - Row-wise adaptive penalty parameters rho_i = rho_0 / max(1e-4, ||G_i||_2)
     preventing gradient distortion across unbalanced constraints.
3. Dual-Aware Objective Direction Pull:
   - Detects whether the polytope is Profit-Driven (c < 0) or Demand-Driven (c > 0).
   - Injects directional pull that breaks both negative-cost and positive-cost origin traps.
4. One-Sided Strong Duality Gap:
   - L_gap = [relu(c^T x + h^T lambda)]^2, eliminating restoring force toward the origin.
5. Fischer-Burmeister Complementarity:
   - phi_eps = lambda + s - sqrt(lambda^2 + s^2 + eps).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


def fischer_burmeister_loss(
    lambda_hat: torch.Tensor,
    slack: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes mean squared Fischer-Burmeister residual:
    phi_eps = lambda + s - sqrt(lambda^2 + s^2 + eps)
    """
    phi = lambda_hat + slack - torch.sqrt(lambda_hat ** 2 + slack ** 2 + eps)
    return torch.mean(phi ** 2)


def iteration_5_kkt_loss(
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    c_norm: torch.Tensor,
    G_norm: torch.Tensor,
    h_norm: torch.Tensor,
    rho_per_row: torch.Tensor,
    epoch: int = 1,
    effective_gamma: float = 0.996,
    is_positive_cost: bool = False,
    w_obj_init: float = 0.5,
    w_stat: float = 1.0,
    w_gap: float = 1.0,
    w_fb: float = 2.0,
    w_alm: float = 1.0,
    w_x_pos: float = 5.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates Iteration 5 PHR Augmented Lagrangian and KKT loss.
    """
    # 1. Normalized Slack & Violation
    Gx_norm = torch.matmul(G_norm, x_hat)
    slack_norm = h_norm - Gx_norm
    g = -slack_norm  # Constraint violation: G x - h

    # 2. Stationarity Residual: c + G^T * lambda
    grad_L = c_norm + torch.matmul(G_norm.T, lambda_hat)
    loss_stat = torch.mean(grad_L ** 2)

    # 3. Annealed Objective Pull
    w_obj = w_obj_init * (effective_gamma ** epoch)
    loss_obj = torch.dot(c_norm, x_hat)

    if is_positive_cost:
        loss_dual_obj = torch.dot(h_norm, lambda_hat)
        loss_obj_term = w_obj * loss_obj + 0.1 * (effective_gamma ** epoch) * loss_dual_obj
    else:
        loss_obj_term = w_obj * loss_obj

    # 4. Pure ReLU One-Sided Strong Duality Gap
    duality_gap = loss_obj + torch.dot(h_norm, lambda_hat)
    loss_gap = torch.relu(duality_gap) ** 2

    # 5. Fischer-Burmeister Complementarity
    loss_fb = fischer_burmeister_loss(lambda_hat, slack_norm, eps=eps)

    # 6. Powell-Hestenes-Rockafellar (PHR) Augmented Lagrangian
    v = torch.relu(lambda_hat + rho_per_row * g)
    loss_alm = torch.mean((1.0 / (2.0 * rho_per_row)) * (v ** 2 - lambda_hat ** 2))

    # 7. Non-negativity
    loss_x_pos = torch.mean(torch.relu(-x_hat) ** 2)

    # Total loss
    total_loss = (
        loss_obj_term +
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_alm * loss_alm +
        w_x_pos * loss_x_pos
    )

    # Pure KKT Residual Metric for Checkpointing
    viol_raw = torch.relu(g)
    kkt_residual = (
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        10.0 * torch.mean(viol_raw ** 2) +
        w_x_pos * loss_x_pos
    )

    max_primal_violation = float(torch.max(viol_raw).item())

    metrics = {
        "loss_total": float(total_loss.item()),
        "kkt_residual": float(kkt_residual.item()),
        "loss_obj": float(loss_obj.item()),
        "loss_stat": float(loss_stat.item()),
        "loss_gap": float(loss_gap.item()),
        "loss_fb": float(loss_fb.item()),
        "loss_alm": float(loss_alm.item()),
        "loss_x_pos": float(loss_x_pos.item()),
        "duality_gap": float(duality_gap.item()),
        "max_primal_violation": max_primal_violation
    }

    return total_loss, metrics
