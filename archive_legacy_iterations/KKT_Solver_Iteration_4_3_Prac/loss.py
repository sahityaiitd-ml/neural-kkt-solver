"""
loss.py
=======
Modular Loss Module for KKT_Solver_Iteration_4_3_Prac.

Provides:
1. Configurable Annealing Schedules:
   - "fixed_decay": Standard geometric decay w_obj * gamma^t.
   - "diameter_adaptive": Adjusts decay based on estimated polytope span so wide
     polytopes maintain sustained exploration until boundary engagement.
2. One-Sided Strong Duality Gap:
   - "one_sided_relu": ReLU(c_tilde^T x + h_tilde^T lambda)^2. Strictly zero restoring force.
3. Normalized Fischer-Burmeister Complementarity.
4. Rigid Primal and Non-Negativity Boundary Penalties.
5. Pure KKT Residual Metric for Model Checkpointing.
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


def compute_annealed_weight(
    epoch: int,
    w_obj_init: float,
    gamma: float,
    anneal_mode: str,
    estimated_diameter: float = 1.0
) -> float:
    """Computes dynamic objective pull weight w_obj(t)."""
    if w_obj_init <= 0:
        return 0.0

    if anneal_mode.lower() == "diameter_adaptive":
        # For wide polytopes, slow down decay so variables travel across interior
        scale_factor = max(1.0, (estimated_diameter / 2.0) ** 0.5)
        effective_gamma = 1.0 - (1.0 - gamma) / scale_factor
        return float(w_obj_init * (effective_gamma ** epoch))
    else:
        # Standard fixed geometric decay
        return float(w_obj_init * (gamma ** epoch))


def iteration_4_3_kkt_loss(
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    c_norm: torch.Tensor,
    G_norm: torch.Tensor,
    h_norm: torch.Tensor,
    epoch: int = 1,
    w_obj_init: float = 0.5,
    gamma: float = 0.996,
    anneal_mode: str = "fixed_decay",
    estimated_diameter: float = 1.0,
    gap_mode: str = "one_sided_relu",
    w_stat: float = 1.0,
    w_gap: float = 1.0,
    w_fb: float = 2.0,
    w_prim: float = 10.0,
    w_x_pos: float = 5.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates Iteration 4.3 loss function.
    """
    # 1. Normalized Slack: s_tilde = h_tilde - G_tilde * x
    Gx_norm = torch.matmul(G_norm, x_hat)
    slack_norm = h_norm - Gx_norm

    # 2. Annealed Objective Pull
    current_w_obj = compute_annealed_weight(
        epoch=epoch,
        w_obj_init=w_obj_init,
        gamma=gamma,
        anneal_mode=anneal_mode,
        estimated_diameter=estimated_diameter
    )
    loss_obj = torch.dot(c_norm, x_hat)

    # 3. Dual Stationarity: c_tilde + G_tilde^T * lambda_tilde = 0
    grad_L = c_norm + torch.matmul(G_norm.T, lambda_hat)
    loss_stat = torch.mean(grad_L ** 2)

    # 4. One-Sided Strong Duality Gap
    duality_gap = loss_obj + torch.dot(h_norm, lambda_hat)
    loss_gap = torch.relu(duality_gap) ** 2

    # 5. Normalized Fischer-Burmeister Complementarity
    loss_fb = fischer_burmeister_loss(lambda_hat, slack_norm, eps=eps)

    # 6. Direct Primal Feasibility Violation (Penalty Walls)
    primal_viol = torch.relu(-slack_norm)
    loss_prim = torch.mean(primal_viol ** 2)
    max_primal_violation = float(torch.max(primal_viol).item())

    # 7. Primal Variable Non-Negativity: x >= 0
    x_neg = torch.relu(-x_hat)
    loss_x_pos = torch.mean(x_neg ** 2)

    # Total Optimization Loss
    total_loss = (
        current_w_obj * loss_obj +
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_x_pos * loss_x_pos
    )

    # Pure KKT Residual Metric (for model checkpointing)
    kkt_residual = (
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_x_pos * loss_x_pos
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "kkt_residual": float(kkt_residual.item()),
        "loss_obj": float(loss_obj.item()),
        "w_obj_current": float(current_w_obj),
        "loss_stat": float(loss_stat.item()),
        "loss_gap": float(loss_gap.item()),
        "loss_fb": float(loss_fb.item()),
        "loss_prim": float(loss_prim.item()),
        "loss_x_pos": float(loss_x_pos.item()),
        "duality_gap": float(duality_gap.item()),
        "max_primal_violation": max_primal_violation
    }

    return total_loss, metrics
