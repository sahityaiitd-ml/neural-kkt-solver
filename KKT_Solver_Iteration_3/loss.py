"""
loss.py
=======
Fischer-Burmeister KKT Loss Formulation (Iteration 3 - Path B).

Reference:
----------
Ashfaq Iftakher, Rahul Golder, M. M. Faruque Hasan (Texas A&M University).
"Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints"
arXiv:2507.08124v1 (July 2025).

Mathematical Formulation:
-------------------------
For inequality constraints G x <= h, slack is defined as:
    s = h - G x

The smoothed Fischer-Burmeister (FB) complementarity function is:
    phi_eps(lambda_i, s_i) = lambda_i + s_i - sqrt(lambda_i^2 + s_i^2 + eps)

Theorem:
    phi_eps(lambda_i, s_i) = 0  <=>  lambda_i >= 0, s_i >= 0, lambda_i * s_i = 0
The single FB operator unifies primal feasibility, dual feasibility, and
complementary slackness into a continuous, differentiable loss term.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def fischer_burmeister_loss(
    lambda_hat: torch.Tensor,
    slack: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes mean squared Fischer-Burmeister residual across all constraints.
    phi_eps = lambda + s - sqrt(lambda^2 + s^2 + eps)
    """
    phi = lambda_hat + slack - torch.sqrt(lambda_hat ** 2 + slack ** 2 + eps)
    return torch.mean(phi ** 2)


def basic_kkt_loss(
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    c: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    w_stat: float = 1.0,
    w_fb: float = 5.0,
    w_prim: float = 1.0,
    w_primal_pos: float = 5.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates the Iteration 3 KKT Loss using the Fischer-Burmeister C-function.

    Parameters:
    -----------
    x_hat : Tensor of shape (n,)
        Primal decision variables.
    lambda_hat : Tensor of shape (m,)
        Dual multipliers (strictly non-negative via Softplus).
    c : Tensor of shape (n,)
        Cost vector.
    G : Tensor of shape (m, n)
        Inequality constraint matrix.
    h : Tensor of shape (m,)
        Constraint right-hand side.
    w_stat : float, default 1.0
        Weight for stationarity condition.
    w_fb : float, default 5.0
        Weight for Fischer-Burmeister complementarity & feasibility condition.
    w_prim : float, default 1.0
        Direct quadratic penalty weight for boundary feasibility.
    w_primal_pos : float, default 5.0
        Weight for primal non-negativity (x >= 0).
    eps : float, default 1e-6
        Smoothing parameter for Fischer-Burmeister square root.
    """
    # 1. Slack computation: s = h - G x
    Gx = torch.matmul(G, x_hat)
    slack = h - Gx  # s >= 0 when feasible, s < 0 when violated

    # 2. Stationarity Condition: grad_x L = c + G^T lambda = 0
    grad_x = c + torch.matmul(G.T, lambda_hat)
    loss_stat = torch.mean(grad_x ** 2)

    # 3. Fischer-Burmeister Unified Complementarity & Feasibility
    loss_fb = fischer_burmeister_loss(lambda_hat, slack, eps=eps)

    # 4. Direct Primal Violation (for diagnostic logging and boundary push)
    # violation occurs when Gx - h > 0, which is -slack > 0
    primal_viol = torch.relu(-slack)
    loss_prim = torch.mean(primal_viol ** 2)
    max_primal_violation = float(torch.max(primal_viol).item())

    # 5. Traditional Complementary Slackness (for diagnostic logging)
    loss_slack = torch.mean((lambda_hat * slack) ** 2)

    # 6. Primal Non-Negativity: x >= 0
    loss_primal_pos = torch.mean(torch.relu(-x_hat) ** 2)

    # Total Weighted Loss
    total_loss = (
        w_stat * loss_stat +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_primal_pos * loss_primal_pos
    )

    metrics = {
        "loss_stat": float(loss_stat.item()),
        "loss_fb": float(loss_fb.item()),
        "loss_prim": float(loss_prim.item()),
        "loss_slack": float(loss_slack.item()),
        "loss_primal_pos": float(loss_primal_pos.item()),
        "max_primal_violation": max_primal_violation,
    }

    return total_loss, metrics


class KKTLoss(nn.Module):
    def __init__(
        self,
        c: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        w_stat: float = 1.0,
        w_fb: float = 5.0,
        w_prim: float = 1.0,
        w_primal_pos: float = 5.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.register_buffer("c", c)
        self.register_buffer("G", G)
        self.register_buffer("h", h)
        self.w_stat = w_stat
        self.w_fb = w_fb
        self.w_prim = w_prim
        self.w_primal_pos = w_primal_pos
        self.eps = eps

    def forward(self, x_hat: torch.Tensor, lambda_hat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        return basic_kkt_loss(
            x_hat, lambda_hat, self.c, self.G, self.h,
            w_stat=self.w_stat,
            w_fb=self.w_fb,
            w_prim=self.w_prim,
            w_primal_pos=self.w_primal_pos,
            eps=self.eps
        )
