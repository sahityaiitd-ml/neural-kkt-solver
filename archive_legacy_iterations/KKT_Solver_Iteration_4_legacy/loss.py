"""
loss.py
=======
Equilibrated KKT Loss with Strong Duality Gap (Iteration 4).

Mathematical Formulation:
-------------------------
Operates on preconditioned, unit-row-normalized matrices (G_tilde, h_tilde, c_tilde).

Components:
1. Stationarity (First-Order Optimality):
       L_stat = (1/n) * ||c_tilde + G_tilde^T * lambda_tilde||_2^2
2. Strong Duality Theorem Gap:
       L_gap = (c_tilde^T * x_hat + h_tilde^T * lambda_tilde)^2
   Directly pulls primal variables x_hat toward the cost gradient c_tilde while
   pushing dual multipliers lambda_tilde to tighten the dual lower bound.
3. Smoothed Fischer-Burmeister (Complementarity & Feasibility):
       phi_eps(lambda_tilde_i, s_tilde_i) = lambda_tilde_i + s_tilde_i - sqrt(lambda_tilde_i^2 + s_tilde_i^2 + eps)
       L_fb = (1/m) * sum(phi_eps^2)
   Since slacks s_tilde are row-normalized to O(1), loss values never explode.
4. Direct Primal Feasibility Penalty:
       L_prim = (1/m) * ||max(0, -s_tilde)||_2^2
5. Primal Non-Negativity:
       L_x_pos = (1/n) * ||max(0, -x_hat)||_2^2
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
    Computes mean squared Fischer-Burmeister residual:
    phi_eps = lambda + s - sqrt(lambda^2 + s^2 + eps)
    """
    phi = lambda_hat + slack - torch.sqrt(lambda_hat ** 2 + slack ** 2 + eps)
    return torch.mean(phi ** 2)


def iteration_4_kkt_loss(
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    c_norm: torch.Tensor,
    G_norm: torch.Tensor,
    h_norm: torch.Tensor,
    w_stat: float = 1.0,
    w_gap: float = 2.0,
    w_fb: float = 2.0,
    w_prim: float = 2.0,
    w_x_pos: float = 2.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates the Iteration 4 Preconditioned KKT Loss with Strong Duality coupling.
    """
    # 1. Normalized Slack: s_tilde = h_tilde - G_tilde * x
    Gx_norm = torch.matmul(G_norm, x_hat)
    slack_norm = h_norm - Gx_norm

    # 2. Stationarity: c_tilde + G_tilde^T * lambda_tilde = 0
    grad_L = c_norm + torch.matmul(G_norm.T, lambda_hat)
    loss_stat = torch.mean(grad_L ** 2)

    # 3. Strong Duality Gap: (c_tilde^T * x + h_tilde^T * lambda_tilde)^2
    primal_obj_norm = torch.dot(c_norm, x_hat)
    dual_obj_norm = -torch.dot(h_norm, lambda_hat)
    duality_gap = primal_obj_norm - dual_obj_norm  # c_norm^T x + h_norm^T lambda
    loss_gap = duality_gap ** 2

    # 4. Normalized Fischer-Burmeister Complementarity
    loss_fb = fischer_burmeister_loss(lambda_hat, slack_norm, eps=eps)

    # 5. Direct Primal Violation
    primal_viol = torch.relu(-slack_norm)
    loss_prim = torch.mean(primal_viol ** 2)
    max_primal_violation = float(torch.max(primal_viol).item())

    # 6. Primal Variable Non-Negativity: x >= 0
    loss_x_pos = torch.mean(torch.relu(-x_hat) ** 2)

    # Total Weighted Loss
    total_loss = (
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_x_pos * loss_x_pos
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_stat": float(loss_stat.item()),
        "loss_gap": float(loss_gap.item()),
        "loss_fb": float(loss_fb.item()),
        "loss_prim": float(loss_prim.item()),
        "loss_x_pos": float(loss_x_pos.item()),
        "duality_gap": float(duality_gap.item()),
        "max_primal_violation": max_primal_violation
    }

    return total_loss, metrics


class KKTLoss(nn.Module):
    def __init__(
        self,
        c_norm: torch.Tensor,
        G_norm: torch.Tensor,
        h_norm: torch.Tensor,
        w_stat: float = 1.0,
        w_gap: float = 2.0,
        w_fb: float = 2.0,
        w_prim: float = 2.0,
        w_x_pos: float = 2.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.register_buffer("c_norm", c_norm)
        self.register_buffer("G_norm", G_norm)
        self.register_buffer("h_norm", h_norm)
        self.w_stat = w_stat
        self.w_gap = w_gap
        self.w_fb = w_fb
        self.w_prim = w_prim
        self.w_x_pos = w_x_pos
        self.eps = eps

    def forward(
        self,
        x_hat: torch.Tensor,
        lambda_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return iteration_4_kkt_loss(
            x_hat, lambda_hat,
            self.c_norm, self.G_norm, self.h_norm,
            w_stat=self.w_stat,
            w_gap=self.w_gap,
            w_fb=self.w_fb,
            w_prim=self.w_prim,
            w_x_pos=self.w_x_pos,
            eps=self.eps
        )
