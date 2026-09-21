"""
loss.py
=======
Equilibrated KKT Loss with Linear Objective Pull & Strong Duality (Iteration 4).

Mathematical Innovations:
-------------------------
1. Linear Primal Objective Pull (w_obj * c_tilde^T * x_hat):
   Provides a persistent, constant gradient grad_x L_obj = c_tilde that never vanishes
   at the origin (x = 0). This actively expels the solution from the trivial interior
   trap into the optimal boundary face.
2. Strong Duality Gap (w_gap * (c_tilde^T * x_hat + h_tilde^T * lambda_tilde)^2):
   Drives the primal cost and dual lower bound toward mutual convergence.
3. Normalized Fischer-Burmeister C-function on O(1) slacks:
   Smoothly enforces complementary slackness without gradient explosion.
4. Direct Primal and Non-Negativity Penalties:
   Acts as rigid elastic walls preventing the solution from penetrating past the feasible domain.
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
    w_obj: float = 0.2,
    w_stat: float = 1.0,
    w_gap: float = 1.0,
    w_fb: float = 2.0,
    w_prim: float = 15.0,
    w_x_pos: float = 5.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates Iteration 4.1 loss function.
    """
    # 1. Normalized Slack: s_tilde = h_tilde - G_tilde * x
    Gx_norm = torch.matmul(G_norm, x_hat)
    slack_norm = h_norm - Gx_norm

    # 2. Linear Primal Objective Pull: c_tilde^T * x_hat
    # Constant gradient grad_x = c_tilde forces x away from the origin (x=0)
    loss_obj = torch.dot(c_norm, x_hat)

    # 3. Stationarity: c_tilde + G_tilde^T * lambda_tilde = 0
    grad_L = c_norm + torch.matmul(G_norm.T, lambda_hat)
    loss_stat = torch.mean(grad_L ** 2)

    # 4. Strong Duality Gap: (c_tilde^T * x + h_tilde^T * lambda_tilde)^2
    duality_gap = loss_obj + torch.dot(h_norm, lambda_hat)
    loss_gap = duality_gap ** 2

    # 5. Normalized Fischer-Burmeister Complementarity
    loss_fb = fischer_burmeister_loss(lambda_hat, slack_norm, eps=eps)

    # 6. Direct Primal Feasibility Violation (Penalty Walls)
    primal_viol = torch.relu(-slack_norm)
    loss_prim = torch.mean(primal_viol ** 2)
    max_primal_violation = float(torch.max(primal_viol).item())

    # 7. Primal Variable Non-Negativity: x >= 0
    loss_x_pos = torch.mean(torch.relu(-x_hat) ** 2)

    # Total Weighted Loss
    total_loss = (
        w_obj * loss_obj +
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_x_pos * loss_x_pos
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_obj": float(loss_obj.item()),
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
        w_obj: float = 0.2,
        w_stat: float = 1.0,
        w_gap: float = 1.0,
        w_fb: float = 2.0,
        w_prim: float = 15.0,
        w_x_pos: float = 5.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.register_buffer("c_norm", c_norm)
        self.register_buffer("G_norm", G_norm)
        self.register_buffer("h_norm", h_norm)
        self.w_obj = w_obj
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
            w_obj=self.w_obj,
            w_stat=self.w_stat,
            w_gap=self.w_gap,
            w_fb=self.w_fb,
            w_prim=self.w_prim,
            w_x_pos=self.w_x_pos,
            eps=self.eps
        )


# Backward compatibility alias
iteration_4_1_kkt_loss = iteration_4_kkt_loss
