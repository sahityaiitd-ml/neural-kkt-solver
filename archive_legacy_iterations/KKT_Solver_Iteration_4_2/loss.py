"""
loss.py
=======
Annealed Objective Pull & One-Sided Duality Gap Loss (Iteration 4.2).

Key Mathematical Formulations:
------------------------------
1. Annealed Objective Pull (w_obj * gamma^t * c_tilde^T * x):
   Provides an initial downward gradient that expels the solver from the origin (x=0)
   basin, then smoothly decays to zero so the KKT boundary conditions govern final
   convergence without persistent distorting gravity.
2. One-Sided Strong Duality Gap (w_gap * ReLU(c_tilde^T * x + h_tilde^T * lambda)^2):
   Weak duality states that c^T x + h^T lambda >= 0 for feasible primal-dual pairs.
   Penalizing ONLY the positive gap eliminates the restoring spring trap that previously
   locked solutions at c_tilde^T x = -0.10.
3. Normalized Fischer-Burmeister Complementarity:
   Enforces lambda_i * s_i = 0 with smooth, symmetric non-exploding gradients.
4. Rigid Primal and Non-Negativity Walls:
   Prevents boundary penetration into infeasible regions.
5. Exact KKT Residual Tracking:
   Tracks model checkpointing on pure KKT satisfaction (feasibility + stationarity + complementarity),
   preventing selection of infeasible solutions with artificially low objectives.
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


def iteration_4_2_kkt_loss(
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    c_norm: torch.Tensor,
    G_norm: torch.Tensor,
    h_norm: torch.Tensor,
    epoch: int = 1,
    w_obj_init: float = 0.5,
    gamma: float = 0.996,
    w_stat: float = 1.0,
    w_gap: float = 1.0,
    w_fb: float = 2.0,
    w_prim: float = 10.0,
    w_x_pos: float = 5.0,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Evaluates Iteration 4.2 loss function.
    """
    # 1. Normalized Slack: s_tilde = h_tilde - G_tilde * x
    Gx_norm = torch.matmul(G_norm, x_hat)
    slack_norm = h_norm - Gx_norm

    # 2. Annealed Objective Pull: w_obj(t) = w_obj_init * (gamma ^ epoch)
    current_w_obj = w_obj_init * (gamma ** epoch) if w_obj_init > 0 else 0.0
    loss_obj = torch.dot(c_norm, x_hat)

    # 3. Dual Stationarity: c_tilde + G_tilde^T * lambda_tilde = 0
    grad_L = c_norm + torch.matmul(G_norm.T, lambda_hat)
    loss_stat = torch.mean(grad_L ** 2)

    # 4. One-Sided Strong Duality Gap: ReLU(c_tilde^T * x + h_tilde^T * lambda_tilde)^2
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

    # Total Optimization Loss (includes decaying exploration pull)
    total_loss = (
        current_w_obj * loss_obj +
        w_stat * loss_stat +
        w_gap * loss_gap +
        w_fb * loss_fb +
        w_prim * loss_prim +
        w_x_pos * loss_x_pos
    )

    # Pure KKT Residual Metric (unbiased metric for checkpointing best model)
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


class KKTLoss(nn.Module):
    def __init__(
        self,
        c_norm: torch.Tensor,
        G_norm: torch.Tensor,
        h_norm: torch.Tensor,
        w_obj_init: float = 0.5,
        gamma: float = 0.996,
        w_stat: float = 1.0,
        w_gap: float = 1.0,
        w_fb: float = 2.0,
        w_prim: float = 10.0,
        w_x_pos: float = 5.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.register_buffer("c_norm", c_norm)
        self.register_buffer("G_norm", G_norm)
        self.register_buffer("h_norm", h_norm)
        self.w_obj_init = w_obj_init
        self.gamma = gamma
        self.w_stat = w_stat
        self.w_gap = w_gap
        self.w_fb = w_fb
        self.w_prim = w_prim
        self.w_x_pos = w_x_pos
        self.eps = eps

    def forward(
        self,
        x_hat: torch.Tensor,
        lambda_hat: torch.Tensor,
        epoch: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return iteration_4_2_kkt_loss(
            x_hat, lambda_hat,
            self.c_norm, self.G_norm, self.h_norm,
            epoch=epoch,
            w_obj_init=self.w_obj_init,
            gamma=self.gamma,
            w_stat=self.w_stat,
            w_gap=self.w_gap,
            w_fb=self.w_fb,
            w_prim=self.w_prim,
            w_x_pos=self.w_x_pos,
            eps=self.eps
        )
