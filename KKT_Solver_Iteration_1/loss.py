"""
loss.py
=======
Implements the Physics-Informed Karush-Kuhn-Tucker (KKT) Loss Function directly from
the literature (Carmine Delle Femine 2024 & KKT-Nets).

The 3 Laws Enforced:
--------------------
1. Stationarity:          ||c + G^T * lambda||^2 = 0   (Force balance on gradients)
2. Primal Feasibility:    ||max(0, G*x - h)||^2  = 0   (Staying inside the boundaries)
3. Complementary Slack:   ||lambda * (G*x - h)||^2 = 0 (Inactive constraints have lambda = 0)

Dual Feasibility (lambda >= 0) is mathematically guaranteed by the Softplus activation
in model.py, so its error is automatically zero.
"""

from typing import Tuple, Dict
import torch


def basic_kkt_loss(
    c: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    w_stat: float = 1.0,
    w_prim: float = 5.0,
    w_slack: float = 2.0,
    w_dual_pos: float = 5.0,
    w_primal_pos: float = 5.0,
    enforce_primal_positive: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the scalar physics-informed KKT loss and logs each component.

    Parameters:
    -----------
    c : torch.Tensor of shape (n,)
        Cost vector from the canonical problem.
    G : torch.Tensor of shape (m, n)
        Constraint matrix for G * x <= h.
    h : torch.Tensor of shape (m,)
        RHS boundary limits vector.
    x_hat : torch.Tensor of shape (n,)
        Predicted primal decision variables from the neural net.
    lambda_hat : torch.Tensor of shape (m,)
        Raw predicted dual multipliers from the neural net.
    w_stat, w_prim, w_slack, w_dual_pos, w_primal_pos : float
        Hyperparameter weights balancing the physical KKT conditions.
    enforce_primal_positive : bool, default True
        Whether to penalize negative decision variables (x < 0).
    normalize : bool, default True
        Whether to divide by (1 + ||c||) and (1 + ||h||) to make the loss scale-independent.

    Returns:
    --------
    total_loss : torch.Tensor
        Scalar differentiable loss for PyTorch backpropagation.
    metrics : Dict[str, float]
        Raw numerical values of each error for logging and plotting.
    """
    # --------------------------------------------------------------------------
    # 1. Stationarity: r_S = c + G^T * lambda = 0
    # --------------------------------------------------------------------------
    grad_lagrangian = c + torch.matmul(G.T, lambda_hat)
    loss_stationarity = torch.sum(grad_lagrangian ** 2)

    # --------------------------------------------------------------------------
    # 2. Primal Feasibility: max(0, G*x - h) = 0
    # --------------------------------------------------------------------------
    slack = torch.matmul(G, x_hat) - h
    primal_violations = torch.relu(slack)
    loss_primal = torch.sum(primal_violations ** 2)

    # --------------------------------------------------------------------------
    # 3. Complementary Slackness: lambda * (G*x - h) = 0
    # --------------------------------------------------------------------------
    slackness_product = lambda_hat * slack
    loss_slackness = torch.sum(slackness_product ** 2)

    # --------------------------------------------------------------------------
    # 4. Dual Feasibility Condition: lambda >= 0  ==>  max(0, -lambda) = 0
    # --------------------------------------------------------------------------
    dual_negativity = torch.relu(-lambda_hat)
    loss_dual_pos = torch.sum(dual_negativity ** 2)

    # --------------------------------------------------------------------------
    # 5. Primal Non-Negativity Condition: x >= 0  ==>  max(0, -x) = 0
    # --------------------------------------------------------------------------
    primal_negativity = torch.relu(-x_hat) if enforce_primal_positive else torch.zeros_like(x_hat)
    loss_primal_pos = torch.sum(primal_negativity ** 2)

    # --------------------------------------------------------------------------
    # Optional Normalization: Prevent large problems from dominating
    # --------------------------------------------------------------------------
    scale_c = (1.0 + torch.sum(c ** 2)) if normalize else 1.0
    scale_h = (1.0 + torch.sum(h ** 2)) if normalize else 1.0

    norm_loss_stat = loss_stationarity / scale_c
    norm_loss_prim = loss_primal / scale_h
    norm_loss_slack = loss_slackness / (scale_c * scale_h).sqrt() if normalize else loss_slackness
    norm_loss_dual = loss_dual_pos / scale_c
    norm_loss_primal_pos = loss_primal_pos / scale_c

    # Total weighted combination
    total_loss = (
        w_stat * norm_loss_stat
        + w_prim * norm_loss_prim
        + w_slack * norm_loss_slack
        + w_dual_pos * norm_loss_dual
        + w_primal_pos * norm_loss_primal_pos
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_stat": float(loss_stationarity.item()),
        "loss_prim": float(loss_primal.item()),
        "loss_slack": float(loss_slackness.item()),
        "loss_dual_pos": float(loss_dual_pos.item()),
        "loss_primal_pos": float(loss_primal_pos.item()),
        "max_primal_violation": float(torch.max(primal_violations).item()) if len(primal_violations) > 0 else 0.0,
    }

    return total_loss, metrics
