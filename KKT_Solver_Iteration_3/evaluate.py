"""
evaluate.py
===========
KKT residual evaluation metrics for Iteration 3.
"""

from typing import Dict, Any
import numpy as np


def evaluate_solution(
    kkt_system,
    x_pred: np.ndarray,
    lambda_pred: np.ndarray,
    tol: float = 1e-3
) -> Dict[str, Any]:
    c = kkt_system.c
    G = kkt_system.G
    h = kkt_system.h

    if hasattr(G, "toarray"):
        G_dense = G.toarray()
    else:
        G_dense = np.array(G)

    # 1. Primal Objective
    primal_obj = float(np.dot(c, x_pred))

    # 2. Slack & Primal Feasibility Violation
    Gx = np.dot(G_dense, x_pred)
    slack = h - Gx
    max_primal_violation = float(np.max(np.maximum(0.0, -slack)))

    # 3. Primal Non-Negativity Violation
    max_x_neg = float(np.max(np.maximum(0.0, -x_pred)))

    # 4. Dual Feasibility Violation (strictly lambda >= 0)
    max_dual_violation = float(np.max(np.maximum(0.0, -lambda_pred)))

    # 5. Stationarity Residual
    grad_L = c + np.dot(G_dense.T, lambda_pred)
    res_stationarity = float(np.linalg.norm(grad_L))

    # 6. Complementary Slackness Residual
    res_slackness = float(np.linalg.norm(lambda_pred * slack))

    # 7. Fischer-Burmeister Residual
    fb_val = lambda_pred + slack - np.sqrt(lambda_pred ** 2 + slack ** 2 + 1e-6)
    res_fb = float(np.linalg.norm(fb_val))

    total_kkt_error = res_stationarity + max_primal_violation + max_dual_violation + res_slackness

    return {
        "primal_objective": primal_obj,
        "max_primal_violation": max_primal_violation,
        "max_x_neg": max_x_neg,
        "max_dual_violation": max_dual_violation,
        "res_stationarity": res_stationarity,
        "res_slackness": res_slackness,
        "res_fb": res_fb,
        "total_kkt_error": total_kkt_error,
        "is_feasible": max_primal_violation <= tol and max_x_neg <= tol and max_dual_violation <= tol,
    }
