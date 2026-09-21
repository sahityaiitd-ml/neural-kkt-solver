"""
evaluate.py
===========
KKT residual evaluation and optimality metrics for Iteration 4.
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

    # 2. Dual Objective (in canonical form min c^T x s.t. Gx <= h: dual is max -h^T lambda)
    dual_obj = float(-np.dot(h, lambda_pred))

    # 3. Strong Duality Gap
    abs_duality_gap = abs(primal_obj - dual_obj)
    rel_duality_gap = abs_duality_gap / max(1.0, abs(primal_obj) + abs(dual_obj))

    # 4. Slack & Primal Feasibility Violation
    Gx = np.dot(G_dense, x_pred)
    slack = h - Gx
    max_primal_violation = float(np.max(np.maximum(0.0, -slack)))

    # 5. Primal Non-Negativity Violation
    max_x_neg = float(np.max(np.maximum(0.0, -x_pred)))

    # 6. Dual Feasibility Violation (strictly lambda >= 0)
    max_dual_violation = float(np.max(np.maximum(0.0, -lambda_pred)))

    # 7. Stationarity Residual: ||c + G^T lambda||
    grad_L = c + np.dot(G_dense.T, lambda_pred)
    res_stationarity = float(np.linalg.norm(grad_L))

    # 8. Complementary Slackness Residual: ||lambda * slack||
    res_slackness = float(np.linalg.norm(lambda_pred * slack))

    total_kkt_error = res_stationarity + max_primal_violation + max_dual_violation + res_slackness

    return {
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "duality_gap": abs_duality_gap,
        "rel_duality_gap": rel_duality_gap,
        "max_primal_violation": max_primal_violation,
        "max_x_neg": max_x_neg,
        "max_dual_violation": max_dual_violation,
        "res_stationarity": res_stationarity,
        "res_slackness": res_slackness,
        "total_kkt_error": total_kkt_error,
        "is_feasible": max_primal_violation <= tol and max_x_neg <= tol and max_dual_violation <= tol,
    }
