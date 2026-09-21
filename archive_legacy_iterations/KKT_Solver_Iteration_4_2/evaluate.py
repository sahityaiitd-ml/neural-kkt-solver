"""
evaluate.py
===========
Rigorous KKT solution evaluation metrics for Iteration 4.2.
Evaluates solutions purely in original, unscaled problem space.
"""

from typing import Dict, Any
import numpy as np


def evaluate_solution(kkt_system, x_opt: np.ndarray, lambda_opt: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates primal-dual candidate solution against exact unscaled KKT conditions.
    """
    c = np.array(kkt_system.c, dtype=np.float64)
    h = np.array(kkt_system.h, dtype=np.float64)
    if hasattr(kkt_system.G, "toarray"):
        G = kkt_system.G.toarray().astype(np.float64)
    else:
        G = np.array(kkt_system.G, dtype=np.float64)

    x = np.nan_to_num(np.array(x_opt, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    lam = np.nan_to_num(np.array(lambda_opt, dtype=np.float64), nan=0.0, posinf=1e6, neginf=0.0)

    # 1. Objectives
    primal_obj = float(np.dot(c, x))
    dual_obj = float(-np.dot(h, lam))
    duality_gap = abs(primal_obj - dual_obj)

    # 2. Primal Feasibility
    slack = h - G @ x
    primal_violation = np.maximum(0.0, -slack)
    max_primal_violation = float(np.max(primal_violation)) if len(primal_violation) > 0 else 0.0

    # 3. Dual Feasibility: lambda >= 0
    dual_violation = np.maximum(0.0, -lam)
    max_dual_violation = float(np.max(dual_violation)) if len(dual_violation) > 0 else 0.0

    # 4. Stationarity: c + G^T * lambda = 0
    stationarity_residual = c + G.T @ lam
    res_stationarity = float(np.linalg.norm(stationarity_residual, ord=np.inf))

    # 5. Complementary Slackness: lambda_i * (h_i - G_i * x) = 0
    comp_slack = lam * slack
    res_complementarity = float(np.linalg.norm(comp_slack, ord=np.inf))

    return {
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "duality_gap": duality_gap,
        "max_primal_violation": max_primal_violation,
        "max_dual_violation": max_dual_violation,
        "res_stationarity": res_stationarity,
        "res_complementarity": res_complementarity,
        "is_primal_feasible": max_primal_violation < 1e-4,
        "is_dual_feasible": max_dual_violation < 1e-4,
        "is_kkt_optimal": max_primal_violation < 1e-3 and res_stationarity < 1e-2 and res_complementarity < 1e-2
    }
