"""
kkt_evaluator.py
================
Computes the exact Karush-Kuhn-Tucker (KKT) residuals for any given candidate
solution (x, lambda, nu) against the canonical optimization matrices (Q, c, G, h, A, b).
"""

import numpy as np


def evaluate_kkt_residuals(matrices: dict, x: np.ndarray, lam: np.ndarray, nu: np.ndarray = None):
    """
    Computes numerical KKT residuals for candidate primal and dual variables.
    
    Canonical Problem Form:
        min_x  1/2 x^T Q x + c^T x
        s.t.   G x <= h
               A x == b
               
    KKT Conditions:
        1. Stationarity:          Q x + c + G^T lambda + A^T nu = 0
        2. Primal Infeasibility:   max(0, G x - h) = 0   and   A x - b = 0
        3. Dual Infeasibility:     max(0, -lambda) = 0
        4. Complementary Slack:   lambda * (G x - h) = 0
        
    Parameters
    ----------
    matrices : dict
        Dictionary returned by PyomoKKTExtractor.get_canonical_matrices() containing Q, c, G, h, A, b
    x : np.ndarray of shape (n,)
        Primal variables
    lam : np.ndarray of shape (m,)
        Dual multipliers for inequality constraints G x <= h
    nu : np.ndarray of shape (p,), optional
        Dual multipliers for equality constraints A x == b (defaults to zeros if None)
        
    Returns
    -------
    dict:
        Residual vectors and scalar L2 norms for each KKT condition.
    """
    Q = matrices["Q"]
    c = matrices["c"]
    G = matrices["G"]
    h = matrices["h"]
    A = matrices["A"]
    b = matrices["b"]
    
    n_vars = len(c)
    n_ineq = len(h)
    n_eq = len(b)
    
    x = np.asarray(x, dtype=np.float64).flatten()
    lam = np.asarray(lam, dtype=np.float64).flatten() if lam is not None else np.zeros(n_ineq)
    nu = np.asarray(nu, dtype=np.float64).flatten() if nu is not None and n_eq > 0 else np.zeros(n_eq)
    
    # 1. Stationarity Residual: r_S = Q x + c + G^T lambda + A^T nu
    r_S = c.copy()
    if np.any(Q != 0):
        r_S += Q @ x
    if n_ineq > 0 and len(lam) > 0:
        r_S += G.T @ lam
    if n_eq > 0 and len(nu) > 0:
        r_S += A.T @ nu
        
    # 2. Primal Inequality Residual: r_I = max(0, G x - h)
    if n_ineq > 0:
        slack = G @ x - h
        r_I = np.maximum(0.0, slack)
    else:
        slack = np.zeros(0)
        r_I = np.zeros(0)
        
    # 3. Primal Equality Residual: r_E = A x - b
    if n_eq > 0:
        r_E = A @ x - b
    else:
        r_E = np.zeros(0)
        
    # 4. Dual Feasibility Residual: r_DF = max(0, -lambda)
    if n_ineq > 0:
        r_DF = np.maximum(0.0, -lam)
    else:
        r_DF = np.zeros(0)
        
    # 5. Complementary Slackness Residual: r_CS = lambda * (G x - h)
    if n_ineq > 0:
        r_CS = lam * slack
    else:
        r_CS = np.zeros(0)
        
    norm_S = float(np.linalg.norm(r_S))
    norm_I = float(np.linalg.norm(r_I)) if len(r_I) > 0 else 0.0
    norm_E = float(np.linalg.norm(r_E)) if len(r_E) > 0 else 0.0
    norm_DF = float(np.linalg.norm(r_DF)) if len(r_DF) > 0 else 0.0
    norm_CS = float(np.linalg.norm(r_CS)) if len(r_CS) > 0 else 0.0
    
    total_kkt_error = norm_S + norm_I + norm_E + norm_DF + norm_CS
    
    return {
        "norm_stationarity": norm_S,
        "norm_primal_ineq": norm_I,
        "norm_primal_eq": norm_E,
        "norm_dual_feasibility": norm_DF,
        "norm_complementary_slackness": norm_CS,
        "total_kkt_error": total_kkt_error,
        "residuals": {
            "r_S": r_S,
            "r_I": r_I,
            "r_E": r_E,
            "r_DF": r_DF,
            "r_CS": r_CS,
        },
        "slack_vector": slack
    }
