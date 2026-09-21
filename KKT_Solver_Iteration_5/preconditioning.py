"""
preconditioning.py
==================
Adaptive Preconditioning Engine for KKT_Solver_Iteration_5.

Features:
- "auto" mode: Intelligently selects between Row L2 and Ruiz equilibration based
  on cost sparsity and column imbalance.
- Full analytical transformations for primal variables and dual multipliers.
- Polytope diameter estimation for span-adaptive exploration.
"""

from typing import Tuple, Dict, Any
import numpy as np


class PreconditionedSystem:
    def __init__(
        self,
        G_norm: np.ndarray,
        h_norm: np.ndarray,
        c_norm: np.ndarray,
        R: np.ndarray,
        C: np.ndarray,
        cost_scale: float,
        method: str,
        estimated_diameter: float
    ):
        self.G_norm = G_norm
        self.h_norm = h_norm
        self.c_norm = c_norm
        self.R = R
        self.C = C
        self.cost_scale = cost_scale
        self.method = method
        self.estimated_diameter = estimated_diameter

    def unscale_primal(self, x_tilde: np.ndarray) -> np.ndarray:
        """Transforms preconditioned primal variables back to original space: x = C^{-1} * x_tilde"""
        return x_tilde / self.C

    def unscale_dual(self, lambda_tilde: np.ndarray) -> np.ndarray:
        """Transforms preconditioned dual multipliers back to original space: lambda = s_c * R^{-1} * lambda_tilde"""
        return self.cost_scale * (lambda_tilde / self.R)


def row_l2_preconditioning(
    G: np.ndarray,
    h: np.ndarray,
    c: np.ndarray
) -> PreconditionedSystem:
    """
    Canonical Row L2 Normalization.
    Each constraint row is scaled by its Euclidean norm.
    """
    m, n = G.shape
    row_norms = np.linalg.norm(G, axis=1)
    R = np.where(row_norms < 1e-8, 1.0, row_norms)
    C = np.ones(n, dtype=np.float64)

    G_norm = G / R[:, np.newaxis]
    h_norm = h / R

    cost_scale = float(max(1.0, np.linalg.norm(c)))
    c_norm = c / cost_scale

    pos_h = np.maximum(0.1, np.abs(h_norm))
    estimated_diameter = float(np.median(pos_h))

    return PreconditionedSystem(
        G_norm=G_norm,
        h_norm=h_norm,
        c_norm=c_norm,
        R=R,
        C=C,
        cost_scale=cost_scale,
        method="row_l2",
        estimated_diameter=estimated_diameter
    )


def ruiz_equilibration(
    G: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-3
) -> PreconditionedSystem:
    """
    Ruiz Row-and-Column Equilibration.
    Iteratively equilibrates both rows and columns to have unit infinity norms.
    """
    m, n = G.shape
    R = np.ones(m, dtype=np.float64)
    C = np.ones(n, dtype=np.float64)
    G_work = G.copy().astype(np.float64)

    for it in range(max_iter):
        row_norms = np.sqrt(np.max(np.abs(G_work), axis=1))
        row_norms = np.where(row_norms < 1e-8, 1.0, row_norms)
        G_work = G_work / row_norms[:, None]
        R = R * row_norms

        col_norms = np.sqrt(np.max(np.abs(G_work), axis=0))
        col_norms = np.where(col_norms < 1e-8, 1.0, col_norms)
        G_work = G_work / col_norms[None, :]
        C = C * col_norms

        if np.max(np.abs(1.0 - row_norms)) < tol and np.max(np.abs(1.0 - col_norms)) < tol:
            break

    h_norm = h / R
    c_scaled = c / C
    cost_scale = float(max(1.0, np.linalg.norm(c_scaled)))
    c_norm = c_scaled / cost_scale

    pos_h = np.maximum(0.1, np.abs(h_norm))
    estimated_diameter = float(np.median(pos_h))

    return PreconditionedSystem(
        G_norm=G_work,
        h_norm=h_norm,
        c_norm=c_norm,
        R=R,
        C=C,
        cost_scale=cost_scale,
        method="ruiz",
        estimated_diameter=estimated_diameter
    )


def adaptive_preconditioning(
    G: np.ndarray,
    h: np.ndarray,
    c: np.ndarray
) -> PreconditionedSystem:
    """
    Intelligent Adaptive Preconditioning selector.
    """
    m, n = G.shape
    c_nonzero_frac = float(np.sum(np.abs(c) > 1e-8)) / max(1, n)
    col_max = np.max(np.abs(G), axis=0)
    nz_col_max = col_max[col_max > 1e-8]
    col_ratio = float(np.max(col_max) / np.min(nz_col_max)) if len(nz_col_max) > 0 else 1.0

    pos_h_raw = np.maximum(0.1, np.abs(h))
    raw_span = float(np.median(pos_h_raw))

    if c_nonzero_frac < 0.30 or raw_span > 100.0:
        sys_obj = row_l2_preconditioning(G, h, c)
        sys_obj.method = "auto(row_l2)"
        return sys_obj
    elif col_ratio > 15.0 and m / n < 3.5:
        sys_obj = ruiz_equilibration(G, h, c)
        sys_obj.method = "auto(ruiz)"
        return sys_obj
    else:
        sys_obj = row_l2_preconditioning(G, h, c)
        sys_obj.method = "auto(row_l2)"
        return sys_obj


def preconditioned_system(
    G: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    method: str = "auto"
) -> PreconditionedSystem:
    """Factory function for preconditioning."""
    if method.lower() == "ruiz":
        return ruiz_equilibration(G, h, c)
    elif method.lower() == "row_l2":
        return row_l2_preconditioning(G, h, c)
    else:
        return adaptive_preconditioning(G, h, c)
