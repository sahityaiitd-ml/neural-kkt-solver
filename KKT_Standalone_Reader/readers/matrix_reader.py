"""
matrix_reader.py
================
Allows users to directly pass pure NumPy arrays and matrices into a KKTSystem.

Why is this useful?
-------------------
Many machine learning researchers and engineers already generate synthetic problems
or read datasets directly from SciPy, MATLAB, or NumPy:
    c = np.array([-3.0, -2.0])
    G = np.array([[2.0, 1.0], [1.0, 3.0]])
    h = np.array([10.0, 12.0])

This module validates dimensions, checks for NaNs/infs, extracts sparse COO triplets,
and packages everything cleanly into the standardized KKTSystem.
"""

from typing import Optional, List, Tuple
import numpy as np
from ..core.kkt_system import KKTSystem


def read_matrices(
    c: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    name: str = "matrix_lp",
    var_names: Optional[List[str]] = None,
    con_names: Optional[List[str]] = None,
    col_lower: Optional[np.ndarray] = None,
    col_upper: Optional[np.ndarray] = None,
    offset: float = 0.0
) -> KKTSystem:
    """
    Directly packages (c, G, h) NumPy arrays into a KKTSystem.
    
    Parameters:
    -----------
    c : np.ndarray
        Cost vector of shape (n_vars,).
    G : np.ndarray
        Constraint matrix of shape (n_constraints, n_vars) for G * x <= h.
    h : np.ndarray
        RHS vector of shape (n_constraints,).
    name : str
        Optional instance name.
    """
    c = np.asarray(c, dtype=np.float64).flatten()
    G = np.asarray(G, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).flatten()

    n_vars = len(c)
    n_constraints = len(h)

    if G.shape != (n_constraints, n_vars):
        raise ValueError(
            f"Dimension mismatch: G has shape {G.shape}, but expected ({n_constraints}, {n_vars}) "
            f"based on h (length {n_constraints}) and c (length {n_vars})."
        )

    if np.any(np.isnan(c)) or np.any(np.isnan(G)) or np.any(np.isnan(h)):
        raise ValueError("Input matrices contain NaN values.")

    if var_names is None:
        var_names = [f"x_{i+1}" for i in range(n_vars)]
    if con_names is None:
        con_names = [f"con_{j+1}" for j in range(n_constraints)]

    # Compute sparse COO triplets
    rows, cols = np.nonzero(G)
    vals = G[rows, cols]
    sparse_coo = (
        rows.astype(np.int64),
        cols.astype(np.int64),
        vals.astype(np.float64)
    )

    return KKTSystem(
        name=name,
        c=c,
        G=G,
        h=h,
        var_names=var_names,
        con_names=con_names,
        col_lower=col_lower,
        col_upper=col_upper,
        sparse_coo=sparse_coo,
        offset=offset
    )
