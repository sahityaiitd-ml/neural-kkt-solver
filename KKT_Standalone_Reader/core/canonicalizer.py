"""
canonicalizer.py
================
Mathematical transformer that converts any optimization problem into the canonical
inequality form:
    min  c^T * x
    s.t. G * x <= h

Why does this matter for KINNs?
-------------------------------
Real-world problems have all kinds of constraints:
  - Upper bounds:       2*x1 + x2 <= 10
  - Lower bounds:       x1 + 3*x2 >= 4
  - Equalities:         x1 + x2 == 5
  - Variable bounds:    0 <= x1 <= 100
  - Maximize objectives: max 3*x1 + 2*x2

This module cleanly converts ALL of those into one unified G*x <= h system:
  - "expr <= U"   --> stays  expr <= U
  - "expr >= L"   --> flips to  -expr <= -L
  - "expr == B"   --> splits into: expr <= B  AND  -expr <= -B
  - "x_i >= lb"   --> becomes  -x_i <= -lb
  - "x_i <= ub"   --> becomes   x_i <= ub
  - "max c^T x"   --> flips to "min -c^T x"
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .kkt_system import KKTSystem


def build_canonical_kkt_system(
    name: str,
    n_vars: int,
    var_names: List[str],
    col_cost: np.ndarray,
    sense: str,
    row_names: List[str],
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    row_coefs: List[Dict[int, float]],
    col_lower: Optional[np.ndarray] = None,
    col_upper: Optional[np.ndarray] = None,
    offset: float = 0.0,
    include_variable_bounds_in_G: bool = True
) -> KKTSystem:
    """
    Takes parsed raw arrays and compiles them into a unified KKTSystem.
    
    Parameters:
    -----------
    name : str
        Problem instance name.
    n_vars : int
        Number of decision variables.
    var_names : List[str]
        List of variable names.
    col_cost : np.ndarray
        Raw cost vector.
    sense : str
        "minimize" or "maximize".
    row_names : List[str]
        Names of the constraint rows.
    row_lower : np.ndarray
        Lower limits on each row (can be -inf).
    row_upper : np.ndarray
        Upper limits on each row (can be +inf).
    row_coefs : List[Dict[int, float]]
        Sparse row coefficients: row_coefs[r] = {col_idx: value}.
    col_lower : Optional[np.ndarray]
        Variable lower bounds (defaults to 0.0).
    col_upper : Optional[np.ndarray]
        Variable upper bounds (defaults to +inf).
    offset : float
        Objective scalar offset.
    include_variable_bounds_in_G : bool
        Whether to explicitly fold variable bounds (x >= lb, x <= ub) into G x <= h.
        Defaults to True (standard for KINNs).
    """
    # 1. Normalize Objective to MINIMIZE:
    # If the user wrote "maximize 3*x1", that is equivalent to "minimize -3*x1".
    c = np.array(col_cost, dtype=np.float64, copy=True)
    if sense.lower() in ("max", "maximize", "maximise"):
        c = -c

    # Set default variable bounds if not provided:
    # In standard LP convention, variables without explicit bounds are non-negative [0, +inf).
    if col_lower is None:
        col_lower = np.zeros(n_vars, dtype=np.float64)
    else:
        col_lower = np.array(col_lower, dtype=np.float64)

    if col_upper is None:
        col_upper = np.full(n_vars, np.inf, dtype=np.float64)
    else:
        col_upper = np.array(col_upper, dtype=np.float64)

    # 2. Build rows of G and h
    G_rows: List[np.ndarray] = []
    h_vals: List[float] = []
    con_labels: List[str] = []

    # Sparse triplet collectors for Graph Neural Networks (GNNs):
    coo_rows: List[int] = []
    coo_cols: List[int] = []
    coo_vals: List[float] = []

    current_row_idx = 0

    def add_inequality_row(coef_dict: Dict[int, float], rhs_val: float, label: str, negate: bool = False):
        """Helper to append one inequality row (negate=True turns >= into <=)."""
        nonlocal current_row_idx
        row_dense = np.zeros(n_vars, dtype=np.float64)
        multiplier = -1.0 if negate else 1.0

        for col_idx, val in coef_dict.items():
            final_val = multiplier * float(val)
            row_dense[col_idx] = final_val
            if abs(final_val) > 1e-15:
                coo_rows.append(current_row_idx)
                coo_cols.append(col_idx)
                coo_vals.append(final_val)

        G_rows.append(row_dense)
        h_vals.append(multiplier * float(rhs_val))
        con_labels.append(label)
        current_row_idx += 1

    # Loop through all structural constraints:
    n_rows = len(row_coefs)
    for r in range(n_rows):
        coefs = row_coefs[r]
        if not coefs:
            continue  # skip degenerate empty rows

        rname = row_names[r] if r < len(row_names) else f"row_{r}"
        lo = float(row_lower[r]) if r < len(row_lower) else -np.inf
        up = float(row_upper[r]) if r < len(row_upper) else np.inf

        # Case A: Equality constraint (lo == up == b)
        if not np.isneginf(lo) and not np.isposinf(up) and abs(lo - up) < 1e-9:
            add_inequality_row(coefs, up, f"{rname}_upper", negate=False)
            add_inequality_row(coefs, lo, f"{rname}_lower", negate=True)

        # Case B: Upper-bounded constraint (expr <= up)
        elif not np.isposinf(up) and np.isneginf(lo):
            add_inequality_row(coefs, up, f"{rname}_upper", negate=False)

        # Case C: Lower-bounded constraint (expr >= lo  ==>  -expr <= -lo)
        elif not np.isneginf(lo) and np.isposinf(up):
            add_inequality_row(coefs, lo, f"{rname}_lower", negate=True)

        # Case D: Ranged constraint (lo <= expr <= up)
        elif not np.isneginf(lo) and not np.isposinf(up):
            add_inequality_row(coefs, up, f"{rname}_upper", negate=False)
            add_inequality_row(coefs, lo, f"{rname}_lower", negate=True)

    # 3. Fold variable bounds into G x <= h
    if include_variable_bounds_in_G:
        for j in range(n_vars):
            vname = var_names[j] if j < len(var_names) else f"x_{j}"
            lb = col_lower[j]
            ub = col_upper[j]

            # Lower bound: x_j >= lb  ==>  -x_j <= -lb
            if not np.isneginf(lb):
                add_inequality_row({j: 1.0}, lb, f"bound_{vname}_lower", negate=True)

            # Upper bound: x_j <= ub  ==>   x_j <= ub
            if not np.isposinf(ub):
                add_inequality_row({j: 1.0}, ub, f"bound_{vname}_upper", negate=False)

    # Convert to standard NumPy arrays
    G = np.array(G_rows, dtype=np.float64) if G_rows else np.zeros((0, n_vars), dtype=np.float64)
    h = np.array(h_vals, dtype=np.float64) if h_vals else np.zeros(0, dtype=np.float64)

    sparse_coo = (
        np.array(coo_rows, dtype=np.int64),
        np.array(coo_cols, dtype=np.int64),
        np.array(coo_vals, dtype=np.float64)
    )

    return KKTSystem(
        name=name,
        c=c,
        G=G,
        h=h,
        var_names=list(var_names),
        con_names=con_labels,
        col_lower=col_lower,
        col_upper=col_upper,
        sparse_coo=sparse_coo,
        offset=offset
    )
