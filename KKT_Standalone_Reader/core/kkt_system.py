"""
kkt_system.py
=============
This file defines the unified data structure (KKTSystem) that every reader outputs.

Why do we need this?
--------------------
Whether a user provides an .mps file, an .lp file, a Pyomo model, or a JSON file,
the Neural Network (KINN) only cares about pure linear algebra:
    min  c^T * x
    s.t. G * x <= h

Every reader in this library converts the input into this exact standard class.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class KKTSystem:
    """
    Standardized problem container for KKT-Informed Neural Networks (KINNs).

    Attributes:
    -----------
    name : str
        Human-readable name of the problem instance (e.g. "afiro", "diet_problem").
    c : np.ndarray
        Cost vector of shape (n_vars,). Always normalized to MINIMIZATION.
        (If the original problem was maximize c^T x, c is flipped to -c).
    G : np.ndarray
        Inequality constraint matrix of shape (n_constraints, n_vars).
        Includes both structural constraints (A x <= b) and variable bounds
        folded into G as:
            -x <= -lb   and   x <= ub
    h : np.ndarray
        Upper bound RHS vector of shape (n_constraints,).
        Pairs with G such that the feasible region is: G * x <= h.
    var_names : List[str]
        Names of the decision variables in order (e.g. ["x1", "x2", ...]).
    con_names : List[str]
        Names of the constraints in order (e.g. ["c1", "bound: x1 >= 0", ...]).
    col_lower : Optional[np.ndarray]
        Original variable lower bounds (shape (n_vars,)). Default: 0.0 or -inf.
    col_upper : Optional[np.ndarray]
        Original variable upper bounds (shape (n_vars,)). Default: +inf.
    sparse_coo : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Sparse triplet representation of G: (row_indices, col_indices, values).
        Perfect for Graph Neural Networks (GNNs) or PyTorch sparse tensors!
    offset : float
        Constant objective offset (default: 0.0).
    """
    name: str
    c: np.ndarray
    G: np.ndarray
    h: np.ndarray
    var_names: List[str] = field(default_factory=list)
    con_names: List[str] = field(default_factory=list)
    col_lower: Optional[np.ndarray] = None
    col_upper: Optional[np.ndarray] = None
    sparse_coo: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    offset: float = 0.0

    @property
    def n_vars(self) -> int:
        """Number of primal decision variables (n)."""
        return len(self.c)

    @property
    def n_constraints(self) -> int:
        """Total number of inequality rows in G (m)."""
        return len(self.h)

    def to_torch(self, device: str = "cpu"):
        """
        Convenience helper to convert (c, G, h) into PyTorch float32 tensors.
        
        Returns:
            dict containing "c", "G", "h" as torch.Tensor ready for your KINN!
        """
        try:
            import torch
            return {
                "c": torch.tensor(self.c, dtype=torch.float32, device=device),
                "G": torch.tensor(self.G, dtype=torch.float32, device=device),
                "h": torch.tensor(self.h, dtype=torch.float32, device=device)
            }
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. Run 'pip install torch' to use to_torch()."
            )

    def summary(self) -> str:
        """Prints a friendly summary of problem dimensions and non-zeros."""
        nnz = np.count_nonzero(self.G)
        density = (nnz / (self.n_vars * self.n_constraints)) * 100 if self.n_vars * self.n_constraints > 0 else 0
        
        return (
            f"=== KKTSystem: '{self.name}' ===\n"
            f"  * Primal Variables (n):    {self.n_vars}\n"
            f"  * Total Inequalities (m):  {self.n_constraints}  (G x <= h)\n"
            f"  * Non-zero entries in G:   {nnz} ({density:.2f}% density)\n"
            f"  * Cost vector range:       [{self.c.min():.2e}, {self.c.max():.2e}]\n"
            f"  * RHS vector range:        [{self.h.min():.2e}, {self.h.max():.2e}]"
        )
