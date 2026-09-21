"""
KKT_Solver_Iteration_4
=======================
Fourth-Generation Physics-Informed KKT Solver for Linear Programming.

Key Enhancements:
-----------------
1. Canonical Row and Cost Preconditioning (Equilibration).
2. Strong Duality Theorem Gap Loss (coupling primal x directly to cost c).
3. Softplus anti-saturation positive bias initialization (+1.0).
4. Smoothed Fischer-Burmeister complementarity on normalized slacks.
"""

from .model import SingleInstanceKINN
from .loss import KKTLoss, iteration_4_kkt_loss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "SingleInstanceKINN",
    "KKTLoss",
    "iteration_4_kkt_loss",
    "solve_kkt_instance",
    "evaluate_solution"
]
