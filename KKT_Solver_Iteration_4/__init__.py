"""
KKT_Solver_Iteration_4
======================
Decoupled Architecture with Linear Objective Pull & Preconditioning.

Key Highlights:
1. Fully decoupled primal and dual networks (no shared backbone).
2. Linear primal objective pull actively forces solution away from the x=0 trap.
3. Row & cost preconditioning ensures O(1) slack and residual scaling.
4. Strong duality gap coupling.
"""

from .model import SingleInstanceKINN
from .loss import KKTLoss, iteration_4_kkt_loss, iteration_4_1_kkt_loss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "SingleInstanceKINN",
    "KKTLoss",
    "iteration_4_kkt_loss",
    "iteration_4_1_kkt_loss",
    "solve_kkt_instance",
    "evaluate_solution"
]
