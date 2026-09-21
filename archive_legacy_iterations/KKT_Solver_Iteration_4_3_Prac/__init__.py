"""
KKT_Solver_Iteration_4_3_Prac
=============================
Experimental Sandbox for testing:
- Modular Preconditioning (Row L2 vs Ruiz Equilibration).
- Diameter-Adaptive Annealing for wide polytopes.
- Analytical Primal and Dual Unscaling.
"""

from .model import SingleInstanceKINN, DirectParameterKINN
from .preconditioning import preconditioned_system, row_l2_preconditioning, ruiz_equilibration, PreconditionedSystem
from .loss import iteration_4_3_kkt_loss, fischer_burmeister_loss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "preconditioned_system",
    "row_l2_preconditioning",
    "ruiz_equilibration",
    "PreconditionedSystem",
    "iteration_4_3_kkt_loss",
    "fischer_burmeister_loss",
    "solve_kkt_instance",
    "evaluate_solution"
]
