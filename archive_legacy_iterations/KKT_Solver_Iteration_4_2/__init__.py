"""
KKT_Solver_Iteration_4_2
========================
Iteration 4.2: Annealed Objective Pull & One-Sided Strong Duality Gap.

Key Innovations:
- Decoupled Primal & Dual Networks (independent representation pathways).
- Annealed Objective Pull (w_obj * gamma^t) expels solver from origin trap without persistent distortion.
- One-Sided Duality Gap ReLU(c_tilde^T x + h_tilde^T lambda)^2 eliminates false equilibrium traps.
- Pure KKT residual model checkpointing.
- Canonical row and cost preconditioning.
"""

from .model import SingleInstanceKINN, DirectParameterKINN
from .loss import iteration_4_2_kkt_loss, fischer_burmeister_loss, KKTLoss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution
from .plot_loss import plot_iteration_4_2_convergence

__all__ = [
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "iteration_4_2_kkt_loss",
    "fischer_burmeister_loss",
    "KKTLoss",
    "solve_kkt_instance",
    "evaluate_solution",
    "plot_iteration_4_2_convergence",
]
