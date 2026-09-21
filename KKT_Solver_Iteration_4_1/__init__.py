"""
KKT_Solver_Iteration_4_1
========================
Cumulative Iteration 4 Solver Suite.
Integrates:
- Adaptive Preconditioning ("auto" mode selects Row L2 for sparse/wide problems, Ruiz for narrow corridors).
- Diameter-Adaptive Annealing for extended exploration on wide polytopes.
- Pure ReLU one-sided duality gap.
- Decoupled Primal & Dual MLPs with positive dual bias initialization.
- Pure KKT residual model checkpointing.
"""

from .model import SingleInstanceKINN, DirectParameterKINN
from .preconditioning import preconditioned_system, row_l2_preconditioning, ruiz_equilibration, PreconditionedSystem
from .loss import iteration_4_1_kkt_loss, iteration_4_3_1_kkt_loss, fischer_burmeister_loss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "preconditioned_system",
    "row_l2_preconditioning",
    "ruiz_equilibration",
    "PreconditionedSystem",
    "iteration_4_1_kkt_loss",
    "iteration_4_3_1_kkt_loss",
    "fischer_burmeister_loss",
    "solve_kkt_instance",
    "evaluate_solution"
]
