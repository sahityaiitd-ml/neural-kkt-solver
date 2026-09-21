"""
KKT_Solver_Iteration_5
======================
Two-Stage Hybrid KKT Optimization Architecture:
- Stage 1: Neural Scout Discovery via Decoupled ALM KINN with PHR Augmented Lagrangian,
  row-scaled penalty rho_i, dual-aware objective pull, and adaptive preconditioning.
- Stage 2: Closed-Form Active-Set Linear Snap (G_viol^+) and Dual Reduced-Cost Basis
  Sparsity Snapping reaching true machine-zero boundary feasibility in microseconds.
"""

from .model import SingleInstanceKINN, DirectParameterKINN
from .preconditioning import preconditioned_system, PreconditionedSystem
from .loss import iteration_5_kkt_loss, fischer_burmeister_loss
from .snap import apply_active_set_linear_snap, apply_basis_sparsity_snapping
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "preconditioned_system",
    "PreconditionedSystem",
    "iteration_5_kkt_loss",
    "fischer_burmeister_loss",
    "apply_active_set_linear_snap",
    "apply_basis_sparsity_snapping",
    "solve_kkt_instance",
    "evaluate_solution"
]
