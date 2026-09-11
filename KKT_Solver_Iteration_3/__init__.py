"""
KKT_Solver_Iteration_3
======================
Physics-Informed Neural Network LP Solver (Iteration 3).

Key Innovation: Path B - Fischer-Burmeister Complementarity Formulation.
Replaces independent primal penalty and slackness penalty with the unified,
smoothed Fischer-Burmeister C-function from Texas A&M (arXiv:2507.08124v1):
phi_eps(lambda, s) = lambda + s - sqrt(lambda^2 + s^2 + eps)
"""

from .model import KINNSolverModel, SingleInstanceKINN, DirectParameterKINN
from .loss import KKTLoss, basic_kkt_loss, fischer_burmeister_loss
from .solver import solve_kkt_instance
from .evaluate import evaluate_solution

__all__ = [
    "KINNSolverModel",
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "KKTLoss",
    "basic_kkt_loss",
    "fischer_burmeister_loss",
    "solve_kkt_instance",
    "evaluate_solution",
]
