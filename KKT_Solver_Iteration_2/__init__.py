"""
KKT_Solver_Iteration_2
======================
Second iteration of the Single-Instance Physics-Informed KKT Neural Solver.

Change vs Iteration 1:
----------------------
  - Backbone activation: ReLU -> GELU (avoids dead neurons near constraint faces)
  - Dual head: raw Linear -> Linear + Softplus (lambda_hat >= 0 enforced structurally)
  - Loss: dual non-negativity term removed (now always zero by construction)

Public Exports:
---------------
  - solve_kkt_instance : Main solving function
  - SingleInstanceKINN : Dual-head dynamic neural network (GELU + Softplus dual head)
  - DirectParameterKINN : Parameterized direct optimizer
  - basic_kkt_loss : Physics-informed loss function (4 terms)
  - compare_kinn_vs_highs : Verification against HiGHS ground truth
"""

from .model import SingleInstanceKINN, DirectParameterKINN
from .loss import basic_kkt_loss
from .solver import solve_kkt_instance
from .evaluate import compare_kinn_vs_highs

__all__ = [
    "solve_kkt_instance",
    "SingleInstanceKINN",
    "DirectParameterKINN",
    "basic_kkt_loss",
    "compare_kinn_vs_highs"
]
