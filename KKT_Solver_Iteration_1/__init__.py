"""
KKT_Solver_Iteration_1
======================
First iteration of the Single-Instance Physics-Informed KKT Neural Solver.

Public Exports:
---------------
  - solve_kkt_instance : Main solving function
  - SingleInstanceKINN : Dual-head dynamic neural network
  - DirectParameterKINN : Parameterized direct optimizer
  - basic_kkt_loss : Physics-informed loss function
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
