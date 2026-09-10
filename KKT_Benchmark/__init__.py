"""
KKT_Benchmark
=============
Official benchmark suite for Neural KKT Solvers using real-world Netlib & MIPLIB linear programs.
"""

from .benchmark_harness import evaluate_solver_on_benchmark

__all__ = ["evaluate_solver_on_benchmark"]
