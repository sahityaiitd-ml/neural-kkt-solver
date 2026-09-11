"""
KKT_Benchmark
=============
Official benchmark suite for Neural KKT Solvers using real-world Netlib & MIPLIB linear programs.
"""

from .benchmark_harness import (
    evaluate_solver_on_benchmark,
    load_benchmark_result,
    compare_saved_benchmarks
)

__all__ = [
    "evaluate_solver_on_benchmark",
    "load_benchmark_result",
    "compare_saved_benchmarks"
]
