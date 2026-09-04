"""
Pyomo KKT Pipeline
==================
A toolkit for modeling optimization problems in Pyomo, extracting their canonical
algebraic forms, formulating exact Karush-Kuhn-Tucker (KKT) optimality equations,
solving them with state-of-the-art solvers, and extracting primal/dual variables
for downstream Neural Network solver training.
"""

from .model_builder import (
    create_sample_lp,
    create_diet_problem,
    create_parametric_lp,
    create_resource_allocation_lp
)
from .kkt_extractor import PyomoKKTExtractor
from .solver import solve_pyomo_model, extract_canonical_solution
from .kkt_evaluator import evaluate_kkt_residuals

__all__ = [
    "create_sample_lp",
    "create_diet_problem",
    "create_parametric_lp",
    "create_resource_allocation_lp",
    "PyomoKKTExtractor",
    "solve_pyomo_model",
    "extract_canonical_solution",
    "evaluate_kkt_residuals",
]
