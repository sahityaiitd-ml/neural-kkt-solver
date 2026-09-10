"""
evaluate.py
===========
Evaluates KINN predicted solutions against HiGHS ground truth.

Computes:
---------
1. Objective Gap (%):        |c^T * x_kinn - c^T * x_highs| / |c^T * x_highs| * 100
2. Max Primal Violation:     max(0, G * x_kinn - h)
3. Stationarity Residual:    ||c + G^T * lambda_kinn||
4. Complementary Slackness:  ||lambda_kinn * (G * x_kinn - h)||
5. Solve time comparison:    HiGHS vs KINN
"""

import time
from typing import Dict, Any
import numpy as np
import pyomo.environ as pyo


def solve_with_highs_oracle(kkt_sys):
    """Solves the canonical problem directly with HiGHS to establish ground truth."""
    start = time.perf_counter()
    model = pyo.ConcreteModel()
    n_vars = kkt_sys.n_vars
    n_cons = kkt_sys.n_constraints

    model.V = pyo.Set(initialize=range(n_vars))
    model.x = pyo.Var(model.V, domain=pyo.Reals)

    model.obj = pyo.Objective(
        expr=sum(float(kkt_sys.c[j]) * model.x[j] for j in range(n_vars)),
        sense=pyo.minimize
    )

    model.C = pyo.Set(initialize=range(n_cons))

    def _rule(m, i):
        row = kkt_sys.G[i]
        expr = sum(float(row[j]) * m.x[j] for j in range(n_vars) if abs(row[j]) > 1e-15)
        return expr <= float(kkt_sys.h[i])

    model.cons = pyo.Constraint(model.C, rule=_rule)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available():
        solver = pyo.SolverFactory("glpk")

    results = solver.solve(model)
    elapsed = time.perf_counter() - start

    x_star = np.array([float(model.x[j].value) for j in range(n_vars)])
    lambda_star = np.array([max(0.0, -float(model.dual[model.cons[i]])) for i in range(n_cons)])
    obj_val = float(pyo.value(model.obj))

    return {
        "x_star": x_star,
        "lambda_star": lambda_star,
        "objective_val": obj_val,
        "solve_time_sec": elapsed
    }


def compare_kinn_vs_highs(kkt_sys, kinn_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes rigorous metrics comparing KINN's solution against HiGHS ground truth.
    """
    c = kkt_sys.c
    G = kkt_sys.G
    h = kkt_sys.h

    # 1. HiGHS Baseline
    highs = solve_with_highs_oracle(kkt_sys)
    z_highs = highs["objective_val"]

    # 2. KINN Prediction
    x_kinn = kinn_result["x_opt"]
    lam_kinn = kinn_result["lambda_opt"]
    z_kinn = float(np.dot(c, x_kinn))

    # 3. Objective Gap
    gap_percent = (abs(z_kinn - z_highs) / (abs(z_highs) + 1e-9)) * 100.0

    # 4. KKT Residuals on KINN's solution
    res_stationarity = float(np.linalg.norm(c + np.dot(G.T, lam_kinn)))
    slack = np.dot(G, x_kinn) - h
    max_violation = float(np.max(np.maximum(0.0, slack)))
    res_slackness = float(np.linalg.norm(lam_kinn * slack))

    comparison = {
        "problem_name": kkt_sys.name,
        "n_vars": kkt_sys.n_vars,
        "n_constraints": kkt_sys.n_constraints,
        "highs_objective": z_highs,
        "kinn_objective": z_kinn,
        "objective_gap_percent": gap_percent,
        "max_primal_violation": max_violation,
        "stationarity_norm": res_stationarity,
        "slackness_norm": res_slackness,
        "highs_time_sec": highs["solve_time_sec"],
        "kinn_time_sec": kinn_result["solve_time_sec"],
        "kinn_iterations": kinn_result["iterations"],
        "kinn_converged": kinn_result["converged"]
    }

    # Print summary table
    print("\n" + "=" * 65)
    print(f"📊 BENCHMARK COMPARISON: '{kkt_sys.name}'")
    print("=" * 65)
    print(f"  * HiGHS Optimal Objective:   {z_highs:,.4f}")
    print(f"  * KINN Predicted Objective:  {z_kinn:,.4f}")
    print(f"  * Objective Value Gap:       {gap_percent:.3f}%")
    print(f"  * Max Primal Infeasibility:  {max_violation:.4e}")
    print(f"  * Stationarity Error:        {res_stationarity:.4e}")
    print(f"  * Slackness Error:           {res_slackness:.4e}")
    print(f"  * KINN Epochs to Solve:      {kinn_result['iterations']} steps")
    print("=" * 65)

    return comparison
