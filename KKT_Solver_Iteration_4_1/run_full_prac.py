"""
run_full_prac.py
================
Evaluates KKT_Solver_Iteration_4_3_1_Prac against Iteration 4.2 on key benchmark problems.
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_4_1 import solve_kkt_instance as solve_4_1, evaluate_solution
from KKT_Solver_Iteration_4 import solve_kkt_instance as solve_4

test_problems = [
    ("2_production_plan", "KKT_Standalone_Reader/examples/2_production_plan.lp", -21253.3333),
    ("sc50a", "KKT_Benchmark/problems/sc50a.mps.gz", -64.575077),
    ("afiro", "KKT_Benchmark/problems/afiro.mps.gz", -464.753143),
    ("blend", "KKT_Benchmark/problems/blend.mps.gz", -30.812150),
    ("degen2", "KKT_Benchmark/problems/degen2.mps.gz", -1435.18),
    ("agg", "KKT_Benchmark/problems/agg.mps.gz", -35991767.29),
    ("flugpl", "KKT_Benchmark/problems/flugpl.mps.gz", 1167185.73),
    ("beaconfd", "KKT_Benchmark/problems/beaconfd.mps.gz", 33592.49),
    ("dcmulti", "KKT_Benchmark/problems/dcmulti.mps.gz", 183975.54),
    ("boeing1", "KKT_Benchmark/problems/boeing1.mps.gz", -335.21),
]


def run_benchmark():
    print("=" * 115)
    print("HEAD-TO-HEAD COMPARISON: Iteration 4 vs Iteration 4.1")
    print("=" * 115)
    header = (
        f"{'Problem':<18} | {'HiGHS Ref':<12} | "
        f"{'Iter 4 Obj':<14} | {'Iter 4 Gap':<12} | "
        f"{'Iter 4.1 Obj':<14} | {'Iter 4.1 Gap':<12} | {'4.1 Viol':<10}"
    )
    print(header)
    print("-" * 115)

    for name, path, z_true in test_problems:
        full_p = os.path.join(project_root, path)
        if not os.path.exists(full_p):
            continue

        prob = load_problem(full_p)

        # Solve with Iteration 4
        r4 = solve_4(prob, max_epochs=1200, lr=0.015, verbose=False)
        m4 = evaluate_solution(prob, r4["x_opt"], r4["lambda_opt"])
        gap4 = abs(m4["primal_objective"] - z_true) / abs(z_true) * 100.0

        # Solve with Iteration 4.1
        r41 = solve_4_1(prob, max_epochs=1200, lr=0.015, precond_method="auto", anneal_mode="diameter_adaptive", verbose=False)
        m41 = evaluate_solution(prob, r41["x_opt"], r41["lambda_opt"])
        gap41 = abs(m41["primal_objective"] - z_true) / abs(z_true) * 100.0

        row = (
            f"{name:<18} | {z_true:<12.2f} | "
            f"{m4['primal_objective']:<14.2f} | {gap4:<11.2f}% | "
            f"{m41['primal_objective']:<14.2f} | {gap41:<11.2f}% | {m41['max_primal_violation']:<10.2e}"
        )
        print(row)
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
