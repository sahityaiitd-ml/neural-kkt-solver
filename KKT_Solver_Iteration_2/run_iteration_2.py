"""
run_iteration_2.py
==================
Main demonstration script for KKT_Solver_Iteration_2.

Solves:
  1. Diet Problem (examples/1_diet_problem.mps)
  2. Production Planning (examples/2_production_plan.lp)
  3. Supply Chain (examples/3_supply_chain.json)
  4. Netlib Benchmark (KKT_Benchmark/problems/afiro.mps.gz)

Verifies convergence and prints a consolidated comparison table against HiGHS!
"""

import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_2 import solve_kkt_instance, compare_kinn_vs_highs
from KKT_Solver_Iteration_2.plot_loss import plot_kinn_convergence


def run_demonstration():
    print("=" * 70)
    print("🧠 KKT_SOLVER_ITERATION_2: GELU BACKBONE + SOFTPLUS DUAL HEAD")
    print("=" * 70)

    test_files = [
        os.path.join(project_root, "KKT_Standalone_Reader", "examples", "1_diet_problem.mps"),
        os.path.join(project_root, "KKT_Standalone_Reader", "examples", "2_production_plan.lp"),
        os.path.join(project_root, "KKT_Standalone_Reader", "examples", "3_supply_chain.json"),
        os.path.join(project_root, "KKT_Benchmark", "problems", "afiro.mps.gz")
    ]

    results_table = []

    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File '{filepath}' not found, skipping.")
            continue

        filename = os.path.basename(filepath)
        print(f"\n📂 Loading Problem: {filename}")
        problem = load_problem(filepath)
        print(problem.summary())

        # 1. Solve using KINN Single-Instance Optimizer
        kinn_res = solve_kkt_instance(
            problem,
            model_type="neural",
            max_epochs=2000,
            lr=0.015,
            tol=1e-4,
            w_stat=1.0,
            w_prim=5.0,
            w_slack=2.0,
            verbose=True
        )

        # 2. Compare against HiGHS Ground Truth
        comp = compare_kinn_vs_highs(problem, kinn_res)
        results_table.append(comp)

        # 3. Save the 4-term KKT loss convergence curve
        plot_path = os.path.join(current_dir, f"convergence_{problem.name}.png")
        plot_kinn_convergence(
            kinn_res["history"],
            title=f"Iteration 2 (GELU + Softplus) — {problem.name}",
            save_path=plot_path
        )

    # 3. Print Consolidated Report Table
    print("\n" + "=" * 80)
    print("🏆 CONSOLIDATED KINN ITERATION 2 RESULTS TABLE")
    print("=" * 80)
    print(f"{'Problem':<22} | {'Vars':<5} | {'Cons':<5} | {'HiGHS Obj':<12} | {'KINN Obj':<12} | {'Gap %':<8} | {'Max Viol':<10}")
    print("-" * 80)
    for r in results_table:
        print(
            f"{r['problem_name'][:22]:<22} | "
            f"{r['n_vars']:<5} | "
            f"{r['n_constraints']:<5} | "
            f"{r['highs_objective']:<12.2f} | "
            f"{r['kinn_objective']:<12.2f} | "
            f"{r['objective_gap_percent']:<8.2f}% | "
            f"{r['max_primal_violation']:<10.2e}"
        )
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
