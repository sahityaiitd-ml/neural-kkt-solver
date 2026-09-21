"""
run_iteration_4.py
==================
Standalone runner and dry-run comparison for Iteration 4.
Solves 2_production_plan.lp and compares directly with Iteration 2 and Iteration 3.
"""

import os
import sys
import numpy as np

# Prevent matplotlib cache permission warning
os.environ["MPLCONFIGDIR"] = "/tmp/.mpl_cache"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_4 import solve_kkt_instance as solve_iter4, evaluate_solution as eval_solution
from KKT_Solver_Iteration_4.plot_loss import plot_iteration_4_convergence
from KKT_Solver_Iteration_3 import solve_kkt_instance as solve_iter3
from KKT_Solver_Iteration_2 import solve_kkt_instance as solve_iter2


def run_dry_test():
    print("=" * 80)
    print("RUNNING ITERATION 4 DRY RUN & MULTI-ITERATION COMPARISON")
    print("Enhancements: Problem Preconditioning + Strong Duality Gap + Anti-Saturation Dual Head")
    print("=" * 80)

    prob_path = os.path.join(project_root, "KKT_Standalone_Reader/examples/2_production_plan.lp")
    if not os.path.exists(prob_path):
        print(f"Error: Problem file not found at {prob_path}")
        return

    kkt_sys = load_problem(prob_path)
    print(f"\nProblem Instance: '{kkt_sys.name}' ({kkt_sys.n_vars} variables, {kkt_sys.n_constraints} constraints)")

    # 1. HiGHS Reference Optimal Value
    highs_opt = -21253.3333

    # 2. Run Iteration 4
    print("\n--- Solving with Iteration 4 (Preconditioned + Strong Duality + Anti-Saturation) ---")
    res_iter4 = solve_iter4(
        kkt_sys,
        max_epochs=1200,
        lr=0.015,
        verbose=True
    )
    metrics_iter4 = eval_solution(kkt_sys, res_iter4["x_opt"], res_iter4["lambda_opt"])

    # 3. Run Iteration 3
    print("\n--- Solving with Iteration 3 (Fischer-Burmeister Baseline) ---")
    res_iter3 = solve_iter3(
        kkt_sys,
        max_epochs=1200,
        lr=0.015,
        verbose=False
    )
    metrics_iter3 = eval_solution(kkt_sys, res_iter3["x_opt"], res_iter3["lambda_opt"])

    # 4. Run Iteration 2
    print("\n--- Solving with Iteration 2 (GELU + Softplus Baseline) ---")
    res_iter2 = solve_iter2(
        kkt_sys,
        max_epochs=1200,
        lr=0.015,
        verbose=False
    )
    metrics_iter2 = eval_solution(kkt_sys, res_iter2["x_opt"], res_iter2["lambda_opt"])

    # 5. Compute Objective Gaps
    gap_iter2 = abs(metrics_iter2["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_iter3 = abs(metrics_iter3["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_iter4 = abs(metrics_iter4["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0

    # 6. Print Comparative Scorecard
    print("\n" + "=" * 95)
    print(f"HEAD-TO-HEAD DRY RUN COMPARISON: '{kkt_sys.name}'")
    print("=" * 95)
    header = (
        f"{'Metric':<30} | {'HiGHS Ref':<14} | {'Iteration 2':<14} | "
        f"{'Iteration 3':<14} | {'Iteration 4':<14}"
    )
    print(header)
    print("-" * 95)

    t2_ms = res_iter2.get("pure_solve_time_ms", res_iter2.get("solve_time_sec", 0.0) * 1000.0)
    t3_ms = res_iter3.get("pure_solve_time_ms", res_iter3.get("solve_time_sec", 0.0) * 1000.0)
    t4_ms = res_iter4.get("pure_solve_time_ms", res_iter4.get("solve_time_sec", 0.0) * 1000.0)

    rows = [
        ("Primal Objective", f"{highs_opt:.2f}", f"{metrics_iter2['primal_objective']:.2f}", f"{metrics_iter3['primal_objective']:.2f}", f"{metrics_iter4['primal_objective']:.2f}"),
        ("Relative Objective Gap %", "Reference", f"{gap_iter2:.2f}%", f"{gap_iter3:.2f}%", f"{gap_iter4:.2f}%"),
        ("Max Primal Infeasibility", "0.00e+00", f"{metrics_iter2['max_primal_violation']:.2e}", f"{metrics_iter3['max_primal_violation']:.2e}", f"{metrics_iter4['max_primal_violation']:.2e}"),
        ("Dual Feasibility Viol", "0.00e+00", f"{metrics_iter2['max_dual_violation']:.2e}", f"{metrics_iter3['max_dual_violation']:.2e}", f"{metrics_iter4['max_dual_violation']:.2e}"),
        ("Stationarity Residual", "0.00e+00", f"{metrics_iter2['res_stationarity']:.2e}", f"{metrics_iter3['res_stationarity']:.2e}", f"{metrics_iter4['res_stationarity']:.2e}"),
        ("Complementary Slackness", "0.00e+00", f"{metrics_iter2['res_slackness']:.2e}", f"{metrics_iter3['res_slackness']:.2e}", f"{metrics_iter4['res_slackness']:.2e}"),
        ("Duality Gap", "0.00e+00", "N/A", "N/A", f"{metrics_iter4['duality_gap']:.2f}"),
        ("Pure Solve Time (ms)", "0.42 ms", f"{t2_ms:.2f} ms", f"{t3_ms:.2f} ms", f"{t4_ms:.2f} ms")
    ]

    for label, ref_val, v2, v3, v4 in rows:
        print(f"{label:<30} | {ref_val:<14} | {v2:<14} | {v3:<14} | {v4:<14}")
    print("=" * 95)

    # Save convergence plot
    plot_path = os.path.join(current_dir, "convergence_2_production_plan.png")
    plot_iteration_4_convergence(
        res_iter4["history"],
        save_path=plot_path,
        title="Iteration 4 (Preconditioned + Strong Duality) Convergence: 2_production_plan"
    )
    print(f"\nConvergence trajectory plot saved to: {plot_path}")


if __name__ == "__main__":
    run_dry_test()
