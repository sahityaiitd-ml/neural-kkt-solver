"""
run_iteration_3.py
==================
Standalone runner for Iteration 3 (Path B: Fischer-Burmeister).
Executes a dry-run on sample canonical LPs and Netlib instances.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_3 import solve_kkt_instance, evaluate_solution
from KKT_Solver_Iteration_3.plot_loss import plot_iteration_3_convergence


def run_dry_test():
    print("=" * 80)
    print("RUNNING ITERATION 3 ARCHITECTURE DRY RUN")
    print("Formulation: Path B - Smoothed Fischer-Burmeister Complementarity")
    print("=" * 80)

    # Test on standard production planning LP
    prob_path = os.path.join(project_root, "KKT_Standalone_Reader/examples/2_production_plan.lp")
    if not os.path.exists(prob_path):
        print(f"Error: Problem file not found at {prob_path}")
        return

    print(f"\n[1/1] Ingesting Problem: {os.path.basename(prob_path)}")
    kkt_sys = load_problem(prob_path)
    print(f"Loaded {kkt_sys.name} (n={kkt_sys.n_vars}, m={kkt_sys.n_constraints})")

    # Execute solve
    result = solve_kkt_instance(
        kkt_sys,
        model_type="neural",
        max_epochs=1200,
        lr=0.015,
        w_stat=1.0,
        w_fb=5.0,
        w_prim=1.0,
        w_primal_pos=5.0,
        verbose=True
    )

    # Evaluate solution
    eval_res = evaluate_solution(kkt_sys, result["x_opt"], result["lambda_opt"])

    print("\n" + "=" * 65)
    print(f"DRY RUN SOLUTION EVALUATION: '{kkt_sys.name}'")
    print("=" * 65)
    print(f"Primal Objective Value:        {eval_res['primal_objective']:>15.4f}")
    print(f"Max Primal Violation:          {eval_res['max_primal_violation']:>15.4e}")
    print(f"Max Primal Negative:           {eval_res['max_x_neg']:>15.4e}")
    print(f"Max Dual Violation:            {eval_res['max_dual_violation']:>15.4e}")
    print(f"Stationarity Residual:         {eval_res['res_stationarity']:>15.4e}")
    print(f"Fischer-Burmeister Residual:   {eval_res['res_fb']:>15.4e}")
    print(f"Complementary Slackness:       {eval_res['res_slackness']:>15.4e}")
    print(f"Solve Time (Pure Algorithmic): {result['solve_time_sec']*1000:>15.2f} ms")
    print("=" * 65)

    plot_path = os.path.join(current_dir, "convergence_2_production_plan.png")
    plot_iteration_3_convergence(
        result["history"],
        save_path=plot_path,
        title="Iteration 3 (Fischer-Burmeister) Convergence: 2_production_plan"
    )
    print(f"Convergence plot saved to: {plot_path}")


if __name__ == "__main__":
    run_dry_test()
