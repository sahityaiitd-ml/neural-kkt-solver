"""
run_iteration_4.py
==================
Demonstration test and multi-iteration comparison for Iteration 4
(Decoupled Architecture with Linear Objective Pull).
"""

import os
import sys
import numpy as np

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
    print("=" * 85)
    print("RUNNING ITERATION 4 DRY RUN (DECOUPLED ARCHITECTURE + LINEAR OBJECTIVE PULL)")
    print("=" * 85)

    prob_path = os.path.join(project_root, "KKT_Standalone_Reader/examples/2_production_plan.lp")
    if not os.path.exists(prob_path):
        print(f"Error: Problem file not found at {prob_path}")
        return

    kkt_sys = load_problem(prob_path)
    highs_opt = -21253.3333

    print(f"\n[TEST 1] Canonical Production Plan ({kkt_sys.n_vars} vars, {kkt_sys.n_constraints} cons)")
    res_4 = solve_iter4(kkt_sys, max_epochs=1200, lr=0.015, verbose=True)
    m_4 = eval_solution(kkt_sys, res_4["x_opt"], res_4["lambda_opt"])

    res_3 = solve_iter3(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_3 = eval_solution(kkt_sys, res_3["x_opt"], res_3["lambda_opt"])

    res_2 = solve_iter2(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_2 = eval_solution(kkt_sys, res_2["x_opt"], res_2["lambda_opt"])

    gap_2 = abs(m_2["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_3 = abs(m_3["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_4 = abs(m_4["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0

    print("\n" + "=" * 85)
    print("HEAD-TO-HEAD SCORECARD: '2_production_plan'")
    print("=" * 85)
    header = f"{'Metric':<26} | {'HiGHS Ref':<12} | {'Iter 2':<12} | {'Iter 3':<12} | {'Iter 4':<12}"
    print(header)
    print("-" * 85)

    rows = [
        ("Primal Objective", f"{highs_opt:.2f}", f"{m_2['primal_objective']:.2f}", f"{m_3['primal_objective']:.2f}", f"{m_4['primal_objective']:.2f}"),
        ("Relative Gap %", "Reference", f"{gap_2:.2f}%", f"{gap_3:.2f}%", f"{gap_4:.2f}%"),
        ("Primal Infeasibility", "0.00e+00", f"{m_2['max_primal_violation']:.2e}", f"{m_3['max_primal_violation']:.2e}", f"{m_4['max_primal_violation']:.2e}"),
        ("Dual Feasibility Viol", "0.00e+00", f"{m_2['max_dual_violation']:.2e}", f"{m_3['max_dual_violation']:.2e}", f"{m_4['max_dual_violation']:.2e}"),
        ("Stationarity Residual", "0.00e+00", f"{m_2['res_stationarity']:.2e}", f"{m_3['res_stationarity']:.2e}", f"{m_4['res_stationarity']:.2e}"),
        ("Duality Gap", "0.00e+00", "N/A", "N/A", f"{m_4['duality_gap']:.2f}"),
        ("Pure Solve Time (ms)", "0.42 ms", f"{res_2['solve_time_sec']*1000:.1f} ms", f"{res_3['solve_time_sec']*1000:.1f} ms", f"{res_4['pure_solve_time_ms']:.1f} ms"),
    ]

    for r in rows:
        print(f"{r[0]:<26} | {r[1]:<12} | {r[2]:<12} | {r[3]:<12} | {r[4]:<12}")
    print("=" * 85)

    # Plot convergence
    plot_path = os.path.join(current_dir, "convergence_2_production_plan.png")
    plot_iteration_4_convergence(
        res_4["history"],
        save_path=plot_path,
        title="Iteration 4 (Decoupled + Objective Pull): 2_production_plan"
    )
    print(f"Convergence plot saved to: {plot_path}")


if __name__ == "__main__":
    run_dry_test()
