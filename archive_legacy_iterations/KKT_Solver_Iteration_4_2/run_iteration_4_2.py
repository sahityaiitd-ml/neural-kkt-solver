"""
run_iteration_4_2.py
====================
Dry run verification and multi-iteration comparison for Iteration 4.2.
Tests 2_production_plan.lp and origin-trap benchmarks (sc50a, afiro, blend).
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
from KKT_Solver_Iteration_4_2 import solve_kkt_instance as solve_iter4_2, evaluate_solution as eval_solution
from KKT_Solver_Iteration_4_2.plot_loss import plot_iteration_4_2_convergence
from KKT_Solver_Iteration_4_1 import solve_kkt_instance as solve_iter4_1
from KKT_Solver_Iteration_4 import solve_kkt_instance as solve_iter4
from KKT_Solver_Iteration_3 import solve_kkt_instance as solve_iter3
from KKT_Solver_Iteration_2 import solve_kkt_instance as solve_iter2


def run_dry_test():
    print("=" * 110)
    print("RUNNING ITERATION 4.2 DRY RUN (ANNEALED PULL + ONE-SIDED DUALITY GAP)")
    print("=" * 110)

    prob_path = os.path.join(project_root, "KKT_Standalone_Reader/examples/2_production_plan.lp")
    if not os.path.exists(prob_path):
        print(f"Error: Problem file not found at {prob_path}")
        return

    kkt_sys = load_problem(prob_path)
    highs_opt = -21253.3333

    print(f"\n[TEST 1] Canonical Production Plan ({kkt_sys.n_vars} vars, {kkt_sys.n_constraints} cons)")
    res_4_2 = solve_iter4_2(kkt_sys, max_epochs=1200, lr=0.015, verbose=True)
    m_4_2 = eval_solution(kkt_sys, res_4_2["x_opt"], res_4_2["lambda_opt"])

    res_4_1 = solve_iter4_1(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_4_1 = eval_solution(kkt_sys, res_4_1["x_opt"], res_4_1["lambda_opt"])

    res_4 = solve_iter4(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_4 = eval_solution(kkt_sys, res_4["x_opt"], res_4["lambda_opt"])

    res_3 = solve_iter3(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_3 = eval_solution(kkt_sys, res_3["x_opt"], res_3["lambda_opt"])

    res_2 = solve_iter2(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)
    m_2 = eval_solution(kkt_sys, res_2["x_opt"], res_2["lambda_opt"])

    gap_2 = abs(m_2["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_3 = abs(m_3["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_4 = abs(m_4["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_4_1 = abs(m_4_1["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0
    gap_4_2 = abs(m_4_2["primal_objective"] - highs_opt) / abs(highs_opt) * 100.0

    print("\n" + "=" * 110)
    print("HEAD-TO-HEAD DRY RUN SCORECARD: '2_production_plan'")
    print("=" * 110)
    header = f"{'Metric':<25} | {'HiGHS Ref':<11} | {'Iter 2':<11} | {'Iter 3':<11} | {'Iter 4':<11} | {'Iter 4.1':<11} | {'Iter 4.2':<11}"
    print(header)
    print("-" * 110)

    rows = [
        ("Primal Objective", f"{highs_opt:.2f}", f"{m_2['primal_objective']:.2f}", f"{m_3['primal_objective']:.2f}", f"{m_4['primal_objective']:.2f}", f"{m_4_1['primal_objective']:.2f}", f"{m_4_2['primal_objective']:.2f}"),
        ("Relative Gap %", "Reference", f"{gap_2:.2f}%", f"{gap_3:.2f}%", f"{gap_4:.2f}%", f"{gap_4_1:.2f}%", f"{gap_4_2:.2f}%"),
        ("Primal Infeasibility", "0.00e+00", f"{m_2['max_primal_violation']:.2e}", f"{m_3['max_primal_violation']:.2e}", f"{m_4['max_primal_violation']:.2e}", f"{m_4_1['max_primal_violation']:.2e}", f"{m_4_2['max_primal_violation']:.2e}"),
        ("Dual Feasibility Viol", "0.00e+00", f"{m_2['max_dual_violation']:.2e}", f"{m_3['max_dual_violation']:.2e}", f"{m_4['max_dual_violation']:.2e}", f"{m_4_1['max_dual_violation']:.2e}", f"{m_4_2['max_dual_violation']:.2e}"),
        ("Stationarity Residual", "0.00e+00", f"{m_2['res_stationarity']:.2e}", f"{m_3['res_stationarity']:.2e}", f"{m_4['res_stationarity']:.2e}", f"{m_4_1['res_stationarity']:.2e}", f"{m_4_2['res_stationarity']:.2e}"),
        ("Pure Solve Time (ms)", "0.42 ms", f"{res_2['solve_time_sec']*1000:.1f} ms", f"{res_3['solve_time_sec']*1000:.1f} ms", f"{res_4['pure_solve_time_ms']:.1f} ms", f"{res_4_1['pure_solve_time_ms']:.1f} ms", f"{res_4_2['pure_solve_time_ms']:.1f} ms"),
    ]

    for r in rows:
        print(f"{r[0]:<25} | {r[1]:<11} | {r[2]:<11} | {r[3]:<11} | {r[4]:<11} | {r[5]:<11} | {r[6]:<11}")
    print("=" * 110)

    # Plot convergence
    plot_path = os.path.join(current_dir, "convergence_2_production_plan.png")
    plot_iteration_4_2_convergence(
        res_4_2["history"],
        save_path=plot_path,
        title="Iteration 4.2 (Annealed Pull + One-Sided Gap): 2_production_plan"
    )

    # Check origin trap problems
    print("\n" + "=" * 110)
    print("ORIGIN TRAP PROBLEMS TEST (sc50a, afiro, blend, flugpl, gen-ip002)")
    print("=" * 110)
    print(f"{'Problem':<12} | {'HiGHS Obj':<12} | {'Iter 4.1 Obj':<14} | {'Iter 4.1 Gap':<12} | {'Iter 4.2 Obj':<14} | {'Iter 4.2 Gap':<12} | {'Iter 4.2 Viol':<12}")
    print("-" * 110)

    trap_probs = ["sc50a", "afiro", "blend", "flugpl", "gen-ip002"]
    for name in trap_probs:
        prob_mps = os.path.join(project_root, f"KKT_Benchmark/problems/{name}.mps.gz")
        sol_npz = os.path.join(project_root, f"KKT_Benchmark/solutions/{name}.npz")
        p = load_problem(prob_mps)
        sol = np.load(sol_npz)
        z_true = float(sol["objective_val"])

        r41 = solve_iter4_1(p, max_epochs=1200, lr=0.015, verbose=False)
        m41 = eval_solution(p, r41["x_opt"], r41["lambda_opt"])
        gap41 = abs(m41["primal_objective"] - z_true) / abs(z_true) * 100.0

        r42 = solve_iter4_2(p, max_epochs=1200, lr=0.015, verbose=False)
        m42 = eval_solution(p, r42["x_opt"], r42["lambda_opt"])
        gap42 = abs(m42["primal_objective"] - z_true) / abs(z_true) * 100.0

        print(f"{name:<12} | {z_true:<12.2f} | {m41['primal_objective']:<14.2f} | {gap41:<11.2f}% | {m42['primal_objective']:<14.2f} | {gap42:<11.2f}% | {m42['max_primal_violation']:<12.2e}")
    print("=" * 110)


if __name__ == "__main__":
    run_dry_test()
