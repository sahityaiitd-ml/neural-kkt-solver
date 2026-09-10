"""
benchmark_harness.py
====================
Universal evaluation harness that benchmarks ANY neural solver iteration
(Iteration 1, Iteration 2, etc.) against the official Netlib & MIPLIB suite.

Timing Protocol:
----------------
Problem files and ground-truth arrays are completely loaded into RAM first.
Timing begins ONLY when the neural solver begins optimizing:
    t0 = time.perf_counter()
    result = solver_function(kkt_system)
    pure_nn_time = time.perf_counter() - t0
"""

import os
import sys
import glob
import time
import json
from typing import Optional, List, Callable
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem


def evaluate_solver_on_benchmark(
    solver_fn: Callable,
    solver_name: str = "Iteration_1",
    problems_dir: Optional[str] = None,
    solutions_dir: Optional[str] = None,
    tier: Optional[str] = None,
    max_problems: Optional[int] = None,
    problem_names: Optional[List[str]] = None,
    verbose: bool = True
):
    """
    Runs a KINN solver across Netlib/MIPLIB benchmarks and reports accuracy & speed.

    Parameters:
    -----------
    solver_fn : callable
        A function matching signature: solver_fn(kkt_sys, **kwargs) -> dict
        returning dict with "x_opt", "lambda_opt", "iterations".
    solver_name : str
        Label for the report (e.g. "Iteration_1_Basic_ReLU").
    tier : str, optional
        Filter by size tier: "small" (<=1k), "medium" (1k-5k), "large" (5k-15k).
    max_problems : int, optional
        Maximum number of problems to evaluate.
    problem_names : list of str, optional
        Explicit list of problem names to evaluate (e.g. ["afiro", "adlittle", "e226"]).
    """
    if problems_dir is None:
        problems_dir = os.path.join(current_dir, "problems")
    if solutions_dir is None:
        solutions_dir = os.path.join(current_dir, "solutions")

    summary_path = os.path.join(current_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Gather available solutions
    available_sols = sorted(glob.glob(os.path.join(solutions_dir, "*.npz")))
    if not available_sols:
        raise FileNotFoundError(f"No solution files found in {solutions_dir}!")

    # Filter target problems
    targets = []
    for sol_path in available_sols:
        clean_name = os.path.basename(sol_path).replace(".npz", "")
        prob_file = os.path.join(problems_dir, f"{clean_name}.mps.gz")
        if not os.path.exists(prob_file):
            continue

        if problem_names is not None and clean_name not in problem_names:
            continue

        meta = summary.get(clean_name, {})
        if tier is not None and meta.get("tier") != tier:
            continue

        targets.append((clean_name, prob_file, sol_path))

    if max_problems is not None:
        targets = targets[:max_problems]

    print("=" * 90)
    print(f"🏁 RUNNING BENCHMARK HARNESS ON: {solver_name}")
    print(f"   Target Problem Set: {len(targets)} problems (Tier: {tier or 'all available'})")
    print("   Timing: Pure algorithmic solve time only (zero loading/file I/O)")
    print("=" * 90)

    scorecard = []

    for idx, (clean_name, filepath, sol_path) in enumerate(targets, 1):
        # 1. Load Problem and Ground Truth into RAM (NOT TIMED)
        kkt_sys = load_problem(filepath)
        ground_truth = np.load(sol_path)

        x_true = ground_truth["x_star"]
        lam_true = ground_truth["lambda_star"]
        z_true = float(ground_truth["objective_val"])
        highs_time_ms = float(ground_truth["pure_solve_time_sec"]) * 1000.0

        if verbose:
            print(f"[{idx:2d}/{len(targets):2d}] ⚡ Solving '{clean_name}' ({kkt_sys.n_vars} vars, {kkt_sys.n_constraints} cons)...")

        # 2. Pure Solve Timer around Neural Solver
        t_start = time.perf_counter()
        result = solver_fn(kkt_sys)
        pure_nn_time_sec = time.perf_counter() - t_start
        pure_nn_time_ms = pure_nn_time_sec * 1000.0

        # 3. Extract Predictions
        x_hat = result["x_opt"]
        lambda_hat = result["lambda_opt"]
        epochs = result.get("iterations", 0)

        # 4. Compute Metrics
        c, G, h = kkt_sys.c, kkt_sys.G, kkt_sys.h
        z_hat = float(np.dot(c, x_hat))
        gap_percent = (abs(z_hat - z_true) / (abs(z_true) + 1e-9)) * 100.0

        slack = np.dot(G, x_hat) - h
        max_primal_violation = float(np.max(np.maximum(0.0, slack)))
        max_dual_violation = float(np.max(np.maximum(0.0, -lambda_hat)))
        res_stationarity = float(np.linalg.norm(c + np.dot(G.T, lambda_hat)))
        res_slackness = float(np.linalg.norm(lambda_hat * slack))

        scorecard.append({
            "problem": clean_name,
            "n_vars": kkt_sys.n_vars,
            "n_cons": kkt_sys.n_constraints,
            "highs_obj": z_true,
            "kinn_obj": z_hat,
            "gap_percent": gap_percent,
            "primal_viol": max_primal_violation,
            "dual_viol": max_dual_violation,
            "stat_error": res_stationarity,
            "slack_error": res_slackness,
            "highs_time_ms": highs_time_ms,
            "kinn_time_ms": pure_nn_time_ms,
            "epochs": epochs
        })

    # 5. Print Consolidated Scorecard Table
    print("\n" + "=" * 95)
    print(f"🏆 OFFICIAL KKT BENCHMARK SCORECARD: {solver_name}")
    print("=" * 95)
    header = (
        f"{'Problem':<14} | {'Vars':<5} | {'Cons':<6} | "
        f"{'HiGHS Obj':<13} | {'KINN Obj':<13} | {'Gap %':<8} | "
        f"{'Prim Viol':<10} | {'HiGHS ms':<9} | {'KINN ms':<9}"
    )
    print(header)
    print("-" * 95)

    for s in scorecard:
        line = (
            f"{s['problem']:<14} | "
            f"{s['n_vars']:<5} | "
            f"{s['n_cons']:<6} | "
            f"{s['highs_obj']:<13.2f} | "
            f"{s['kinn_obj']:<13.2f} | "
            f"{s['gap_percent']:<8.2f}% | "
            f"{s['primal_viol']:<10.2e} | "
            f"{s['highs_time_ms']:<9.2f} | "
            f"{s['kinn_time_ms']:<9.2f}"
        )
        print(line)
    print("=" * 95)

    return scorecard


if __name__ == "__main__":
    from KKT_Solver_Iteration_1 import solve_kkt_instance

    def run_iteration_1_wrapper(kkt_sys):
        return solve_kkt_instance(
            kkt_sys,
            model_type="neural",
            max_epochs=1200,
            lr=0.015,
            verbose=False
        )

    # Example: Run on standard benchmark sample across Netlib and MIPLIB
    sample_problems = ["afiro", "flugpl", "adlittle", "bell5", "e226", "blend", "share2b", "sc50a"]
    evaluate_solver_on_benchmark(
        solver_fn=run_iteration_1_wrapper,
        solver_name="KKT_Solver_Iteration_1 (Basic ReLU Baseline)",
        problem_names=sample_problems
    )
