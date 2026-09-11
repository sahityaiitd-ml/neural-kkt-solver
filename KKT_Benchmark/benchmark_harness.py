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
import csv
from typing import Optional, List, Callable, Dict, Any
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
    verbose: bool = True,
    save_json: bool = True,
    save_csv: bool = True,
    results_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Runs a KINN solver across Netlib/MIPLIB benchmarks and reports accuracy & speed.
    Automatically saves benchmark results to JSON and CSV for reusability.

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
    verbose : bool
        If True, logs per-problem progress during the run.
    save_json : bool
        If True, saves full run metadata and problem scorecard to a JSON file.
    save_csv : bool
        If True, saves tabular metrics to a CSV file.
    results_dir : str, optional
        Target directory to archive benchmark results. Defaults to KKT_Benchmark/results.
    filename : str, optional
        Custom filename (without extension) for the saved files. If None, derived from solver_name.
    """
    if problems_dir is None:
        problems_dir = os.path.join(current_dir, "problems")
    if solutions_dir is None:
        solutions_dir = os.path.join(current_dir, "solutions")
    if results_dir is None:
        results_dir = os.path.join(current_dir, "results")

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

    print("=" * 95)
    print(f"[BENCHMARK] RUNNING BENCHMARK HARNESS ON: {solver_name}")
    print(f"   Target Problem Set: {len(targets)} problems (Tier: {tier or 'all available'})")
    print("   Timing: Pure algorithmic solve time only (zero loading/file I/O)")
    print("=" * 95)

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
            print(f"[{idx:2d}/{len(targets):2d}] Solving '{clean_name}' ({kkt_sys.n_vars} vars, {kkt_sys.n_constraints} cons)...")

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
    print(f"[SCORECARD] OFFICIAL KKT BENCHMARK SCORECARD: {solver_name}")
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

    # 6. Save Results to Disk
    if save_json or save_csv:
        os.makedirs(results_dir, exist_ok=True)
        if filename is None:
            clean_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in solver_name.lower())
            base_fname = clean_base.strip("_")
        else:
            base_fname = filename.replace(".json", "").replace(".csv", "")

        gaps = [s["gap_percent"] for s in scorecard]
        times = [s["kinn_time_ms"] for s in scorecard]
        prim_viols = [s["primal_viol"] for s in scorecard]
        dual_viols = [s["dual_viol"] for s in scorecard]

        summary_data = {
            "num_problems": len(scorecard),
            "mean_gap_percent": float(np.mean(gaps)) if gaps else 0.0,
            "median_gap_percent": float(np.median(gaps)) if gaps else 0.0,
            "min_gap_percent": float(np.min(gaps)) if gaps else 0.0,
            "max_gap_percent": float(np.max(gaps)) if gaps else 0.0,
            "mean_primal_violation": float(np.mean(prim_viols)) if prim_viols else 0.0,
            "max_primal_violation": float(np.max(prim_viols)) if prim_viols else 0.0,
            "mean_dual_violation": float(np.mean(dual_viols)) if dual_viols else 0.0,
            "max_dual_violation": float(np.max(dual_viols)) if dual_viols else 0.0,
            "mean_time_ms": float(np.mean(times)) if times else 0.0,
            "total_time_ms": float(np.sum(times)) if times else 0.0
        }

        if save_json:
            json_file = os.path.join(results_dir, f"{base_fname}.json")
            payload = {
                "solver_name": solver_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tier": tier,
                "summary": summary_data,
                "scorecard": scorecard
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[SAVED] JSON benchmark results saved to: {json_file}")

        if save_csv:
            csv_file = os.path.join(results_dir, f"{base_fname}.csv")
            fields = [
                "problem", "n_vars", "n_cons", "highs_obj", "kinn_obj",
                "gap_percent", "primal_viol", "dual_viol", "stat_error",
                "slack_error", "highs_time_ms", "kinn_time_ms", "epochs"
            ]
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in scorecard:
                    writer.writerow(row)
            print(f"[SAVED] CSV benchmark results saved to: {csv_file}")

    return scorecard


def load_benchmark_result(
    path_or_name: str,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads a saved benchmark result dictionary from disk.
    Accepts either an absolute/relative path or a filename without extension in results_dir.
    """
    if results_dir is None:
        results_dir = os.path.join(current_dir, "results")

    if os.path.isfile(path_or_name):
        target_path = path_or_name
    else:
        candidate = os.path.join(results_dir, path_or_name if path_or_name.endswith(".json") else f"{path_or_name}.json")
        if os.path.isfile(candidate):
            target_path = candidate
        else:
            raise FileNotFoundError(f"Benchmark result file not found: {path_or_name} in {results_dir}")

    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_saved_benchmarks(
    run_identifiers: List[str],
    results_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Compares two or more saved benchmark JSON files side-by-side without re-running solvers.

    Parameters:
    -----------
    run_identifiers : list of str
        List of filenames or paths (e.g. ["iteration_1_25_problems", "iteration_2_25_problems", "iteration_3_25_problems"]).
    results_dir : str, optional
        Directory where JSON files are located. Defaults to KKT_Benchmark/results.
    """
    runs = [load_benchmark_result(rid, results_dir) for rid in run_identifiers]
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to perform a comparison!")

    names = [r["solver_name"] for r in runs]
    cards = [{item["problem"]: item for item in r["scorecard"]} for r in runs]

    # Shared problems across all runs
    common_problems = sorted(list(set.intersection(*[set(c.keys()) for c in cards])))

    print("\n" + "=" * 115)
    print(f"[COMPARISON] MULTI-ITERATION BENCHMARK COMPARISON ({len(common_problems)} Common Problems)")
    for i, name in enumerate(names, 1):
        print(f"   [{i}] {name}")
    print("=" * 115)

    # Header
    col_headers = ["Problem", "Vars", "Cons"]
    for i in range(len(runs)):
        col_headers.append(f"Iter {i+1} Gap%")
    col_headers.append("Best Gap")
    col_headers.append("Best Solver")

    header_line = f"{col_headers[0]:<14} | {col_headers[1]:<5} | {col_headers[2]:<6} | "
    for i in range(len(runs)):
        header_line += f"{col_headers[3+i]:<12} | "
    header_line += f"{col_headers[-2]:<10} | {col_headers[-1]:<14}"
    print(header_line)
    print("-" * 115)

    comparison_records = []
    wins_per_run = [0] * len(runs)

    for prob in common_problems:
        prob_runs = [c[prob] for c in cards]
        n_v = prob_runs[0]["n_vars"]
        n_c = prob_runs[0]["n_cons"]
        gaps = [pr["gap_percent"] for pr in prob_runs]

        best_idx = int(np.argmin(gaps))
        wins_per_run[best_idx] += 1

        row_str = f"{prob:<14} | {n_v:<5} | {n_c:<6} | "
        for g in gaps:
            row_str += f"{g:<11.2f}% | "
        row_str += f"{gaps[best_idx]:<9.2f}% | Iteration {best_idx + 1}"
        print(row_str)

        comparison_records.append({
            "problem": prob,
            "n_vars": n_v,
            "n_cons": n_c,
            "gaps": gaps,
            "best_solver_index": best_idx + 1,
            "best_gap": gaps[best_idx]
        })

    print("=" * 115)
    print("[WIN SUMMARY]")
    for i, name in enumerate(names, 1):
        print(f"   * Iteration {i} ({name}): Won on {wins_per_run[i-1]} / {len(common_problems)} problems ({wins_per_run[i-1]/len(common_problems)*100:.1f}%)")
    print("=" * 115)

    return comparison_records


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
        solver_name="Iteration_1_Basic_ReLU",
        problem_names=sample_problems,
        filename="iteration_1_sample_test"
    )

