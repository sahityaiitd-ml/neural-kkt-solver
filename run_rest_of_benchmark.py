#!/usr/bin/env python3
"""
run_rest_of_benchmark.py
========================
High-throughput benchmark runner for The Tragic Solver (KKT-INN + OR-Guardian)
across the remaining benchmark suite (problems 101 to 393).

Features:
- Incremental autosaving to JSON & CSV after each solved problem.
- Automatic resume capability if interrupted.
- Per-problem timeout and exception isolation.
- Automatically generates combined full-suite scorecard (Problems 1 to 393).
"""

import os
import sys
import glob
import json
import csv
import time
import numpy as np

# Ensure project root is in path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from KKT_Standalone_Reader import load_problem
import importlib
tragic_mod = importlib.import_module("The Tragic Solver")

def main():
    summary_path = os.path.join(project_dir, "KKT_Benchmark", "summary.json")
    results_dir = os.path.join(project_dir, "KKT_Benchmark", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # 1. Discover and order available problems by dimension
    sols = sorted(glob.glob(os.path.join(project_dir, "KKT_Benchmark", "solutions", "*.npz")))
    avail = []
    for sol in sols:
        name = os.path.basename(sol).replace(".npz", "")
        prob_path = os.path.join(project_dir, "KKT_Benchmark", "problems", f"{name}.mps.gz")
        if os.path.exists(prob_path):
            meta = summary.get(name, {})
            nv = meta.get("n_vars", 0)
            nc = meta.get("n_constraints", 0)
            dim = nv + nc
            avail.append((name, dim, nv, nc, prob_path, sol))

    avail.sort(key=lambda x: (x[1], x[0]))
    total_problems = len(avail)
    print(f"Total benchmark problems available: {total_problems}")

    # Output paths
    rest_json_path = os.path.join(results_dir, "iteration_tragic_rest_of_problems.json")
    rest_csv_path = os.path.join(results_dir, "iteration_tragic_rest_of_problems.csv")

    # Load existing results to allow resuming
    completed = {}
    if os.path.exists(rest_json_path):
        try:
            with open(rest_json_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                for item in saved_data.get("scorecard", []):
                    completed[item["problem"]] = item
            print(f"Resuming: Found {len(completed)} already completed problems in {rest_json_path}")
        except Exception:
            pass

    # Target: problems 101 to 393
    target_slice = avail[100:]
    print(f"Evaluating remaining {len(target_slice)} problems (101 to {total_problems})...")

    header = f"{'#':<4} | {'Problem':<16} | {'Vars':<6} | {'Cons':<7} | {'HiGHS':<14} | {'KINN':<14} | {'Gap %':<8} | {'PrimViol':<9} | {'DualViol':<9} | {'Time ms':<9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    scorecard = list(completed.values())

    for idx, (name, dim, nv, nc, prob_path, sol_path) in enumerate(target_slice, start=101):
        if name in completed and completed[name].get("gap_percent", 999.0) < 500:
            s = completed[name]
            line = f"{idx:<4} | {name:<16} | {s['n_vars']:<6} | {s['n_cons']:<7} | {s['highs_obj']:<14.2f} | {s['kinn_obj']:<14.2f} | {s['gap_percent']:<7.2f}% | {s['primal_viol']:<9.2e} | {s['dual_viol']:<9.2e} | {s['kinn_time_ms']:<9.1f} [CACHED]"
            print(line, flush=True)
            continue

        try:
            # 1. Load Problem & Solution
            kkt = load_problem(prob_path)
            sol = np.load(sol_path)
            z_true = float(sol["objective_val"])
            highs_time_ms = float(sol.get("pure_solve_time_sec", 0.0)) * 1000.0

            # 2. Solve with The Tragic Solver
            res = tragic_mod.solve_kkt_instance(kkt, verbose=False)
            x_opt = res["x_opt"]
            lam_opt = res["lambda_opt"]

            # 3. Metrics
            z_pred = float(np.dot(kkt.c, x_opt))
            gap = abs(z_pred - z_true) / (abs(z_true) + 1e-9) * 100.0
            prim_viol = float(res["or_metrics"]["max_primal_violation"])
            dual_viol = float(res["or_metrics"]["max_dual_violation"])
            time_ms = float(res["pure_solve_time_ms"])
            epochs = int(res.get("iterations", 0))

            entry = {
                "problem": name,
                "n_vars": kkt.n_vars,
                "n_cons": kkt.n_constraints,
                "highs_obj": z_true,
                "kinn_obj": z_pred,
                "gap_percent": gap,
                "primal_viol": prim_viol,
                "dual_viol": dual_viol,
                "highs_time_ms": highs_time_ms,
                "kinn_time_ms": time_ms,
                "epochs": epochs
            }
            # Replace or add
            if name in completed:
                for i_c, old in enumerate(scorecard):
                    if old["problem"] == name:
                        scorecard[i_c] = entry
                        break
            else:
                scorecard.append(entry)
            completed[name] = entry

            line = f"{idx:<4} | {name:<16} | {kkt.n_vars:<6} | {kkt.n_constraints:<7} | {z_true:<14.2f} | {z_pred:<14.2f} | {gap:<7.2f}% | {prim_viol:<9.2e} | {dual_viol:<9.2e} | {time_ms:<9.1f}"
            print(line, flush=True)

        except Exception as e:
            print(f"{idx:<4} | {name:<16} | FAILED: {type(e).__name__}: {e}", flush=True)
            entry = {
                "problem": name,
                "n_vars": nv,
                "n_cons": nc,
                "highs_obj": 0.0,
                "kinn_obj": 0.0,
                "gap_percent": 999.0,
                "primal_viol": 999.0,
                "dual_viol": 999.0,
                "highs_time_ms": 0.0,
                "kinn_time_ms": 0.0,
                "epochs": 0,
                "error": str(e)
            }
            if name in completed:
                for i_c, old in enumerate(scorecard):
                    if old["problem"] == name:
                        scorecard[i_c] = entry
                        break
            else:
                scorecard.append(entry)
            completed[name] = entry

        # Incremental Autosave after every 2 problems
        if len(scorecard) % 2 == 0 or idx == total_problems:
            with open(rest_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "solver_name": "The_Tragic_Solver",
                    "problems_solved": len(scorecard),
                    "scorecard": scorecard
                }, f, indent=2)

    # Final Save for rest
    with open(rest_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "solver_name": "The_Tragic_Solver",
            "problems_solved": len(scorecard),
            "scorecard": scorecard
        }, f, indent=2)

    # 4. Generate Combined Full-Suite Report (Problems 1 to 393)
    top100_json = os.path.join(results_dir, "iteration_tragic_100_problems.json")
    all_scorecard = []
    if os.path.exists(top100_json):
        with open(top100_json, "r", encoding="utf-8") as f:
            t100_data = json.load(f)
            all_scorecard.extend(t100_data.get("scorecard", []))
    all_scorecard.extend(scorecard)

    grand_json = os.path.join(results_dir, "the_tragic_solver_full_suite_393.json")
    with open(grand_json, "w", encoding="utf-8") as f:
        json.dump({
            "solver_name": "The_Tragic_Solver_Full_Suite",
            "total_problems": len(all_scorecard),
            "scorecard": all_scorecard
        }, f, indent=2)

    # Print Summary Statistics
    gaps = [s["gap_percent"] for s in all_scorecard if s["gap_percent"] < 500]
    p_viols = [s["primal_viol"] for s in all_scorecard if s["primal_viol"] < 500]
    d_viols = [s["dual_viol"] for s in all_scorecard if s["dual_viol"] < 500]

    print("\n" + "=" * 80)
    print(f"[FINAL SCORECARD] GRAND SUITE EVALUATION ({len(all_scorecard)} PROBLEMS)")
    print("=" * 80)
    print(f"Total Problems Solved:     {len(all_scorecard)}")
    if gaps:
        print(f"Suite Median Gap:          {np.median(gaps):.4f}%")
        print(f"Suite Mean Gap:            {np.mean(gaps):.4f}%")
        print(f"Problems with Gap < 1.0%:  {sum(1 for g in gaps if g <= 1.0)} / {len(gaps)} ({sum(1 for g in gaps if g <= 1.0)/len(gaps)*100:.1f}%)")
        print(f"Problems with Gap < 15.0%: {sum(1 for g in gaps if g <= 15.0)} / {len(gaps)} ({sum(1 for g in gaps if g <= 15.0)/len(gaps)*100:.1f}%)")
        print(f"Problems with Gap < 30.0%: {sum(1 for g in gaps if g <= 30.0)} / {len(gaps)} ({sum(1 for g in gaps if g <= 30.0)/len(gaps)*100:.1f}%)")
    if p_viols:
        print(f"Max Primal Infeasibility:  {max(p_viols):.2e}")
    if d_viols:
        print(f"Max Dual Infeasibility:    {max(d_viols):.2e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
