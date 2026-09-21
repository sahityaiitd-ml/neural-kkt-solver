"""
test_knobs.py
=============
Automated Diagnostic Ablation Matrix for KKT_Solver_Iteration_4_3_Prac.

Evaluates 4 configurations across the 6 archetype benchmark problems:
1. Config 1: row_l2 + fixed_decay (Iteration 4.2 Baseline)
2. Config 2: ruiz + fixed_decay (Ruiz Row & Column Equilibration)
3. Config 3: row_l2 + diameter_adaptive (Adaptive Annealing)
4. Config 4: ruiz + diameter_adaptive (Combined Ruiz + Adaptive Annealing)
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_4_3_Prac import solve_kkt_instance, evaluate_solution

archetype_problems = [
    ("2_production_plan", "KKT_Standalone_Reader/examples/2_production_plan.lp", -21253.3333),
    ("sc50a", "KKT_Benchmark/problems/sc50a.mps.gz", -64.575077),
    ("afiro", "KKT_Benchmark/problems/afiro.mps.gz", -464.753143),
    ("blend", "KKT_Benchmark/problems/blend.mps.gz", -30.812150),
    ("degen2", "KKT_Benchmark/problems/degen2.mps.gz", -1435.18),
    ("agg", "KKT_Benchmark/problems/agg.mps.gz", -35991767.29),
]

configs = [
    ("Config 1 (RowL2+Fixed)", "row_l2", "fixed_decay"),
    ("Config 2 (Ruiz+Fixed)", "ruiz", "fixed_decay"),
    ("Config 3 (RowL2+Adapt)", "row_l2", "diameter_adaptive"),
    ("Config 4 (Ruiz+Adapt)", "ruiz", "diameter_adaptive"),
]


def run_ablation_matrix():
    print("=" * 115)
    print("KKT_Solver_Iteration_4_3_Prac: ABLATION MATRIX ON 6 ARCHETYPE PROBLEMS")
    print("=" * 115)

    results = {}

    for prob_key, prob_path, z_true in archetype_problems:
        full_path = os.path.join(project_root, prob_path)
        if not os.path.exists(full_path):
            print(f"[SKIP] Problem file not found: {full_path}")
            continue

        prob = load_problem(full_path)
        print(f"\n[EVALUATING] {prob_key} ({prob.n_vars} vars, {prob.n_constraints} cons, HiGHS: {z_true:.2f})...")
        results[prob_key] = {"z_true": z_true, "configs": {}}

        for cfg_name, precond, anneal in configs:
            import torch
            torch.manual_seed(42)

            res = solve_kkt_instance(
                prob,
                model_type="neural",
                precond_method=precond,
                anneal_mode=anneal,
                max_epochs=1200,
                lr=0.015,
                verbose=False
            )
            m = evaluate_solution(prob, res["x_opt"], res["lambda_opt"])
            gap = abs(m["primal_objective"] - z_true) / abs(z_true) * 100.0
            viol = m["max_primal_violation"]
            results[prob_key]["configs"][cfg_name] = {
                "obj": m["primal_objective"],
                "gap": gap,
                "viol": viol,
                "time_ms": res["pure_solve_time_ms"]
            }
            print(f"   * {cfg_name:<25} -> Obj: {m['primal_objective']:>12.2f} | Gap: {gap:>7.2f}% | Viol: {viol:.2e}")

    # Consolidated Scorecard Table
    print("\n" + "=" * 115)
    print("CONSOLIDATED ABLATION SCORECARD (GAP % AND PRIMAL VIOLATION)")
    print("=" * 115)
    header = (
        f"{'Problem':<18} | {'HiGHS Ref':<11} | "
        f"{'Config 1 (Base)':<15} | {'Config 2 (Ruiz)':<15} | "
        f"{'Config 3 (Adapt)':<15} | {'Config 4 (Both)':<15} | {'Best Config':<15}"
    )
    print(header)
    print("-" * 115)

    for prob_key in results:
        z_t = results[prob_key]["z_true"]
        cfgs = results[prob_key]["configs"]

        gaps = {k: cfgs[k]["gap"] for k in cfgs}
        best_cfg = min(gaps, key=gaps.get)

        def cell(k):
            g = cfgs[k]["gap"]
            v = cfgs[k]["viol"]
            return f"{g:.2f}% ({v:.1e})"

        row = (
            f"{prob_key:<18} | {z_t:<11.2f} | "
            f"{cell(configs[0][0]):<15} | {cell(configs[1][0]):<15} | "
            f"{cell(configs[2][0]):<15} | {cell(configs[3][0]):<15} | "
            f"{best_cfg.split(' ')[1]:<15}"
        )
        print(row)
    print("=" * 115)


if __name__ == "__main__":
    run_ablation_matrix()
