"""
generate_solutions.py
=====================
Solves all Netlib & MIPLIB benchmark files using HiGHS to establish exact ground truth.

Strict Timing Protocol:
------------------------
The problem is completely parsed and loaded into RAM BEFORE the timer starts.
The timer measures ONLY the pure mathematical solve algorithm:
    timer_start = time.perf_counter()
    solver.run()
    pure_solve_time = time.perf_counter() - timer_start

Device Safety & Tiering:
------------------------
- Tier 1 (Small  <= 1k vars/cons):   Ultra-fast, ideal for rapid iteration
- Tier 2 (Medium 1k - 5k vars/cons): Standard evaluation set
- Tier 3 (Large  5k - 15k vars/cons):Large-scale stress test
- Tier 4 (Huge   > 15k vars/cons):   Cataloged but skipped from local solve to protect RAM
"""

import os
import sys
import glob
import time
import json
import numpy as np
import scipy.sparse as sp
import highspy

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from KKT_Standalone_Reader import load_problem


def solve_and_record_pure_time(kkt_sys):
    """
    Builds the model in memory using highspy C++ sparse arrays,
    then measures ONLY the solver execution time.
    """
    c = kkt_sys.c
    G = kkt_sys.G
    h_vec = kkt_sys.h
    n_vars = kkt_sys.n_vars
    n_cons = kkt_sys.n_constraints

    csr = sp.csr_matrix(G)
    h_solve = highspy.Highs()
    h_solve.setOptionValue("output_flag", False)
    h_solve.setOptionValue("time_limit", 30.0)

    # 1. Variables [-inf, +inf] because bounds are already in G * x <= h
    var_lower = np.full(n_vars, -np.inf, dtype=np.float64)
    var_upper = np.full(n_vars, np.inf, dtype=np.float64)
    h_solve.addVars(n_vars, var_lower, var_upper)

    cols = np.arange(n_vars, dtype=np.int32)
    h_solve.changeColsCost(n_vars, cols, np.ascontiguousarray(c, dtype=np.float64))

    # 2. Constraints: -inf <= G * x <= h
    row_lower_vec = np.full(n_cons, -np.inf, dtype=np.float64)
    row_upper_vec = np.ascontiguousarray(h_vec, dtype=np.float64)
    start = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    index = np.ascontiguousarray(csr.indices, dtype=np.int32)
    value = np.ascontiguousarray(csr.data, dtype=np.float64)
    num_nz = int(csr.nnz)

    h_solve.addRows(n_cons, row_lower_vec, row_upper_vec, num_nz, start, index, value)

    # 3. Pure mathematical solve (TIMED ONLY HERE)
    t0 = time.perf_counter()
    h_solve.run()
    pure_solve_time = time.perf_counter() - t0

    # 4. Extract solution vectors (NOT TIMED)
    status = str(h_solve.getModelStatus())
    sol = h_solve.getSolution()
    info = h_solve.getInfo()

    x_star = np.array(sol.col_value, dtype=np.float64)
    r_dual = np.array(sol.row_dual, dtype=np.float64)
    lambda_star = np.maximum(0.0, -r_dual)
    obj_val = float(info.objective_function_value)

    # 5. Mathematical KKT Residual Verification
    res_stationarity = float(np.linalg.norm(c + csr.T.dot(lambda_star)))
    slack = csr.dot(x_star) - h_vec
    res_primal = float(np.linalg.norm(np.maximum(0.0, slack)))
    res_dual = float(np.linalg.norm(np.maximum(0.0, -lambda_star)))
    res_slackness = float(np.linalg.norm(lambda_star * slack))
    total_kkt_error = float(res_stationarity + res_primal + res_dual + res_slackness)

    return {
        "status": status,
        "x_star": x_star,
        "lambda_star": lambda_star,
        "objective_val": obj_val,
        "pure_solve_time_sec": pure_solve_time,
        "kkt_error": total_kkt_error
    }


def main():
    problems_dir = os.path.join(current_dir, "problems")
    solutions_dir = os.path.join(current_dir, "solutions")
    os.makedirs(solutions_dir, exist_ok=True)

    problem_files = sorted(glob.glob(os.path.join(problems_dir, "*.mps.gz")))
    print("=" * 85)
    print(f"🔬 RUNNING HIGHS GROUND-TRUTH ENGINE ON {len(problem_files)} BENCHMARK PROBLEMS")
    print("   Protocol: Pure algorithmic solve time only (zero file/memory I/O)")
    print("=" * 85)

    catalog = {}
    summary_path = os.path.join(current_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            catalog = {}

    solved_count = 0
    skipped_count = 0
    total_t0 = time.time()

    for idx, filepath in enumerate(problem_files, 1):
        filename = os.path.basename(filepath)
        clean_name = filename.replace(".mps.gz", "")
        out_npz = os.path.join(solutions_dir, f"{clean_name}.npz")

        # Check existing solution in cache
        if os.path.exists(out_npz) and clean_name in catalog and catalog[clean_name].get("optimal_objective") is not None:
            solved_count += 1
            continue

        # Fast dimension scan via Highs C++
        h_dim = highspy.Highs()
        h_dim.setOptionValue("output_flag", False)
        st = h_dim.readModel(filepath)
        if st != highspy.HighsStatus.kOk:
            continue
        n_cols, n_rows = h_dim.getNumCol(), h_dim.getNumRow()
        max_dim = max(n_cols, n_rows)

        if max_dim <= 1000:
            tier = "small"
        elif max_dim <= 5000:
            tier = "medium"
        elif max_dim <= 15000:
            tier = "large"
        else:
            tier = "huge"

        # Protect device RAM: skip huge models (> 15k vars/cons)
        if tier == "huge":
            skipped_count += 1
            catalog[clean_name] = {
                "filename": filename,
                "n_vars": n_cols,
                "n_constraints": n_rows,
                "tier": tier,
                "status": "skipped_due_to_device_ram_limit",
                "solution_file": None
            }
            continue

        # Load and Solve
        try:
            kkt_sys = load_problem(filepath)
            sol = solve_and_record_pure_time(kkt_sys)
            pure_ms = sol["pure_solve_time_sec"] * 1000.0

            if "Optimal" in sol["status"]:
                np.savez_compressed(
                    out_npz,
                    x_star=sol["x_star"],
                    lambda_star=sol["lambda_star"],
                    objective_val=sol["objective_val"],
                    pure_solve_time_sec=sol["pure_solve_time_sec"],
                    kkt_error=sol["kkt_error"],
                    var_names=np.array(kkt_sys.var_names),
                    con_names=np.array(kkt_sys.con_names)
                )

                catalog[clean_name] = {
                    "filename": filename,
                    "n_vars": kkt_sys.n_vars,
                    "n_constraints": kkt_sys.n_constraints,
                    "tier": tier,
                    "status": "optimal",
                    "optimal_objective": sol["objective_val"],
                    "highs_pure_solve_time_ms": round(pure_ms, 3),
                    "kkt_verification_error": sol["kkt_error"],
                    "solution_file": f"solutions/{clean_name}.npz"
                }
                solved_count += 1
                print(
                    f"[{idx:3d}/{len(problem_files):3d}] ✓ {clean_name:<20} ({tier:<6}) | "
                    f"vars={kkt_sys.n_vars:5d}, cons={kkt_sys.n_constraints:6d} | "
                    f"HiGHS={pure_ms:7.2f} ms | obj={sol['objective_val']:14.4f} | "
                    f"KKT err={sol['kkt_error']:.1e}"
                )
            else:
                catalog[clean_name] = {
                    "filename": filename,
                    "n_vars": kkt_sys.n_vars,
                    "n_constraints": kkt_sys.n_constraints,
                    "tier": tier,
                    "status": sol["status"],
                    "solution_file": None
                }
                print(f"[{idx:3d}/{len(problem_files):3d}] ⚠️ {clean_name:<20} ({tier}) status: {sol['status']}")

        except Exception as e:
            catalog[clean_name] = {
                "filename": filename,
                "n_vars": n_cols,
                "n_constraints": n_rows,
                "tier": tier,
                "status": f"error: {str(e)}",
                "solution_file": None
            }
            print(f"[{idx:3d}/{len(problem_files):3d}] ✗ {clean_name:<20} ({tier}) error: {e}")

        # Periodically flush summary to disk every 10 problems
        if idx % 10 == 0:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)

    # Final save
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    total_duration = time.time() - total_t0
    print("\n" + "=" * 85)
    print(f"🎉 BATCH COMPLETE IN {total_duration:.1f}s!")
    print(f"   • Solved & Saved: {solved_count} benchmark solutions in solutions/")
    print(f"   • Cataloged Total: {len(catalog)} problems in summary.json")
    print(f"   • RAM-Protected (Skipped > 15k vars): {skipped_count} instances")
    print("=" * 85)


if __name__ == "__main__":
    main()
