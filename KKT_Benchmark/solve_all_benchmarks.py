"""
solve_all_benchmarks.py
=======================
Solves all remaining benchmark problems directly using the HiGHS C++ sparse engine.
Zero memory overflow risk because matrices are maintained in sparse CSR format.
Strictly measures pure algorithmic solve time (zero disk I/O, zero text parsing).
"""

import os
import sys
import glob
import time
import json
import numpy as np
import scipy.sparse as sp
import highspy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

problems_dir = os.path.join(current_dir, "problems")
solutions_dir = os.path.join(current_dir, "solutions")
summary_path = os.path.join(current_dir, "summary.json")
os.makedirs(solutions_dir, exist_ok=True)


def solve_sparse_canonical(filepath, timeout_sec=60.0):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", timeout_sec)
    st = h.readModel(filepath)
    if st != highspy.HighsStatus.kOk:
        return False, "read_error", None, 0, 0, 0, 0, None, None, [], []

    num_cols = h.getNumCol()
    cols = np.arange(num_cols, dtype=np.int32)
    types = np.zeros(num_cols, dtype=np.uint8)
    h.changeColsIntegrality(num_cols, cols, types)

    # Pure solve timer (strictly around solver execution)
    t0 = time.perf_counter()
    h.run()
    pure_time = time.perf_counter() - t0

    st_solve = str(h.getModelStatus())
    if "Optimal" not in st_solve:
        return False, st_solve, None, pure_time, 0, num_cols, h.getNumRow(), None, None, [], []

    sol = h.getSolution()
    info = h.getInfo()
    lp = h.getLp()

    x = np.array(sol.col_value, dtype=np.float64)
    r_dual = np.array(sol.row_dual, dtype=np.float64)
    c_dual = np.array(sol.col_dual, dtype=np.float64)
    c = np.array(lp.col_cost_, dtype=np.float64)

    var_names = list(lp.col_names_) if lp.col_names_ else [f"x_{j}" for j in range(num_cols)]
    raw_row_names = list(lp.row_names_) if lp.row_names_ else [f"row_{i}" for i in range(lp.num_row_)]

    row_lower = np.array(lp.row_lower_, dtype=np.float64)
    row_upper = np.array(lp.row_upper_, dtype=np.float64)
    col_lower = np.array(lp.col_lower_, dtype=np.float64)
    col_upper = np.array(lp.col_upper_, dtype=np.float64)

    a_mat = lp.a_matrix_
    A = sp.csc_matrix((a_mat.value_, a_mat.index_, a_mat.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()

    G_blocks, h_blocks, lam_blocks, con_names = [], [], [], []

    # 1. Row upper bounds
    mask_ru = row_upper < 1e20
    if np.any(mask_ru):
        idx_ru = np.where(mask_ru)[0]
        G_blocks.append(A[idx_ru])
        h_blocks.append(row_upper[idx_ru])
        lam_blocks.append(np.maximum(0.0, -r_dual[idx_ru]))
        con_names.extend([f"{raw_row_names[i]}_upper" for i in idx_ru])

    # 2. Row lower bounds
    mask_rl = row_lower > -1e20
    if np.any(mask_rl):
        idx_rl = np.where(mask_rl)[0]
        G_blocks.append(-A[idx_rl])
        h_blocks.append(-row_lower[idx_rl])
        lam_blocks.append(np.maximum(0.0, r_dual[idx_rl]))
        con_names.extend([f"{raw_row_names[i]}_lower" for i in idx_rl])

    # 3. Col lower bounds
    mask_cl = col_lower > -1e20
    if np.any(mask_cl):
        idx_cl = np.where(mask_cl)[0]
        I_cl = sp.eye(num_cols, format="csr")[idx_cl]
        G_blocks.append(-I_cl)
        h_blocks.append(-col_lower[idx_cl])
        lam_blocks.append(np.maximum(0.0, c_dual[idx_cl]))
        con_names.extend([f"bound_{var_names[j]}_lower" for j in idx_cl])

    # 4. Col upper bounds
    mask_cu = col_upper < 1e20
    if np.any(mask_cu):
        idx_cu = np.where(mask_cu)[0]
        I_cu = sp.eye(num_cols, format="csr")[idx_cu]
        G_blocks.append(I_cu)
        h_blocks.append(col_upper[idx_cu])
        lam_blocks.append(np.maximum(0.0, -c_dual[idx_cu]))
        con_names.extend([f"bound_{var_names[j]}_upper" for j in idx_cu])

    G_csr = sp.vstack(G_blocks)
    h_vec = np.concatenate(h_blocks)
    lam = np.concatenate(lam_blocks)

    # 4 KKT conditions check
    res_stat = float(np.linalg.norm(c + G_csr.T.dot(lam)))
    slack = G_csr.dot(x) - h_vec
    res_prim = float(np.linalg.norm(np.maximum(0.0, slack)))
    res_dual = float(np.linalg.norm(np.maximum(0.0, -lam)))
    res_slack = float(np.linalg.norm(lam * slack))
    tot_err = res_stat + res_prim + res_dual + res_slack

    return True, st_solve, float(info.objective_function_value), pure_time, tot_err, num_cols, G_csr.shape[0], x, lam, var_names, con_names


def main():
    problem_files = sorted(glob.glob(os.path.join(problems_dir, "*.mps.gz")))
    solution_files = set(glob.glob(os.path.join(solutions_dir, "*.npz")))

    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    missing_files = []
    for f in problem_files:
        clean = os.path.basename(f).replace(".mps.gz", "")
        sol_path = os.path.join(solutions_dir, f"{clean}.npz")
        if sol_path not in solution_files or clean not in summary or summary[clean].get("optimal_objective") is None:
            missing_files.append(f)

    print("=" * 85)
    print(f"🚀 SOLVING REMAINING {len(missing_files)} PROBLEMS USING HIGHS SPARSE ENGINE")
    print(f"   Total Suite: {len(problem_files)} | Already Solved: {len(problem_files) - len(missing_files)}")
    print("=" * 85)

    solved_new = 0
    t_start_all = time.time()

    for idx, filepath in enumerate(missing_files, 1):
        filename = os.path.basename(filepath)
        clean_name = filename.replace(".mps.gz", "")
        out_npz = os.path.join(solutions_dir, f"{clean_name}.npz")

        ok, st, obj, pt, err, nv, nc, x, lam, vnames, cnames = solve_sparse_canonical(filepath, timeout_sec=60.0)

        max_dim = max(nv, nc)
        tier = "small" if max_dim <= 1000 else "medium" if max_dim <= 5000 else "large" if max_dim <= 15000 else "huge"

        if ok and x is not None and lam is not None:
            np.savez_compressed(
                out_npz,
                x_star=x,
                lambda_star=lam,
                objective_val=obj,
                pure_solve_time_sec=pt,
                kkt_error=err,
                var_names=np.array(vnames),
                con_names=np.array(cnames)
            )

            summary[clean_name] = {
                "filename": filename,
                "n_vars": nv,
                "n_constraints": nc,
                "tier": tier,
                "status": "optimal",
                "optimal_objective": obj,
                "highs_pure_solve_time_ms": round(pt * 1000.0, 3),
                "kkt_verification_error": err,
                "solution_file": f"solutions/{clean_name}.npz"
            }
            solved_new += 1
            print(
                f"[{idx:3d}/{len(missing_files):3d}] ✓ {clean_name:<22} ({tier:<6}) | "
                f"vars={nv:6d}, cons={nc:6d} | HiGHS={pt*1000:7.1f} ms | "
                f"obj={obj:14.4f} | KKT err={err:.1e}"
            )
        else:
            summary[clean_name] = {
                "filename": filename,
                "n_vars": nv,
                "n_constraints": nc,
                "tier": tier,
                "status": st,
                "solution_file": None
            }
            print(f"[{idx:3d}/{len(missing_files):3d}] ⚠️ {clean_name:<22} ({tier:<6}) | status: {st}")

        # Periodically save summary to disk
        if idx % 5 == 0 or idx == len(missing_files):
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    total_time = time.time() - t_start_all
    print("\n" + "=" * 85)
    print(f"🎉 SOLVED {solved_new} NEW INSTANCES IN {total_time:.1f}s!")
    total_solved = len(glob.glob(os.path.join(solutions_dir, "*.npz")))
    print(f"📁 Total Solutions in solutions/: {total_solved} / {len(problem_files)}")
    print("=" * 85)


if __name__ == "__main__":
    main()
