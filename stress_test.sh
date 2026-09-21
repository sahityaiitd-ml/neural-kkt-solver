#!/usr/bin/env bash
# ==============================================================================
# stress_test.sh: Benchmark & stress-test runner for KKT Solvers (Iteration 1, 2, 3)
# ==============================================================================

# 1. Resolve Project Root Directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="${PROJECT_DIR}/.venv/bin/python"

# Check virtual environment
if [ ! -f "${PYTHON_EXEC}" ]; then
    echo "[ERROR] Virtual environment not found at ${PYTHON_EXEC}"
    echo "Please ensure .venv is installed in ${PROJECT_DIR}"
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
cd "${PROJECT_DIR}" || exit 1

# Arguments: ./stress_test.sh [iteration: 1|2|3|compare] [mode: 8|25|all]
ITERATION="${1:-3}"
MODE="${2:-25}"

echo "=============================================================================="
echo "[STRESS TEST RUNNER] KKT Neural Solver Suite"
echo "  Project Path : ${PROJECT_DIR}"
echo "  Python Exec  : ${PYTHON_EXEC}"
echo "  Target Iter  : ${ITERATION}"
echo "  Problem Set  : ${MODE}"
echo "=============================================================================="

if [ "${ITERATION}" == "compare" ]; then
    "${PYTHON_EXEC}" - << 'EOF'
from KKT_Benchmark.benchmark_harness import compare_saved_benchmarks

try:
    import os
    runs = ["iteration_1_25_problems", "iteration_2_25_problems", "iteration_3_25_problems"]
    for r in ["iteration_4_25_problems", "iteration_4_1_25_problems", "iteration_4_2_25_problems", "iteration_4_3_1_25_problems", "iteration_tragic_25_problems", "the_tragic_solver_25_problems"]:
        if os.path.exists(f"KKT_Benchmark/results/{r}.json"):
            runs.append(r)
    compare_saved_benchmarks(runs)
except Exception as e:
    print(f"[NOTE] Could not complete comparison: {e}")
    print("Ensure benchmark JSON files exist in KKT_Benchmark/results/")
EOF
    exit 0
fi

# Run benchmark for selected iteration
"${PYTHON_EXEC}" - << EOF
import sys
from KKT_Benchmark import evaluate_solver_on_benchmark

iter_num = "${ITERATION}"
mode = "${MODE}"
iter_slug = iter_num.replace(".", "_")

if iter_num == "1":
    from KKT_Solver_Iteration_1 import solve_kkt_instance
    solver_name = "Iteration_1_Basic_ReLU"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, model_type="neural", max_epochs=1000, lr=0.015, verbose=False)
elif iter_num == "2":
    from KKT_Solver_Iteration_2 import solve_kkt_instance
    solver_name = "Iteration_2_GELU_Softplus"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, verbose=False)
elif iter_num == "3":
    from KKT_Solver_Iteration_3 import solve_kkt_instance
    solver_name = "Iteration_3_Fischer_Burmeister"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, verbose=False)
elif iter_num == "4":
    from KKT_Solver_Iteration_4 import solve_kkt_instance
    solver_name = "Iteration_4_Preconditioned_Strong_Duality"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, verbose=False)
elif iter_num in ["4.1", "41", "4_1"]:
    from KKT_Solver_Iteration_4_1 import solve_kkt_instance
    solver_name = "Iteration_4_1_Decoupled_LinearPull"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, verbose=False)
elif iter_num in ["4.2", "42", "4_2"]:
    from KKT_Solver_Iteration_4_2 import solve_kkt_instance
    solver_name = "Iteration_4_2_AnnealedPull_OneSidedGap"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, verbose=False)
elif iter_num in ["4.3.1", "431", "4_3_1", "4.3.1_prac"]:
    from KKT_Solver_Iteration_4_3_1_Prac import solve_kkt_instance
    solver_name = "Iteration_4_3_1_Prac_Adaptive"
    def run_solver(kkt):
        return solve_kkt_instance(kkt, max_epochs=1000, lr=0.015, precond_method="auto", anneal_mode="diameter_adaptive", verbose=False)
elif iter_num.lower() in ["tragic", "the_tragic_solver", "final", "5"]:
    import importlib
    tragic_mod = importlib.import_module("The Tragic Solver")
    solver_name = "The_Tragic_Solver"
    def run_solver(kkt):
        return tragic_mod.solve_kkt_instance(kkt, verbose=False)
else:
    print(f"[ERROR] Invalid iteration '{iter_num}'. Use 1, 2, 3, 4, 4.1, 4.2, 4.3.1, tragic, or compare.")
    sys.exit(1)

test_25 = [
    "afiro", "adlittle", "agg", "agg2", "bandm",
    "beaconfd", "bell3a", "bell5", "blend", "boeing1",
    "bore3d", "brandy", "capri", "dcmulti", "degen2",
    "e226", "egout", "enigma", "etamacro", "flugpl",
    "forplan", "sc50a", "share2b", "standata", "gen-ip002"
]

if mode == "8" or mode == "quick":
    p_list = ["afiro", "flugpl", "adlittle", "bell5", "e226", "blend", "share2b", "sc50a"]
    fname = f"iteration_{iter_slug}_8_problems"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        problem_names=p_list,
        filename=fname
    )
elif mode == "25" or mode == "medium":
    fname = f"iteration_{iter_slug}_25_problems"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        problem_names=test_25,
        filename=fname
    )
elif mode == "100":
    import json, glob, os
    summary_path = "KKT_Benchmark/summary.json"
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    sols = glob.glob("KKT_Benchmark/solutions/*.npz")
    avail = []
    for sol in sols:
        name = os.path.basename(sol).replace(".npz", "")
        if os.path.exists(f"KKT_Benchmark/problems/{name}.mps.gz"):
            meta = summary.get(name, {})
            dim = meta.get("n_vars", 0) + meta.get("n_constraints", 0)
            avail.append((name, dim))
    avail.sort(key=lambda x: (x[1], x[0]))
    top_100 = [x[0] for x in avail[:100]]
    fname = f"iteration_{iter_slug}_100_problems"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        problem_names=top_100,
        filename=fname
    )
elif mode == "all" or mode == "101":
    fname = f"iteration_{iter_slug}_tier1_all"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        tier="small",
        filename=fname
    )
else:
    print(f"[ERROR] Unknown mode '{mode}'. Use 8, 25, 100, or all.")
    sys.exit(1)
EOF

