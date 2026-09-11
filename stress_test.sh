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
    compare_saved_benchmarks([
        "iteration_1_25_problems",
        "iteration_2_25_problems",
        "iteration_3_25_problems"
    ])
except Exception as e:
    print(f"[NOTE] Could not compare all 3 runs: {e}")
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
else:
    print(f"[ERROR] Invalid iteration '{iter_num}'. Use 1, 2, 3, or compare.")
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
    fname = f"iteration_{iter_num}_8_problems"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        problem_names=p_list,
        filename=fname
    )
elif mode == "25" or mode == "medium":
    fname = f"iteration_{iter_num}_25_problems"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        problem_names=test_25,
        filename=fname
    )
elif mode == "all" or mode == "101":
    fname = f"iteration_{iter_num}_tier1_all"
    evaluate_solver_on_benchmark(
        solver_fn=run_solver,
        solver_name=solver_name,
        tier="small",
        filename=fname
    )
else:
    print(f"[ERROR] Unknown mode '{mode}'. Use 8, 25, or all.")
    sys.exit(1)
EOF

