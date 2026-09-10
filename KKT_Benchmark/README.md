# KKT Benchmark Suite

The `KKT_Benchmark` directory provides an automated evaluation suite to test and compare Neural KKT Solvers (Iteration 1, Iteration 2, etc.) against official, real-world **Netlib** and **MIPLIB** benchmark linear programs.

---

## Directory Structure

```
KKT_Benchmark/
├── problems/                 # 412 official .mps.gz benchmark files (Netlib, MIPLIB 2017, MIPLIB 3.0)
├── solutions/                # 393 ground-truth solutions (.npz) precomputed via HiGHS C++
│   └── *.npz                 # Contains x_star, lambda_star, objective_val, pure_solve_time_sec, kkt_error
├── summary.json              # Complete catalog of all 412 instances with dimensions, tiers, and stats
├── generate_solutions.py     # Script to generate ground truth solutions using HiGHS
├── benchmark_harness.py      # Universal evaluation harness for testing any solver iteration
└── README.md                 # Complete documentation
```

---

## Dataset Size & Tiering (Device Constraints)

To allow benchmarking on standard workstations and laptops without running out of memory (OOM), problems are categorized by scale:

| Tier | Scale | Problem Count | Description & Hardware Suitability |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Small)** | $\le 1,000$ vars/cons | **102 problems** | Ultra-fast execution (< 1 ms in HiGHS); ideal for rapid iteration & debugging. |
| **Tier 2 (Medium)** | $1,000 - 5,000$ vars/cons | **104 problems** | Standard academic evaluation suite; fits comfortably in GPU/RAM. |
| **Tier 3 (Large)** | $5,000 - 15,000$ vars/cons | **71 problems** | Scalability stress test. |
| **Tier 4 (Huge)** | $> 15,000$ vars/cons | **135 problems** | Massive industrial instances; 118 solved, 17 skipped to protect RAM/timeout. |
| **TOTAL** | | **412 problems** | **393 solutions generated & verified!** |

---

## Pure Algorithmic Solve-Time Protocol

To ensure 100% fair scientific comparison:
1. **Zero I/O in Timer:** Problem loading from disk, text/binary MPS parsing, and memory matrix allocation occur **before** the timer starts.
2. **Pure Algorithmic Solve Time:** The timer starts strictly when the solver begins optimizing and ends the instant solution vectors $\hat{x}$ and $\hat{\lambda}$ are returned:
   ```python
   # Load problem and ground-truth into RAM (NOT TIMED)
   kkt_sys = load_problem(filepath)
   ground_truth = np.load(sol_path)

   # PURE SOLVE TIME TIMER
   t_start = time.perf_counter()
   result = solver_fn(kkt_sys)
   pure_solve_time = time.perf_counter() - t_start
   ```
3. Both HiGHS and the Neural KKT Solvers are evaluated under identical conditions.

---

## How to Run the Benchmark Harness

To evaluate an iteration on a curated sample:
```bash
./.venv/bin/python KKT_Benchmark/benchmark_harness.py
```

### To evaluate a new solver iteration (e.g., Iteration 2):
```python
from KKT_Benchmark import evaluate_solver_on_benchmark
from KKT_Solver_Iteration_2 import solve_kkt_instance

def run_iteration_2(kkt_sys):
    return solve_kkt_instance(kkt_sys, max_epochs=1500, lr=0.01)

# Evaluate on Tier 1 (Small) problems
evaluate_solver_on_benchmark(
    solver_fn=run_iteration_2,
    solver_name="KKT_Solver_Iteration_2",
    tier="small",
    max_problems=20
)
```
