# Neural Network Based KKT Solver

**B.Tech. Project (Industrial Optimization)**  
**Department of Mechanical Engineering**  

### Project Team:
* **Students:** Sahitya Rankawat (2023ME21131), Manav Gupta (2023ME20733), Jeet Anand (2023ME20874)  
* **Supervisor:** Prof. Kartikey Sharma  

---

## Overview
This project develops an optimization solver to predict optimal primal and dual variables of any LP problem, based on solving the **Karush-Kuhn-Tucker (KKT) optimality conditions** using **Neural Networks (KINN)**. 

Instead of relying on slow, sequential iterative algorithms (like classical Simplex or Interior-Point methods), this neural solver directly learns to satisfy the KKT optimality conditions in an unsupervised, physics-informed framework, outputting optimal primal variables ($\hat{x}$) and dual multipliers ($\hat{\lambda}$) simultaneously.

---

## Repository Structure

```
├── Weekly Reports (Home Device)/  # Weekly progress reports, uploaded documents & milestone tracker
│   ├── IIT Delhi Reports/         # Official college progress reports submitted at IIT Delhi
│   │   ├── I1 Progress report_1.pdf
│   │   └── I1 Progress Report_2.pdf
│   ├── Weekly_Progress_Report_Week_1.md # Comprehensive Week 1 collaborative milestone report
│   └── README.md                  # Weekly progress index and roadmap
│
├── KKT_Standalone_Reader/         # Universal problem ingestion engine (Zero-dependency)
│   ├── core/                      # Canonical LP representation & KKT verification
│   ├── readers/                   # MPS, LP, Pyomo, JSON & NumPy/SciPy matrix readers
│   ├── examples/                  # Sample problem files (Diet, Production, Supply Chain)
│   └── tests/test_reader.py       # Automated test suite (5/5 passing, < 1e-14 error)
│
├── KKT_Solver_Iteration_1/        # Baseline ReLU Solver (5-term KKT loss)
├── KKT_Solver_Iteration_2/        # GELU + Softplus Dual Head Solver (4-term KKT loss)
├── KKT_Solver_Iteration_3/        # Fischer-Burmeister Complementarity Formulation
│
├── KKT_Benchmark/                 # Official Academic Benchmark Suite
│   ├── problems/                  # 412 official Netlib & MIPLIB .mps.gz benchmark files
│   ├── solutions/                 # 393 verified HiGHS ground-truth solutions (.npz)
│   ├── results/                   # Archived JSON & CSV scorecards across all solver iterations
│   ├── summary.json               # Catalog of all 412 instances with dimensions and metrics
│   ├── benchmark_harness.py       # Automated evaluation harness with serialization & comparison
│   └── generate_solutions.py      # High-throughput C++ HiGHS sparse streaming solver
│
├── stress_test.sh                 # Standalone stress-test runner & cross-iteration comparator
├── ITERATION_DEVELOPMENT_GUIDE.md # Developer guide for teammates
├── requirements.txt              # Project dependencies
└── .gitignore                    # Standard Python, environment, and cache ignore rules
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Problem Ingestion Engine
Run the automated reader test suite to verify canonical extraction across all 4 formats:
```bash
python KKT_Standalone_Reader/tests/test_reader.py
```

### 3. Run Single-Instance KINN Baseline (Iteration 1)
Solve a canonical production planning LP and plot the 5-term loss convergence:
```bash
python KKT_Solver_Iteration_1/run_iteration_1.py
```

### 4. Benchmark Against Official Netlib / MIPLIB Instances
Evaluate any solver on curated academic benchmarks under the strict zero-I/O timing protocol:
```bash
python KKT_Benchmark/benchmark_harness.py
```

### 5. Automated Stress Testing & Multi-Iteration Comparison
Execute stress tests on specific iterations or compare all archived JSON runs side-by-side:
```bash
# Run 25-problem benchmark on Iteration 3
./stress_test.sh 3 25

# Compare all archived iterations instantly (zero re-solve overhead)
./stress_test.sh compare
```

---

## Benchmarking & Evaluation Methodology

1. **Strict Algorithmic Timing Protocol:** Timing starts strictly after the problem matrices are loaded into RAM and ends the instant solution vectors $(\hat{x}, \hat{\lambda})$ are returned. Zero disk I/O or parsing overhead is included in the timers.
2. **True Optimality Metrics:** Because linear programs frequently have non-unique optimal solutions (entire optimal faces), measuring Euclidean distance $\|x - x^*\|$ to a single Simplex corner is mathematically flawed. Solvers are evaluated using:

   - **Relative Objective Gap:** $\frac{\|c^T \hat{x} - \hat{z}\|}{\max(1, \hat{z}\)}$
   - **Primal Feasibility Violation:** $\|\max(0, G\hat{x} - h)\|_\infty$
   - **Dual Feasibility Violation:** $\|\min(0, \hat{\lambda})\|_\infty$
   - **Stationarity Residual:** $\|c + G^T \hat{\lambda}\|_\infty$
   - **Complementary Slackness:** $\|\hat{\lambda} \odot (h - G\hat{x})\|_\infty$
