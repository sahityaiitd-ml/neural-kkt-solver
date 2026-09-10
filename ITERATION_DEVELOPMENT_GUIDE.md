# Developer Guide: Building, Testing & Benchmarking New Solver Iterations

> **B.Tech Project: Neural Network Based KKT Solver**  
> **Target Audience:** Project Teammates (Sahitya, Manav, Jeet)  
> **Purpose:** Standardized workflow for developing new neural KKT solver iterations without breaking existing pipelines or Git history.

---

## 📑 Table of Contents
1. [Mental Model & Architecture Overview](#1-mental-model--architecture-overview)
2. [Step-by-Step: Creating a New Iteration](#2-step-by-step-creating-a-new-iteration)
3. [Where to Make Changes (File Responsibilities)](#3-where-to-make-changes-file-responsibilities)
4. [Testing Protocol (3-Stage Verification)](#4-testing-protocol-3-stage-verification)
5. [Understanding Benchmark Metrics](#5-understanding-benchmark-metrics)
6. [Git Collaboration & Submission Checklist](#6-git-collaboration--submission-checklist)

---

## 1. Mental Model & Architecture Overview

Every solver iteration in this repository is **completely decoupled and self-contained**. It takes a standardized linear program from `KKT_Standalone_Reader` and outputs optimal primal $(\hat{x})$ and dual $(\hat{\lambda})$ vectors:

```
                  ┌──────────────────────────────┐
                  │    KKT_Standalone_Reader     │
                  │   Loads MPS / LP / JSON      │
                  └──────────────┬───────────────┘
                                 │ Returns Canonical System:
                                 │ min c^T x  s.t.  G x <= h
                                 ▼
                  ┌──────────────────────────────┐
                  │   KKT_Solver_Iteration_X     │
                  │                              │
                  │  ├── model.py    (Network)   │
                  │  ├── loss.py     (KKT terms) │
                  │  ├── solver.py   (Optimizer) │
                  │  └── evaluate.py (Residuals) │
                  └──────────────┬───────────────┘
                                 │ Returns:
                                 │ x̂ (primal), λ̂ (dual)
                                 ▼
                  ┌──────────────────────────────┐
                  │        KKT_Benchmark         │
                  │   Pure Algorithmic Timing    │
                  │   Relative Gap vs HiGHS      │
                  │   Prints Scorecard Table     │
                  └──────────────────────────────┘
```

---

## 2. Step-by-Step: Creating a New Iteration

Suppose you are building **Iteration 2** (`KKT_Solver_Iteration_2`). Follow these exact steps:

### Step 2.1: Clone the Previous Iteration Folder
In your terminal, copy the existing iteration:
```bash
cp -r KKT_Solver_Iteration_1 KKT_Solver_Iteration_2
```

### Step 2.2: Rename the Standalone Runner Script
Rename the runner file inside the new folder:
```bash
mv KKT_Solver_Iteration_2/run_iteration_1.py KKT_Solver_Iteration_2/run_iteration_2.py
```

### Step 2.3: Update Imports Inside the Runner Script
Open `KKT_Solver_Iteration_2/run_iteration_2.py` and update the import statement at the top:
```python
# CHANGE THIS:
from KKT_Solver_Iteration_1 import solve_kkt_instance

# TO THIS:
from KKT_Solver_Iteration_2 import solve_kkt_instance
```

*(Note: Files like `model.py`, `loss.py`, and `solver.py` use relative imports like `from .model import KINNSolverModel`, so they do **not** need any import changes!)*

### Step 2.4: Verify the Clone Runs Without Edits
Before touching any neural code, make sure the baseline runs inside the new directory:
```bash
python KKT_Solver_Iteration_2/run_iteration_2.py
```
If it runs and prints the loss epochs, your new iteration workspace is set up correctly.

---

## 3. Where to Make Changes (File Responsibilities)

Each file has a single responsibility. Only edit the file that corresponds to your research hypothesis:

```
KKT_Solver_Iteration_X/
├── model.py     ──► Neural Network Architecture (Layers, Activations, Heads)
├── loss.py      ──► Mathematical Objective (KKT Condition Penalties & ALM)
├── solver.py    ──► Optimization Routine (Adam, Schedulers, Early Stopping)
├── evaluate.py  ──► Mathematical Verification (Residuals & Feasibility)
└── README.md    ──► Your Hypothesis & Findings Documentation
```

### 1. Modifying Architecture (`model.py`)
Edit this file when testing **new network designs, activation functions, or output layers**:
- **Hidden Layers:** Change layer width (e.g., 128 to 256), depth (2 to 4 layers), or add residual skip connections.
- **Activation Functions:** Replace `nn.ReLU()` with smooth activations like `nn.GELU()` or `nn.SiLU()` to prevent dead neurons:
  ```python
  self.activation = nn.GELU()
  ```
- **Structural Dual Non-Negativity:** The dual multiplier for inequality constraints must be strictly $\hat{\lambda} \ge 0$. Instead of penalizing negative values in the loss, structurally enforce it by wrapping the dual head in `nn.Softplus()`:
  ```python
  # Primal Head (Decision variables x can be positive or negative)
  self.head_x = nn.Linear(hidden_dim, n_vars)

  # Dual Head (Multipliers lambda must be strictly non-negative)
  self.head_lambda = nn.Sequential(
      nn.Linear(hidden_dim, n_cons),
      nn.Softplus()  # Guarantees lambda_hat >= 0 unconditionally
  )
  ```

### 2. Modifying the Loss Function (`loss.py`)
Edit this file when changing **how KKT violations are penalized**:
- The baseline 5-term KKT loss is:
  $$\mathcal{L} = w_{\text{stat}} \mathcal{L}_{\text{stat}} + w_{\text{prim}} \mathcal{L}_{\text{prim}} + w_{\text{slack}} \mathcal{L}_{\text{slack}} + w_{\text{x\_nonneg}} \mathcal{L}_{\text{x\_nonneg}} + w_{\text{dual\_nonneg}} \mathcal{L}_{\text{dual\_nonneg}}$$
- **Augmented Lagrangian Multiplier (ALM) Updates:** If feasibility and stationarity stall each other (loss plateau), implement dynamic penalty scaling $\mu_{k+1} = \min(\rho \mu_k, \mu_{\max})$ to force constraints inside the feasible boundary.
- If you used `nn.Softplus()` in `model.py`, set $w_{\text{dual\_nonneg}} = 0$ since negative duals are mathematically impossible.

### 3. Modifying Training Dynamics (`solver.py`)
Edit this file to change **how the network optimizes**:
- **Learning Rate Scheduling:** Adjust `ReduceLROnPlateau(factor=0.5, patience=100)`.
- **Optimizer:** Swap `torch.optim.Adam` with `torch.optim.AdamW` (for weight decay) or `torch.optim.LBFGS` (for second-order curvature).
- **Snapshotting:** Always keep the best-checkpoint mechanism enabled. Neural networks on LPs bounce near constraint boundaries late in training; restoring the historical minimum loss checkpoint is crucial for accuracy.

---

## 4. Testing Protocol (3-Stage Verification)

Never benchmark on hundreds of problems before completing Stage 1 and Stage 2!

### Stage 1: Fast Single-Instance Sanity Check (~2 seconds)
Run the local script on a standard test problem:
```bash
python KKT_Solver_Iteration_2/run_iteration_2.py
```
**Pass Criteria:**
- Training completes without `NaN` or `Inf` losses.
- Generated plot (`convergence_production_plan.png`) shows a steadily decreasing curve.

### Stage 2: The 8-Problem Netlib Scorecard (~5 seconds)
Benchmark your iteration against the official 8 Netlib problems:
Create a short test script (e.g. `test_iter2.py`) or run via python terminal:

```python
from KKT_Benchmark import evaluate_solver_on_benchmark
from KKT_Solver_Iteration_2 import solve_kkt_instance

# 1. Define wrapper for your solver
def run_my_iteration(kkt_sys):
    return solve_kkt_instance(
        kkt_sys,
        max_epochs=1200,
        lr=0.015,
        verbose=False
    )

# 2. Benchmark on the official scorecard problems
scorecard = evaluate_solver_on_benchmark(
    solver_fn=run_my_iteration,
    solver_name="Iteration 2 (Softplus + GELU)",
    problem_names=["afiro", "flugpl", "adlittle", "bell5", "e226", "blend", "share2b", "sc50a"]
)
```
**Pass Criteria:**
- Inspect the terminal output scorecard table.
- Compare `Gap %` and `Prim Viol` directly against Iteration 1's scorecard.

### Stage 3: Full Tier 1 Academic Stress Test (~1 minute)
Once Stage 2 shows improvement, evaluate across all small instances in the suite:
```python
evaluate_solver_on_benchmark(
    solver_fn=run_my_iteration,
    solver_name="Iteration 2 (Softplus + GELU)",
    tier="small",        # Tests on official Tier 1 problems (<= 1,000 dims)
    max_problems=30      # Caps test to first 30 problems
)
```

---

## 5. Understanding Benchmark Metrics

When inspecting the terminal scorecard table, here is how to interpret each column:

| Column | What It Measures | Target Value | What It Tells You |
| :--- | :--- | :--- | :--- |
| **HiGHS Obj** | Classical optimal objective $z^*$ | Reference | Ground-truth benchmark computed via C++ HiGHS. |
| **KINN Obj** | Neural objective $c^T \hat{x}$ | $\approx z^*$ | The objective value achieved by your neural network. |
| **Gap %** | $\frac{\|c^T \hat{x} - z^*\|}{\max(1, \|z^*\|)} \times 100\%$ | $< 5.0\%$ | **Primary Accuracy Metric.** Shows proximity to global optimality. |
| **Prim Viol** | $\|\max(0, G\hat{x} - h)\|_\infty$ | $< 10^{-4}$ | **Feasibility Metric.** If large, the network cheated the objective by violating constraints. |
| **HiGHS ms** | HiGHS pure solve time (ms) | $< 5\text{ ms}$ | Reference classical baseline (on CPU). |
| **KINN ms** | Neural pure solve time (ms) | $< 500\text{ ms}$ | **Strict Zero-I/O Timer.** Measures pure forward + backward optimization only. |

> [!IMPORTANT]
> **Why we DO NOT measure $\|x - x^*\|$ (Euclidean Distance):**  
> Linear programs frequently have **non-unique optimal solutions** (entire optimal faces or edges). Simplex stops at an extreme corner vertex; neural networks converge to the interior/analytic center of that same optimal face. Therefore, `Gap %` and `Prim Viol` are the true gold-standard metrics.

---

## 6. Git Collaboration & Submission Checklist

To keep the main branch clean and avoid Git conflicts among teammates, follow this checklist:

### 1. Work on a Feature Branch
```bash
git checkout -b feature/iteration-2-softplus
```

### 2. Check What Files Are Being Staged
Before committing, always check `git status`:
```bash
git status
```
- Your new folder `KKT_Solver_Iteration_2/` should be untracked.
- **Never force-add large dataset files** (e.g. `.mps.gz` or `.npz` inside `KKT_Benchmark/problems` or `solutions`). `.gitignore` protects against this automatically.

### 3. Write Clear, Descriptive Commits
```bash
git add KKT_Solver_Iteration_2/
git commit -m "feat: Add Iteration 2 with Softplus dual head and GELU backbone"
```

### 4. Push Your Branch & Open Review
```bash
git push origin feature/iteration-2-softplus
```

### 5. Document Your Results
Write a brief 1-paragraph summary of your scorecard in [`Weekly Reports (Home Device)/README.md`](file:///Users/ranky/Desktop/BTech%20Project/Weekly%20Reports%20(Home%20Device)/README.md) so everyone can see what architectural change worked and what failed.
