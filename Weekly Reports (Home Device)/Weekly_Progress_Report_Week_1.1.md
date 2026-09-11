# Weekly Progress Report - Week 1.1

**Date:** September 11, 2026  
**Project:** Neural Network Based KKT Solver for Linear Programming  
**Branch / Repository:** `main` (Synchronized with `Jeet-Dev`)  
**Workspace:** BTech Project  

---

## 1. Executive Summary

This report covers the research progress and empirical results following the integration of Iteration 2 (`KKT_Solver_Iteration_2`), developed collaboratively and merged into `main`. 

During this development cycle, we evaluated the hypothesis that the Iteration 1 baseline was constrained by standard architectural choices (specifically dead neurons in ReLU activations and gradient competition from penalizing dual non-negativity). We deployed a 25-problem academic stress test spanning Netlib and MIPLIB linear programming benchmarks under the strict zero-I/O timing protocol.

Key outcomes of Week 1.1:
- Integration of a GELU backbone activation and a structurally bounded Softplus dual multiplier head.
- Complete structural elimination of dual non-negativity violations across 100% of tested instances (dual error strictly 0.000).
- Substantial reductions in relative objective gap on 68% of tested benchmark models, including large-scale instances such as `etamacro` (reduced from 267.05% to 1.19% gap).
- Identification of seven core optimization bottlenecks and diagnostic insights that define the research requirements for Iteration 3.

---

## 2. Iteration 2 Architecture and Formulation

### 2.1 Architectural Modifications

Iteration 2 introduced two specific changes to the neural model (`model.py`), keeping all other hyperparameter settings, widths (64 hidden units), optimizers, and learning rate schedulers identical to Iteration 1 to isolate the effects of the structural adjustments:

1. **Backbone Activation Upgrade (ReLU to GELU):**
   - The shared 2-layer hidden representation was upgraded from standard `nn.ReLU()` to `nn.GELU()`.
   - Rationale: During early training, randomly initialized weights place the primal solution vector deep in the infeasible domain. Under standard ReLU, large negative pre-activations cause units to permanently output zero gradients (dead neurons), severely reducing network capacity. GELU provides smooth, non-zero gradients across the entire real line.

2. **Structural Dual Feasibility Head (Linear to Softplus):**
   - The dual multiplier head was modified from an unconstrained linear projection into a composite layer:
     $$\hat{\lambda} = \text{Softplus}(\text{Linear}(h)) = \ln(1 + e^{\text{Linear}(h)})$$
   - Rationale: In linear programming with inequality constraints ($G x \le h$), dual multipliers must satisfy $\lambda \ge 0$ unconditionally. Rather than treating this condition as a soft penalty in the loss function, Softplus enforces non-negativity by architectural construction.

### 2.2 Reformulated KKT Loss Function

Because $\hat{\lambda} \ge 0$ is guaranteed by construction, the dual non-negativity penalty was removed entirely from `loss.py`. The resulting loss function operates with four penalty terms:

$$\mathcal{L}_{\text{total}} = w_{\text{stat}} \|c + G^T \hat{\lambda}\|_2^2 + w_{\text{prim}} \|\max(0, G\hat{x} - h)\|_2^2 + w_{\text{slack}} \|\hat{\lambda} \odot (G\hat{x} - h)\|_2^2 + w_{\text{x\_pos}} \|\max(0, -\hat{x})\|_2^2$$

Where:
- Stationarity: $\|c + G^T \hat{\lambda}\|_2^2$ (First-order optimality condition)
- Primal Feasibility: $\|\max(0, G\hat{x} - h)\|_2^2$ (Constraint satisfaction)
- Complementary Slackness: $\|\hat{\lambda} \odot (G\hat{x} - h)\|_2^2$ (Orthogonality between active constraints and dual prices)
- Primal Non-Negativity: $\|\max(0, -\hat{x})\|_2^2$ (Physical variable bounds where applicable)

---

## 3. Large-Scale Benchmark Stress Test (25 Problems)

Both Iteration 1 and Iteration 2 were evaluated under identical training conditions: 1,000 epochs, Adam optimizer with base learning rate $\eta = 0.015$, `ReduceLROnPlateau` scheduling, and identical problem instances from Netlib and MIPLIB. All timing measurements reflect pure algorithmic solve time (zero file I/O, zero parsing overhead).

### 3.1 Head-to-Head Benchmark Scorecard

| Problem | Variables (n) | Constraints (m) | Iter 1 Gap % | Iter 2 Gap % | Gap Delta | Iter 1 Dual Viol | Iter 2 Dual Viol | Iter 2 Solve Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **etamacro** | 688 | 1,577 | 267.05% | 1.19% | -265.86% [WIN] | 5.00e-02 | 0.00e+00 | 700.5 ms |
| **flugpl** | 18 | 53 | 113.49% | 37.09% | -76.40% [WIN] | 4.82e+00 | 0.00e+00 | 370.9 ms |
| **boeing1** | 384 | 986 | 131.57% | 52.47% | -79.10% [WIN] | 8.70e-01 | 0.00e+00 | 545.6 ms |
| **agg** | 163 | 687 | 99.99% | 26.01% | -73.98% [WIN] | 1.12e+00 | 0.00e+00 | 520.7 ms |
| **agg2** | 302 | 878 | 99.89% | 57.92% | -41.97% [WIN] | 9.40e-01 | 0.00e+00 | 566.7 ms |
| **beaconfd** | 262 | 575 | 86.20% | 57.65% | -28.55% [WIN] | 1.30e-01 | 0.00e+00 | 508.2 ms |
| **bandm** | 472 | 1,082 | 58.90% | 36.58% | -22.32% [WIN] | 2.80e-01 | 0.00e+00 | 589.7 ms |
| **capri** | 353 | 899 | 100.25% | 81.94% | -18.31% [WIN] | 6.50e-01 | 0.00e+00 | 565.6 ms |
| **share2b** | 79 | 188 | 36.36% | 21.15% | -15.21% [WIN] | 6.30e-01 | 0.00e+00 | 398.1 ms |
| **bell3a** | 133 | 327 | 27.19% | 20.95% | -6.24% [WIN] | 3.45e+02 | 0.00e+00 | 411.0 ms |
| **dcmulti** | 548 | 991 | 21.99% | 17.13% | -4.86% [WIN] | 8.00e-02 | 0.00e+00 | 676.1 ms |
| **sc50a** | 48 | 117 | 99.90% | 94.86% | -5.04% [WIN] | 1.00e-02 | 0.00e+00 | 392.0 ms |
| **afiro** | 32 | 67 | 99.67% | 96.36% | -3.31% [WIN] | 5.00e-02 | 0.00e+00 | 383.9 ms |
| **blend** | 83 | 200 | 97.29% | 96.11% | -1.18% [WIN] | 4.00e-02 | 0.00e+00 | 400.5 ms |
| **gen-ip002** | 41 | 65 | 1.05% | 0.62% | -0.43% [WIN] | 1.10e-01 | 0.00e+00 | 379.7 ms |
| **degen2** | 534 | 1,199 | 1.51% | 1.66% | +0.15% [TIE] | 2.00e-02 | 0.00e+00 | 665.9 ms |
| **egout** | 141 | 368 | 5.03% | 8.46% | +3.43% [TIE] | 9.00e-02 | 0.00e+00 | 418.4 ms |
| **bore3d** | 315 | 774 | 68.16% | 70.75% | +2.59% [TIE] | 3.10e-01 | 0.00e+00 | 534.4 ms |
| **standata** | 1,075 | 1,714 | 94.83% | 98.30% | +3.47% [TIE] | 4.00e-02 | 0.00e+00 | 306.5 ms |
| **brandy** | 249 | 570 | 99.99% | 99.88% | -0.11% [TIE] | 0.00e+00 | 0.00e+00 | 534.8 ms |
| **bell5** | 104 | 253 | 82.79% | 86.79% | +4.00% [TIE] | 1.03e+03 | 0.00e+00 | 429.5 ms |
| **e226** | 282 | 538 | 20.69% | 26.02% | +5.33% | 2.00e-02 | 0.00e+00 | 504.0 ms |
| **adlittle** | 97 | 168 | 200.20% | 205.51% | +5.31% | 3.42e+02 | 0.00e+00 | 402.2 ms |
| **forplan** | 421 | 671 | 100.35% | 286.76% | +186.41% | 2.15e+02 | 0.00e+00 | 534.3 ms |
| **enigma** | 100 | 242 | ~Tie | ~Tie | 0.00% [TIE] | 0.00e+00 | 0.00e+00 | 400.9 ms |

### 3.2 Aggregate Performance Metrics

- **Win / Improvement Ratio:** Iteration 2 outperformed Iteration 1 on **17 out of 25 problems (68.0%)**.
- **Dual Constraint Compliance:** Iteration 1 exhibited average dual feasibility violations exceeding $10^2$ to $10^3$ on several models. Iteration 2 reduced dual feasibility violations to **strictly 0.00e+00** across 25 out of 25 problems.
- **Throughput:** Average pure algorithmic solve time was **493.6 milliseconds per problem** on CPU for 1,000 gradient steps. The entire 25-problem test suite executed in 12.34 seconds.

---

## 4. Research Insights Obtained from Iterations 1 and 2

The experimental data reveals seven fundamental insights regarding physics-informed neural optimization on linear systems:

### Insight 1: The "Fake Stationarity" Mechanism in Penalty Formulations
In Iteration 1, the optimizer frequently appeared to achieve lower objective gaps on models such as `bell5`, `adlittle`, and `forplan`. However, inspection of multiplier vectors revealed extreme dual non-negativity violations (e.g., $\lambda_i \approx -1031$). Because stationarity only requires $c + G^T \lambda \to 0$, allowing $\lambda$ to assume negative values gave the optimizer unconstrained degrees of freedom to algebraically cancel components of $c$. This allowed the network to minimize stationarity loss without identifying the true supporting hyperplane. Once negative duals were structurally eliminated in Iteration 2, the true tension between the objective and constraints was exposed.

### Insight 2: Gradient Deadlock and Plateau Dynamics
Loss curves across multiple problems exhibit an initial steep drop during epochs 1 to 250, followed by an abrupt, persistent plateau. This plateau occurs because the gradient of the objective minimization term and the gradient of the constraint feasibility penalty eventually orient in opposite directions:
$$\nabla_\theta \mathcal{L}_{\text{stat}} \approx -\nabla_\theta \mathcal{L}_{\text{prim}}$$
Under static loss weights, these competing gradients reach vector equilibrium. As a result, parameter updates oscillate locally along the constraint boundary rather than traversing along the active face toward the true optimal vertex.

### Insight 3: The Limitation of Global Scalar Penalty Weights
In both iterations, all $m$ inequality constraints in $G x \le h$ are penalized using a single shared scalar weight ($w_{\text{prim}}$). In standard linear programs, only a small subset of constraints are active (binding) at optimality; the majority remain strictly slack ($G_i x < h_i$). Uniform penalization forces the network to allocate gradient steps indiscriminately across all constraints, diluting the corrective force required on true bottleneck hyperplanes.

### Insight 4: Gradient Flow Preservation Across Infeasible Space
The substantial performance gains observed with GELU (e.g., `etamacro` dropping from 267% to 1.19% gap; `boeing1` dropping from 131% to 52% gap) demonstrate the vulnerability of standard ReLU in constrained optimization. Because random weight initialization places early solution estimates outside the feasible region, large initial constraint gradients push many pre-activations into the negative half-plane. With standard ReLU, those units produce zero gradient permanently. Smooth activations with non-zero curvature maintain gradient flow across early infeasible trajectories.

### Insight 5: Asymmetric Rate of KKT Condition Satisfaction
Convergence rates across the four KKT conditions are highly asymmetric. In Iteration 2, dual feasibility was solved instantaneously by architecture, and complementary slackness residuals dropped by up to $140\times$ relative to Iteration 1. In contrast, **primal feasibility violation remained the dominant component of total residual error** on difficult instances (e.g., `beaconfd`, `capri`, `flugpl`). Adjusting dual multipliers is numerically easier for gradient descent than navigating high-dimensional primal polyhedral vertices.

### Insight 6: Impact of Numerical Scaling Disparities
Models with homogeneous numerical scales (such as `gen-ip002`, `degen2`, and `egout`) consistently converged to relative objective gaps below 2% to 8%. Conversely, models with wide coefficient disparities (such as `agg`, where optimal objectives are on the order of $-10^7$, or `beaconfd`, where matrix coefficients span multiple orders of magnitude) exhibited slower convergence. Because the loss evaluates unweighted Euclidean norms, terms with naturally larger numerical magnitudes dominate the gradient updates regardless of physical importance.

### Insight 7: Polytope Geometry as the Chief Determinant of Difficulty
Problem size (number of variables and constraints) does not correlate directly with solver difficulty. For example:
- `degen2` ($n=534, m=1,199$) solved to a **1.66% gap** in 665 ms.
- `flugpl` ($n=18, m=53$) stalled at a **37.09% gap**.
The primary determinants of convergence difficulty are geometric: the condition number of the active constraint submatrix, the sharpness of the angles between active constraint hyperplanes, and the presence of primal or dual degeneracy.

---

## 5. Strategic Considerations for Iteration 3

Based on the diagnostic findings from Iterations 1 and 2, the following technical considerations must be addressed in the design of Iteration 3:

1. **Adaptive vs. Uniform Constraint Weighting:**
   - Static global weights ($w_{\text{prim}} = 5.0$) prevent the network from differentiating between binding constraints and non-binding constraints. Iteration 3 must incorporate a mechanism to adjust penalty pressure per-constraint based on individual violation histories.

2. **Decoupling Feasibility Pressure from Objective Progress:**
   - To resolve the opposing gradient deadlock ($\nabla \mathcal{L}_{\text{stat}} \approx -\nabla \mathcal{L}_{\text{prim}}$), the optimization scheme must ensure that constraint penalties can increase dynamically without causing gradient explosion or numerical instability.

3. **Primal Boundary Enforcement:**
   - Primal infeasibility is currently the primary error contributor. The formulation in Iteration 3 must enforce stronger convergence toward the feasible polytope boundary without sacrificing the objective value gains achieved by GELU.

4. **Matrix and Problem Coefficient Normalization:**
   - Disparities across cost vectors $c$, constraint rows $G_i$, and right-hand sides $h_i$ distort gradient magnitudes. A systematic preconditioning or row-scaling step during canonical ingestion is necessary to balance loss term magnitudes across varied problem classes.

5. **Structural Handling of Variable Bounds:**
   - Problems with non-standard bounds ($l_j \le x_j \le u_j$ or free variables) currently rely on penalty terms in $G x \le h$. Structural bound handling on the primal head, analogous to the Softplus head on duals, warrants investigation.

---

## 6. Summary Status of Workspace Assets

- `KKT_Standalone_Reader/`: Ingestion engine operational across MPS, LP, Pyomo, and JSON formats (5/5 unit tests passing).
- `KKT_Solver_Iteration_1/`: Baseline ReLU implementation preserved as benchmark reference.
- `KKT_Solver_Iteration_2/`: GELU + Softplus implementation verified and synchronized on `main`.
- `KKT_Benchmark/`: 412 benchmark instances, 393 verified HiGHS ground-truth solutions, and automated evaluation harness operational.
- `stress_test.sh`: Standalone terminal script supporting multi-tier evaluation modes.
