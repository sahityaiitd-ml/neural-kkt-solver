# KKT_Solver_Iteration_4_2: Annealed Objective Pull & One-Sided Strong Duality Gap

Iteration 4.2 resolves the two persisting failure modes identified in Iteration 4.1:
1. The **-0.10 Equilibrium Trap** on origin-stagnant problems (e.g. `sc50a`, `afiro`).
2. The **Overshoot / Boundary Penetration** on ill-conditioned problems (e.g. `blend`, `bandm`, `e226`).

---

## 1. Mathematical Foundations

### Root Cause Analysis of Iteration 4.1 Failures

In Iteration 4.1, the loss included:
$$\mathcal{L}_{4.1} = w_{\text{obj}} \tilde{c}^T x + w_{\text{gap}} (\tilde{c}^T x + \tilde{h}^T \lambda)^2 + \dots$$

1. **The Equilibrium Trap:**
   When $\lambda \approx 0$, the gradient with respect to $x$ is:
   $$\nabla_x \mathcal{L} = w_{\text{obj}} \tilde{c} + 2 w_{\text{gap}} (\tilde{c}^T x) \tilde{c} = 0 \implies \tilde{c}^T x = -\frac{w_{\text{obj}}}{2 w_{\text{gap}}} = -0.10$$
   The squared duality gap acted as a quadratic restoring spring that balanced the linear pull at $\tilde{c}^T x = -0.10$, halting optimization and producing a 99.85% gap on problems whose true optimum was negative (e.g. `sc50a` with optimal -64.58).

2. **Persistent Gravity Overshoot:**
   A constant linear pull $w_{\text{obj}} \tilde{c}^T x$ exerts an uninterrupted downward force throughout all 1500 epochs. Once the solution reached the polytope boundary, the persistent pull overpowered the penalty wall on 11 problems.

---

## 2. Iteration 4.2 Innovations

### Innovation 1: Annealed Objective Pull
Instead of a constant weight, the objective pull decays exponentially over epochs:
$$w_{\text{obj}}(t) = w_{\text{obj,init}} \cdot \gamma^t, \quad \gamma = 0.996$$
- **Early epochs ($t < 150$):** High pull force expels the solver from the trivial origin trap ($x \approx 0$) into the polytope interior.
- **Late epochs ($t > 400$):** $w_{\text{obj}}(t) \to 0$, eliminating distorting gravity so exact KKT conditions and boundary walls govern final convergence without boundary penetration.

### Innovation 2: One-Sided Strong Duality Gap
By Weak Duality, for any primal feasible $x$ and dual feasible $\lambda$:
$$\tilde{c}^T x + \tilde{h}^T \lambda \ge 0$$
At optimality (Strong Duality): $\tilde{c}^T x^* + \tilde{h}^T \lambda^* = 0$.
We formulate the duality gap as a one-sided penalty:
$$\mathcal{L}_{\text{gap}} = \operatorname{ReLU}(\tilde{c}^T x + \tilde{h}^T \lambda)^2$$
- Penalizes suboptimality ($\tilde{c}^T x + \tilde{h}^T \lambda > 0$).
- Does **not** penalize progress ($\tilde{c}^T x + \tilde{h}^T \lambda \le 0$).
- Completely eliminates the $-0.10$ restoring spring trap!

### Innovation 3: Pure KKT Residual Checkpointing
Rather than tracking the lowest raw loss (which favored solutions with large negative objectives that violated constraints), the solver checkpoints based on the **pure KKT residual**:
$$\mathcal{R}_{\text{KKT}} = w_{\text{stat}} \mathcal{L}_{\text{stat}} + w_{\text{gap}} \mathcal{L}_{\text{gap}} + w_{\text{fb}} \mathcal{L}_{\text{fb}} + w_{\text{prim}} \mathcal{L}_{\text{prim}} + w_{x} \mathcal{L}_{x\text{-pos}}$$
Only points that genuinely satisfy the KKT conditions are saved.

### Innovation 4: Decoupled Architecture & Canonical Preconditioning
- Primal and Dual branches are completely decoupled (separate hidden layers).
- Dual output layer initialized with $+1.0$ linear bias to avoid Softplus saturation.
- Row-wise Euclidean normalization on $G, h$ ensures $\mathcal{O}(1)$ constraint slacks.
- Cost vector Euclidean normalization stabilizes gradients.
- Exact dual multiplier unscaling upon completion: $\lambda_i = s_c \frac{\tilde{\lambda}_i}{d_i}$.

---

## 3. Architecture Summary

```
Input: Fixed Latent Seeds z_p, z_d in R^16
|
|--> Primal Net: Linear(16,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64, n_vars) -> x_hat
|
|--> Dual Net:   Linear(16,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64, n_cons) -> Softplus (bias=+1.0) -> lambda_hat
```

---

## 4. Usage

```python
from KKT_Standalone_Reader import load_problem
from KKT_Solver_Iteration_4_2 import solve_kkt_instance, evaluate_solution

prob = load_problem("path/to/problem.mps.gz")
result = solve_kkt_instance(prob, max_epochs=1200, lr=0.015)
metrics = evaluate_solution(prob, result["x_opt"], result["lambda_opt"])

print(f"Primal Objective: {metrics['primal_objective']:.2f}")
print(f"Primal Infeasibility: {metrics['max_primal_violation']:.2e}")
```
