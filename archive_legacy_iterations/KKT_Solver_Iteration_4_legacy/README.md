# KKT_Solver_Iteration_4: Preconditioned Strong Duality KINN

## Overview
Iteration 4 introduces three major advancements to resolve the vanishing and exploding gradient bottlenecks observed in previous iterations:

1. **Step 1: Canonical Row & Cost Preconditioning (Equilibration)**
   - Normalizes constraint rows so that $\|G_i\|_2 = 1.0$.
   - Normalizes cost vector by $s_c = \max(1.0, \|c\|_2)$.
   - Slacks $\tilde{s}_i = \tilde{h}_i - \tilde{G}_i x$ represent exact perpendicular geometric distances on an $\mathcal{O}(1)$ scale, eliminating the $10^{12}$ loss explosion seen on large-scale models.

2. **Step 2 + Strong Duality Theorem: Duality Gap Loss**
   - Introduces $\mathcal{L}_{\text{gap}} = (\tilde{c}^T \hat{x} + \tilde{h}^T \tilde{\lambda})^2$.
   - Directly couples the cost vector $c$ to the primal decision variables $\hat{x}$, preventing the primal gradient from vanishing inside the feasible polytope.

3. **Step 4: Anti-Saturation Dual Head Initialization**
   - Initializes the dual head bias to $+1.0$ so multipliers begin in the high-gradient linear regime ($\sigma(z) \approx 0.73$) rather than the zero-gradient flat regime of Softplus.

4. **Normalized Fischer-Burmeister Operator**
   - Applies the smoothed C-function $\phi_\epsilon(\tilde{\lambda}_i, \tilde{s}_i)$ on normalized coordinates, smoothly enforcing complementary slackness without gradient explosion.

## Architecture
- **Backbone:** 2 hidden layers with GELU activation (64 units).
- **Primal Head:** Linear projection to $\mathbb{R}^n$.
- **Dual Head:** Linear projection (bias initialized to $+1.0$) followed by Softplus to strictly enforce non-negativity.
- **Dual Multiplier Unscaling:** Automatically unscales $\lambda_i = s_c \frac{\tilde{\lambda}_i}{d_i}$ for exact mathematical compliance with standard KKT residuals.
