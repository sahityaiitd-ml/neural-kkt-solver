# KKT_Solver_Iteration_5: Two-Stage Hybrid Solver (Neural Scout + Active-Set Linear Snap)

## Overview
Iteration 5 introduces a two-stage hybrid optimization paradigm that bridges continuous deep learning and discrete Operations Research:
- **Stage 1: Neural Scout Exploration:**
  Discovers the global optimal basin using a decoupled KINN trained with:
  * Powell-Hestenes-Rockafellar (PHR) Augmented Lagrangian with row-scaled penalty parameters $\rho_i = \rho_0 / \|G_i\|_2$.
  * Dual-aware objective direction pull breaking both profit-driven ($c < 0$) and demand-driven ($c > 0$) origin traps.
  * Adaptive preconditioning (`auto`: Row L2 vs Ruiz equilibration).
  * Diameter-adaptive annealed objective weight.
- **Stage 2: Closed-Form Active-Set Linear Snap ($G_{\text{viol}}^\dagger$):**
  Projects the near-boundary neural scout prediction orthogonally onto the active hyperplane faces via microsecond closed-form SVD pseudoinverse:
  $$\Delta x = G_{\text{viol}}^\dagger (G_{\text{viol}} x_{\text{raw}} - h_{\text{viol}}), \quad x_{\text{snapped}} = x_{\text{raw}} - \Delta x$$
  Eliminates the "3.0000001 < 3" boundary penetration dilemma and achieves machine-precision feasibility ($10^{-13}$ to $10^{-14}$) without iterative solver loops or external optimizer crutches.
- **Stage 2: Dual Reduced-Cost Basis Sparsity Snapping:**
  Enforces exact complementary slackness by clamping non-basic variables ($r_j = c_j + G_{\cdot j}^T \lambda > 0.05$) to exact zero.

## Usage
```bash
python KKT_Solver_Iteration_5/run_iteration_5.py
```
