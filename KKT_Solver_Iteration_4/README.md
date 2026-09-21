# KKT_Solver_Iteration_4: Decoupled Architecture & Linear Objective Pull

## Overview
Iteration 4 directly resolves the two fundamental limitations diagnosed in early physics-informed solvers:
1. **Gradient Cross-Talk in Shared Representation:** Primal variables x and dual multipliers lambda had opposing gradient demands that collided in the shared hidden layers.
2. **The Trivial Origin Trap (x ~ 0):** Because quadratic penalties and squared duality gaps vanish near the origin, the solver frequently collapsed into a lazy interior point where x ~ 0, producing ~99% objective gaps on models like `afiro`, `sc50a`, and `blend`.

## Key Technical Enhancements
- **Decoupled Primal & Dual Networks:** Primal and dual heads branch from completely separate 2-layer GELU networks with independent weights and latent seeds.
- **Linear Primal Objective Pull:** Adds w_obj * (c_tilde^T * x_hat) to the loss. Because its gradient is constant (c_tilde), it exerts persistent downward gravitational pull dragging x away from the origin toward the cost-minimizing boundary.
- **Canonical Row Normalization:** Retains unit row-scaling to prevent loss explosions.
- **Anti-Saturation Dual Initialization:** Retains positive bias initialization (+1.0) on the dual head so multipliers start in the high-gradient linear regime of Softplus.

## Usage
```bash
python KKT_Solver_Iteration_4/run_iteration_4.py
```
