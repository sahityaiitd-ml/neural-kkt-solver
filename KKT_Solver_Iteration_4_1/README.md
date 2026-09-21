# KKT_Solver_Iteration_4_1: Cumulative Iteration 4 Solver Suite

## Overview
Iteration 4.1 integrates all major breakthroughs developed across the Iteration 4 research campaign:
1. **Decoupled Architecture:** Independent primal and dual MLPs eliminating cross-talk between primal objective optimization and dual stationarity.
2. **One-Sided ReLU Strong Duality Gap:** $\mathcal{L}_{\text{gap}} = [\max(0, c_{\text{norm}}^T \hat{x} + h_{\text{norm}}^T \hat{\lambda})]^2$, strictly removing the negative restoring force that previously caused origin collapse.
3. **Diameter-Adaptive Annealed Objective Pull:** $w_{\text{obj}}(t) = w_0 \gamma_{\text{eff}}^t$, where $\gamma_{\text{eff}}$ adapts to the estimated polytope diameter, ensuring sustained exploration on wide polytopes (e.g. `agg`) and rapid convergence on compact ones (`sc50a`, `afiro`).
4. **Adaptive Preconditioning ("auto"):** Automatically chooses between Ruiz matrix equilibration (for ill-conditioned corridors like `blend`) and Row L2 scaling (for sparse-cost polytopes like `afiro` and `sc50a`).
5. **Pure KKT Metric Checkpointing:** Tracks and saves the model state based strictly on true KKT residuals rather than time-decaying loss values.

## Usage
```bash
python KKT_Solver_Iteration_4_1/run_iteration_4_1.py
```
