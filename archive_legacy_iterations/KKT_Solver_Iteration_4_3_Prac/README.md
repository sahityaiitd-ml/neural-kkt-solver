# KKT_Solver_Iteration_4_3_Prac: Experimental Diagnostic Sandbox

A dedicated experimental sandbox designed to systematically test and isolate architectural components:
1. **Preconditioning Regimes:** Canonical Row L2 vs. Symmetric Ruiz Equilibration.
2. **Annealing Mechanisms:** Fixed Exponential Decay vs. Polytope-Diameter Adaptive Decay.
3. **Unscaling Transformations:** Analytical Primal ($x = C^{-1} \tilde{x}$) and Dual ($\lambda = s_c R^{-1} \tilde{\lambda}$) mappings.

---

## Experimental Configurations

| Configuration | Preconditioning | Annealing Schedule | Description |
|---|---|---|---|
| **Config 1** | Row L2 | Fixed Exponential ($\gamma = 0.996$) | Iteration 4.2 Baseline |
| **Config 2** | Ruiz Equilibration | Fixed Exponential ($\gamma = 0.996$) | Row + Column scaling to test narrow corridor stability (`blend`) |
| **Config 3** | Row L2 | Diameter-Adaptive Annealing | Extends pull duration on wide polytopes (`agg`) |
| **Config 4** | Ruiz Equilibration | Diameter-Adaptive Annealing | Combined row/column equilibration + diameter-adaptive pull |

---

## Test Protocol

Execute the diagnostic ablation matrix across representative problem archetypes:
```bash
./.venv/bin/python KKT_Solver_Iteration_4_3_Prac/test_knobs.py
```
