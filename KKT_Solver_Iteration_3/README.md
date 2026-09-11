# KKT Solver - Iteration 3

## Hypothesis and Motivation

Iteration 1 identified dead neurons under ReLU and artificial stationarity cheating via negative dual multipliers. Iteration 2 addressed those with a GELU backbone and a Softplus dual head. 

However, benchmark stress testing across 25 problems revealed that the standard quadratic penalty formulation:
    L_slack = ||lambda (h - Gx)||^2
suffers from a key mathematical flaw: whenever a constraint is severely violated (s_i < 0) but the network predicts lambda_i = 0, the product lambda_i * s_i evaluates to zero. The network is therefore blinded to constraint violations in the slackness term.

In Iteration 3 (Path B), we implement the smoothed Fischer-Burmeister (FB) complementarity operator from Texas A&M (arXiv:2507.08124v1):
    phi_eps(lambda_i, s_i) = lambda_i + s_i - sqrt(lambda_i^2 + s_i^2 + eps)

Theorem:
    phi_eps(lambda_i, s_i) = 0  <=>  lambda_i >= 0, s_i >= 0, lambda_i * s_i = 0

This operator simultaneously enforces primal feasibility, dual feasibility, and complementary slackness in a single, smooth, differentiable penalty term.

---

## Architectural and Mathematical Summary

| Component | Iteration 2 | Iteration 3 | Purpose |
| :--- | :--- | :--- | :--- |
| Backbone Activation | GELU | GELU | Continuous gradient flow throughout infeasible space |
| Dual Multiplier Head | Softplus | Softplus | Unconditional dual feasibility (lambda >= 0) |
| Complementarity Loss | Simple product ||lambda * s||^2 | Fischer-Burmeister phi_eps(lambda, s) | Unifies feasibility and slackness; prevents zero-loss cheating when s < 0 |
| Primal Feasibility | Independent penalty | Embedded in FB + auxiliary penalty | Joint optimization of boundary proximity and multiplier pricing |

---

## Running the Architecture Dry Run

```bash
./.venv/bin/python KKT_Solver_Iteration_3/run_iteration_3.py
```
