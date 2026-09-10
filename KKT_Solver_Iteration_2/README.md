# KKT Solver - Iteration 2

**Hypothesis:** Iteration 1's baseline is held back by two avoidable things in its
learning setup, not by the KKT formulation itself:

1. **Dead ReLU neurons.** The backbone uses `nn.ReLU()`. Early in training the
   network sits well outside the feasible region, many pre-activations go
   negative, and those units receive zero gradient for the rest of the run. The
   effective capacity of the network shrinks exactly when it still has the most
   work to do.
2. **A loss term fighting itself.** Dual feasibility (`λ ≥ 0`) is only *penalised*
   (`w_dual_pos · ‖max(0, −λ)‖²`). That penalty competes with stationarity and
   complementary slackness for the same gradient budget, and near the optimal
   face the three terms trade off against each other instead of all going to zero.

## What changed from Iteration 1

| File | Change |
| :--- | :--- |
| `model.py` | `nn.ReLU()` → `nn.GELU()` in the backbone. Dual head `nn.Linear` → `nn.Sequential(nn.Linear, nn.Softplus())`, so `λ̂ = softplus(z) ≥ 0` for every forward pass. |
| `loss.py` | Removed the dual non-negativity term and its `w_dual_pos` weight. It is now identically zero by construction, so it carries no gradient signal. The primal non-negativity term (`x ≥ 0`) is **unchanged** from Iteration 1. |
| `solver.py` | Dropped the `w_dual_pos` argument. Optimiser, scheduler, gradient clipping, and best-loss checkpoint snapshotting are unchanged. |
| `evaluate.py`, `plot_loss.py` | Identical to Iteration 1. |

Everything else — the latent seed, hidden width (64), Adam + `ReduceLROnPlateau`,
`max_norm=10` clipping, `tol=1e-4` early stop — is byte-identical to Iteration 1
so the comparison isolates the GELU + Softplus change.

## Loss (4 terms)

$$\mathcal{L} = w_{\text{stat}}\,\|c + G^\top \hat\lambda\|_2^2
             + w_{\text{prim}}\,\|\max(0, G\hat x - h)\|_2^2
             + w_{\text{slack}}\,\|\hat\lambda \odot (G\hat x - h)\|_2^2
             + w_{\text{x+}}\,\|\max(0, -\hat x)\|_2^2$$

Dual feasibility $\hat\lambda \ge 0$ is enforced by the Softplus head, not the loss.

## Expected effect

- **Dual violation → exactly 0** on every problem (structural, not learned).
- Lower **stationarity** and **complementary-slackness** residuals at convergence,
  since those terms no longer share gradient budget with a dual-sign penalty.
- Fewer epochs to reach `tol`, and a smoother convergence curve
  (`convergence_production_plan.png`) from GELU keeping the backbone trainable.
- **No change** to how negative-`x` problems are handled — the `x ≥ 0` term is the
  same as Iteration 1, so problems with free or negative-lower-bound variables
  carry the same caveat they did before.

## Running

```bash
# Stage 1: single-instance sanity check
python KKT_Solver_Iteration_2/run_iteration_2.py

# Stage 2: 8-problem Netlib scorecard
python - <<'PY'
from KKT_Benchmark import evaluate_solver_on_benchmark
from KKT_Solver_Iteration_2 import solve_kkt_instance

def run_iter2(kkt_sys):
    return solve_kkt_instance(kkt_sys, max_epochs=1200, lr=0.015, verbose=False)

evaluate_solver_on_benchmark(
    solver_fn=run_iter2,
    solver_name="Iteration 2 (GELU + Softplus dual head)",
    problem_names=["afiro", "flugpl", "adlittle", "bell5", "e226", "blend", "share2b", "sc50a"],
)
PY
```

> Requires `torch` (see project `requirements.txt`). Note: the committed `blend`
> ground-truth solution is inconsistent with the reader's parsed `(c, G, h)` —
> its scorecard row is not meaningful until that benchmark is regenerated.

## Findings

_To be filled in after running Stage 1 and Stage 2. Record the scorecard here and
copy a one-paragraph summary into `Weekly Reports (Home Device)/README.md`._
