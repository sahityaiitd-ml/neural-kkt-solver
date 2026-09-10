# KKT Solver - Iteration 1

A single-instance Physics-Informed Neural Network (KINN) optimization engine that dynamically builds a dual-head neural network for any Linear Programming problem and solves it using pure KKT physics.

---

## 🔬 How it Works

1. **Dynamic Architecture:**
   * Receives $(c, G, h)$ from `KKT_Standalone_Reader`.
   * Automatically sets **Primal Output Head** to size $n$ (decision variables $\hat{x}$).
   * Automatically sets **Dual Output Head** to size $m$ (multipliers $\hat{\lambda}$), followed by `Softplus` so $\hat{\lambda} \ge 0$ at all times.

2. **Physics-Informed KKT Loss:**
   $$\mathcal{L}_{\text{total}} = w_1 \|c + G^T \hat{\lambda}\|_2^2 + w_2 \|\max(0, G\hat{x} - h)\|_2^2 + w_3 \|\hat{\lambda} \odot (G\hat{x} - h)\|_2^2$$
   * No external labels or answers needed! The network minimizes the 3 physical conditions of optimization.

3. **Verification against HiGHS:**
   * Solves each benchmark problem with HiGHS to establish ground truth.
   * Logs Objective Gap ($\%$), maximum constraint violation, and KKT residual decay.

---

## 🚀 Running the Demonstration

To run the full suite on the example problems and Netlib's `afiro.mps`:

```bash
python3 KKT_Solver_Iteration_1/run_iteration_1.py
```
