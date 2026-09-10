# KKT Standalone Reader

A high-performance, solver-agnostic problem reader and KKT formulation engine built specifically for **KKT-Informed Neural Networks (KINNs)** and optimization research.

---

## Key Highlights

* **100% Pure Python + NumPy:** Works on **Mac, Windows, Linux, and Google Colab** with zero compiler or installation headaches.
* **Zero Solver Dependency for Reading:** Reads `.mps`, `.lp`, `.mps.gz`, `.json`, and Pyomo models without requiring HiGHS or Gurobi to be installed.
* **Unified Canonical Output:** Always produces the standardized $(c, G, h)$ inequality system where:
  $$\min_x c^T x \quad \text{subject to} \quad G x \le h$$
* **KINN & GNN Ready:**
  * `problem.to_torch()` produces PyTorch float32 tensors ready for immediate neural network training.
  * `problem.sparse_coo` provides `(row_idx, col_idx, values)` for Graph Neural Networks.

---

## Quick Start

### 1. Load an MPS or LP File (Netlib / MIPLIB)
```python
from KKT_Standalone_Reader import load_problem

# Reads .mps, .lp, or compressed .mps.gz / .lp.gz
problem = load_problem("examples/1_diet_problem.mps")

print(problem.summary())
# Access canonical matrices:
c, G, h = problem.c, problem.G, problem.h
```

### 2. Export Directly to PyTorch for KINN
```python
tensors = problem.to_torch(device="cpu")
c_tensor = tensors["c"]  # shape (n,)
G_tensor = tensors["G"]  # shape (m, n)
h_tensor = tensors["h"]  # shape (m,)
```

---

## The 4 User-Facing Input Methods

| Method | Source | Example Code |
| :--- | :--- | :--- |
| **1. File Format** | `.mps`, `.lp`, `.mps.gz` | `problem = load_problem("examples/1_diet_problem.mps")` |
| **2. Pyomo Model** | `pyo.ConcreteModel` | `problem = load_problem(my_pyomo_model)` |
| **3. JSON Format** | `.json` file or `dict` | `problem = load_problem("examples/3_supply_chain.json")` |
| **4. Raw Matrices** | Direct NumPy arrays | `from KKT_Standalone_Reader import read_matrices`<br>`problem = read_matrices(c, G, h)` |

---

## Testing and Verification

Run the automated test suite to verify dimensions and check that KKT residual errors against HiGHS ground truth are $< 10^{-14}$:

```bash
python3 KKT_Standalone_Reader/tests/test_reader.py
```

All 5 test suites (MPS, LP, JSON, Pyomo, and real Netlib `afiro.mps`) pass with **100% mathematical precision**.
