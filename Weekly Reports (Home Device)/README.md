# Weekly Reports & Progress Tracker (Home Device)

This folder tracks all research milestones, project documentation, weekly progress reports, and uploaded documents completed collaboratively for the **BTech Project: Neural KKT Single-Instance LP Solver**.

---

## 📑 Repository of Reports & Uploaded Documents

| Document | Folder / Location | Format | Description / Milestones Covered |
| :--- | :--- | :--- | :--- |
| `I1 Progress report_1.pdf` | `IIT Delhi Reports/` | PDF | Initial Iteration 1 Progress Report submitted for BTP review. |
| `I1 Progress Report_2.pdf` | `IIT Delhi Reports/` | PDF | Second Iteration 1 Progress Report covering early formulations and pipeline design. |
| `Weekly_Progress_Report_Week_1.md` | Root of Weekly Reports | Markdown | **Week 1 Comprehensive Report:**<br>• Universal Standalone Reader (`KKT_Standalone_Reader`) with dual-engine parser (MPS, LP, Pyomo, JSON).<br>• First-Generation Single-Instance KINN Baseline Solver (`KKT_Solver_Iteration_1`) with 5-term physics-informed loss and plateau LR scheduling.<br>• Official Academic Benchmark Suite (`KKT_Benchmark`) comprising 412 Netlib & MIPLIB instances and 393 verified solutions.<br>• Mathematical metric definitions (Relative Gap, KKT Residuals) and theoretical insights (non-uniqueness, matrix sparsity, error stagnation). |

---

## 📅 Chronological Timeline & Future Log

- [x] **Week 1 (Completed Sep 10, 2026):**
  - Built `KKT_Standalone_Reader` (5/5 tests passing with $< 10^{-14}$ residual error).
  - Built `KKT_Solver_Iteration_1` baseline single-instance solver.
  - Curated 412 official benchmark instances in `KKT_Benchmark` with 393 verified HiGHS ground-truth solutions across 4 scale tiers.
  - Defined strict zero-I/O pure algorithmic solve-time benchmarking protocol.
  - Established KKT residual metric framework and investigated LP non-uniqueness & matrix sparsity.

- [ ] **Week 2 (Upcoming):**
  - Implement `KKT_Solver_Iteration_2`:
    - `Softplus` activation on the dual head to guarantee $\hat{\lambda} \ge 0$.
    - Continuous smooth hidden activations (`GELU` or `SiLU`) to eliminate dead ReLU units.
    - Augmented Lagrangian Multiplier (ALM) method to resolve loss plateaus and opposing gradient deadlocks.
  - Evaluate Iteration 2 on `KKT_Benchmark` harness against Iteration 1 baseline.

- [ ] **Week 3 (Planned):**
  - GPU batching architecture and parallel multi-instance solving.
  - Feasibility projection layers and warm-starting studies for MIP relaxations.

---
*Note: Any new reports or documents uploaded will be archived and referenced in this directory.*
