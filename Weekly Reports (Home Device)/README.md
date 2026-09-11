# Weekly Reports and Progress Tracker (Home Device)

This folder tracks all research milestones, project documentation, weekly progress reports, and uploaded documents completed collaboratively for the BTech Project: Neural Network Based KKT Solver for Linear Programming.

---

## Repository of Reports and Uploaded Documents

| Document | Location | Format | Description and Milestones Covered |
| :--- | :--- | :--- | :--- |
| `I1 Progress report_1.pdf` | `IIT Delhi Reports/` | PDF | Initial Iteration 1 Progress Report submitted for BTP review. |
| `I1 Progress Report_2.pdf` | `IIT Delhi Reports/` | PDF | Second Iteration 1 Progress Report covering early formulations and pipeline design. |
| `Weekly_Progress_Report_Week_1.md` | Root of Weekly Reports | Markdown | **Week 1 Comprehensive Report:**<br>- Universal Standalone Reader (`KKT_Standalone_Reader`) with dual-engine parser (MPS, LP, Pyomo, JSON).<br>- First-Generation Single-Instance KINN Baseline Solver (`KKT_Solver_Iteration_1`) with 5-term physics-informed loss and plateau LR scheduling.<br>- Official Academic Benchmark Suite (`KKT_Benchmark`) comprising 412 Netlib and MIPLIB instances and 393 verified solutions.<br>- Mathematical metric definitions (Relative Gap, KKT Residuals) and theoretical insights. |
| `Weekly_Progress_Report_Week_1.1.md` | Root of Weekly Reports | Markdown | **Week 1.1 Iteration 2 Report:**<br>- Integration of GELU backbone and structural Softplus dual multiplier head.<br>- Head-to-head 25-problem benchmark stress test across Netlib and MIPLIB instances (68% improvement rate, 100% elimination of dual infeasibility).<br>- Seven core research insights on gradient deadlock, uniform weighting flaws, and premature neuron death.<br>- Technical considerations for Iteration 3. |
| `Weekly_Progress_Report_Week_2.md` | Root of Weekly Reports | Markdown | **Week 2 Iteration 3 Architecture & 25-Problem Benchmark:**<br>- Selection and formulation of Path B: Smoothed Fischer-Burmeister complementarity function (arXiv:2507.08124v1).<br>- Automated persistent JSON/CSV benchmarking and comparative analysis pipeline in `KKT_Benchmark/results/`.<br>- Implementation of `KKT_Solver_Iteration_3` unifying primal feasibility and slackness.<br>- Architecture dry run validation on `2_production_plan.lp` (objective gap improved from 6.61% to 1.56%).<br>- Official 25-problem head-to-head stress test across Iterations 1, 2, and 3 (breakthroughs on `flugpl` to 0.96% gap and `adlittle` to 19.31% gap).<br>- Analysis of unscaled slack vulnerabilities and roadmap for Iteration 4 (matrix preconditioning + Augmented Lagrangian). |

---

## Chronological Timeline and Milestone Log

- [x] **Week 1 (Completed September 10, 2026):**
  - Built `KKT_Standalone_Reader` (5/5 unit tests passing with < 1e-14 residual error).
  - Built `KKT_Solver_Iteration_1` baseline single-instance solver.
  - Curated 412 official benchmark instances in `KKT_Benchmark` with 393 verified HiGHS ground-truth solutions across 4 scale tiers.
  - Defined strict zero-I/O pure algorithmic solve-time benchmarking protocol.
  - Established KKT residual metric framework and investigated LP non-uniqueness and matrix sparsity.

- [x] **Week 1.1 (Completed September 11, 2026):**
  - Evaluated Iteration 2 (`KKT_Solver_Iteration_2`) merged from `Jeet-Dev`.
  - Upgraded backbone from ReLU to GELU and implemented structural Softplus dual head.
  - Verified 100% elimination of dual multiplier violations across all test models.
  - Executed 25-problem benchmark stress test comparing Iteration 1 vs. Iteration 2 (total suite executed in 12.34 seconds, ~490 ms per LP).
  - Uncovered seven foundational insights into gradient deadlock, asymmetric convergence, and scaling disparities.
  - Documented architectural requirements for Iteration 3.

- [x] **Week 2 (Completed September 11, 2026):**
  - Implemented `KKT_Solver_Iteration_3` following Path B (Smoothed Fischer-Burmeister C-function).
  - Completed initial architecture dry run on `2_production_plan.lp` (gap reduced to 1.56%, dual violations strictly 0.000).
  - Engineered persistent JSON and CSV benchmark archiving in `KKT_Benchmark/results/` for instant zero-rerun comparisons.
  - Executed official 25-problem stress test across Iterations 1, 2, and 3 under the zero-I/O timing protocol.
  - Analyzed empirical trade-offs and established the necessity of row preconditioning alongside Augmented Lagrangian for Iteration 4.

- [ ] **Iteration 4 (Planned):**
  - Systematic problem preconditioning (Ruiz / geometric mean row-scaling) during ingestion.
  - Augmented Lagrangian (ALM) dynamic multiplier updates and adaptive penalty schedules.

