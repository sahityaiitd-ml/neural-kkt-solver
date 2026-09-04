# Neural Network Based KKT Solver

**B.Tech. Project (Industrial Optimization)**  
**Department of Mechanical Engineering**  

### Project Team:
* **Students:** Sahitya Rankawat (2023ME21131), Manav Gupta (2023ME20733), Jeet Anand (2023ME20874)  
* **Supervisor:** Prof. Kartikey Sharma  

---

## Overview
This project develops an ultra-fast, data-driven optimization solver based on solving the **Karush-Kuhn-Tucker (KKT) optimality conditions** using **Neural Networks**. 

Instead of relying on slow, sequential iterative algorithms (like classical Simplex or Interior-Point methods), this neural solver directly learns to satisfy the KKT optimality conditions in an unsupervised/physics-informed framework, outputting optimal primal variables (x*) and dual multipliers (\lambda^*) in a single forward pass.

The `pyomo_kkt_pipeline.ipynb` is the Notebook representation of `pyomo_pipeline` for better representation and understanding. Once we agree on the changes the main folders are then modified.
---

## Repository Structure

```
├── pyomo_kkt_pipeline.ipynb      # Main end-to-end interactive notebook (Pyomo -> KKT -> ANN)
├── pyomo_pipeline/               # Core Python package for Pyomo modeling & KKT extraction
│   ├── model_builder.py          # Parametric LP model builders
│   ├── kkt_extractor.py          # Canonical (c, G, h) extraction engine
│   ├── solver.py                 # HiGHS direct solver integration
│   └── kkt_evaluator.py          # 4-term KKT residual error verification
├── benchmarks/                  # Benchmark dataset generation pipeline
│   ├── input/                   # Benchmark problem files (.mps, .lp, .mps.gz from Netlib & MIPLIB)
│   ├── dataset/                 # Preprocessed canonical (.npz) instances with summary.json
│   └── batch_kkt_dataset_pipeline.py  # Automated batch dataset builder
├── literature_review/           # Foundational research papers
│   ├── 2409.09087v1.pdf         # KINN (KKT-Informed Neural Networks)
│   └── 2410.15973v1.pdf         # KKT Nets (IIT Dharwad)
├── reports/                     # Bi-weekly project progress reports
└── .gitignore                   # Standard Python / environment ignore rules
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or directly:
pip install pyomo highspy torch numpy matplotlib
```

### 2. Run Main Pipeline Notebook
Open and run `pyomo_kkt_pipeline.ipynb` in Jupyter or Google Colab:
- **Part 1:** Problem modeling in Pyomo and direct solving with HiGHS.
- **Part 2:** Automated KKT algebraic extraction and residual validation ($\approx 10^{-16}$ error).
- **Part 3:** HiGHS benchmark file reader & first iteration Artificial Neural Network (ANN) solver inspection.
