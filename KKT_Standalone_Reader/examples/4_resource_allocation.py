"""
4_resource_allocation.py
========================
Medium-scale multi-project resource allocation problem formulated in Pyomo.

Scenario:
---------
An engineering organization has 6 potential R&D projects (P1 to P6) competing
for 3 scarce organizational resources:
  1. Senior Engineering Hours   (Limit: 1,200 hours)
  2. Cloud GPU Compute Budget   (Limit: 2,500 GPU-hours)
  3. Direct Capital Investment  (Limit: $500k)

Goal:
-----
Maximize the total organizational Return on Investment (ROI).
"""

import pyomo.environ as pyo


def create_resource_allocation_model() -> pyo.ConcreteModel:
    """
    Builds and returns the Pyomo ConcreteModel for resource allocation.
    """
    model = pyo.ConcreteModel(name="Multi_Project_Resource_Allocation")

    # 6 Candidate Projects
    projects = ["P1_AI_Search", "P2_Edge_Vision", "P3_KKT_Solver", "P4_Data_Lake", "P5_Robotics", "P6_NLP_Core"]
    model.PROJECTS = pyo.Set(initialize=projects)

    # Expected ROI (Profit in $k per unit of project funding)
    roi = {
        "P1_AI_Search":   65.0,
        "P2_Edge_Vision": 50.0,
        "P3_KKT_Solver":  95.0,
        "P4_Data_Lake":   40.0,
        "P5_Robotics":    80.0,
        "P6_NLP_Core":    75.0,
    }

    # Senior Engineering Hours required per unit of funding
    eng_hours = {
        "P1_AI_Search":   120.0,
        "P2_Edge_Vision":  80.0,
        "P3_KKT_Solver":  160.0,
        "P4_Data_Lake":    50.0,
        "P5_Robotics":    140.0,
        "P6_NLP_Core":    110.0,
    }

    # Cloud GPU Compute hours required
    gpu_hours = {
        "P1_AI_Search":   300.0,
        "P2_Edge_Vision": 250.0,
        "P3_KKT_Solver":  400.0,
        "P4_Data_Lake":   100.0,
        "P5_Robotics":    350.0,
        "P6_NLP_Core":    450.0,
    }

    # Capital cost ($k)
    capital_cost = {
        "P1_AI_Search":    60.0,
        "P2_Edge_Vision":  45.0,
        "P3_KKT_Solver":   80.0,
        "P4_Data_Lake":    35.0,
        "P5_Robotics":     90.0,
        "P6_NLP_Core":     70.0,
    }

    # Decision variables: fraction of project executed [0.0, 1.5]
    def _bounds_rule(m, p):
        return (0.0, 1.5)

    model.funding = pyo.Var(model.PROJECTS, domain=pyo.NonNegativeReals, bounds=_bounds_rule)

    # Objective: Maximize total expected ROI
    model.total_roi = pyo.Objective(
        expr=sum(roi[p] * model.funding[p] for p in model.PROJECTS),
        sense=pyo.maximize
    )

    # Constraint 1: Total Senior Engineering Hours <= 1,200
    model.c_eng = pyo.Constraint(
        expr=sum(eng_hours[p] * model.funding[p] for p in model.PROJECTS) <= 1200.0
    )

    # Constraint 2: Total Cloud GPU Compute <= 2,500
    model.c_gpu = pyo.Constraint(
        expr=sum(gpu_hours[p] * model.funding[p] for p in model.PROJECTS) <= 2500.0
    )

    # Constraint 3: Total Capital Investment <= 500k
    model.c_capital = pyo.Constraint(
        expr=sum(capital_cost[p] * model.funding[p] for p in model.PROJECTS) <= 500.0
    )

    # Constraint 4: Strategic balance: Core AI (P1 + P3 + P6) must receive at least 1.0 unit
    model.c_strategic = pyo.Constraint(
        expr=model.funding["P1_AI_Search"] + model.funding["P3_KKT_Solver"] + model.funding["P6_NLP_Core"] >= 1.0
    )

    return model


if __name__ == "__main__":
    from KKT_Standalone_Reader import load_problem

    # Build the model
    model = create_resource_allocation_model()
    
    # Load into standardized KKTSystem
    kkt_sys = load_problem(model)
    print(kkt_sys.summary())
