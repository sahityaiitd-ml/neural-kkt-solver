"""
model_builder.py
================
Defines various Pyomo optimization models (LPs, parametric problems, resource allocation)
for demonstration, data generation, and KKT extraction.
"""

import pyomo.environ as pyo
import numpy as np


def create_sample_lp():
    """
    Creates a standard 2-variable LP:
        max  3*x1 + 2*x2
        s.t. 2*x1 +   x2 <= 10
               x1 + 3*x2 <= 12
               x1, x2 >= 0
    (Equivalent to minimizing -3*x1 - 2*x2)
    """
    model = pyo.ConcreteModel(name="Sample_2D_LP")
    
    # Variables
    model.x1 = pyo.Var(domain=pyo.NonNegativeReals, name="x1")
    model.x2 = pyo.Var(domain=pyo.NonNegativeReals, name="x2")
    
    # Objective (Minimize form standard)
    model.obj = pyo.Objective(expr=-3.0 * model.x1 - 2.0 * model.x2, sense=pyo.minimize)
    
    # Constraints
    model.c1 = pyo.Constraint(expr=2.0 * model.x1 + 1.0 * model.x2 <= 10.0)
    model.c2 = pyo.Constraint(expr=1.0 * model.x1 + 3.0 * model.x2 <= 12.0)
    
    return model


def create_diet_problem():
    """
    Classic Stigler Diet Problem:
    Find the cheapest combination of foods that satisfies daily nutritional requirements.
    
    Foods: [Bread, Milk, Eggs]
    Nutrients: [Calories, Protein, Fat]
    """
    model = pyo.ConcreteModel(name="Diet_Problem")
    
    foods = ["Bread", "Milk", "Eggs"]
    nutrients = ["Calories", "Protein", "Fat"]
    
    # Cost per unit of food
    cost = {"Bread": 2.0, "Milk": 3.5, "Eggs": 1.5}
    
    # Nutrition matrix (amount of nutrient per unit of food)
    nutrition = {
        ("Bread", "Calories"): 250, ("Bread", "Protein"): 8,   ("Bread", "Fat"): 2,
        ("Milk", "Calories"): 150,  ("Milk", "Protein"): 8,   ("Milk", "Fat"): 5,
        ("Eggs", "Calories"): 70,   ("Eggs", "Protein"): 6,   ("Eggs", "Fat"): 5,
    }
    
    # Minimum daily requirements
    min_req = {"Calories": 2000, "Protein": 60, "Fat": 30}
    
    # Variables: amount of each food to buy
    model.food = pyo.Var(foods, domain=pyo.NonNegativeReals)
    
    # Objective: Minimize total cost
    model.total_cost = pyo.Objective(
        expr=sum(cost[f] * model.food[f] for f in foods),
        sense=pyo.minimize
    )
    
    # Constraints: Satisfy nutritional requirements
    model.nutrition_con = pyo.ConstraintList()
    for n in nutrients:
        model.nutrition_con.add(
            sum(nutrition[f, n] * model.food[f] for f in foods) >= min_req[n]
        )
        
    return model


def create_resource_allocation_lp(num_products=4, num_resources=3, seed=42):
    """
    Creates a multi-variable Resource Allocation Linear Program:
        max  profit^T * x
        s.t. UsageMatrix * x <= Capacity
             x >= 0
    """
    rng = np.random.default_rng(seed)
    
    # Random problem parameters
    profits = rng.uniform(10, 50, size=num_products)
    capacities = rng.uniform(100, 500, size=num_resources)
    usage = rng.uniform(1, 10, size=(num_resources, num_products))
    
    model = pyo.ConcreteModel(name="Resource_Allocation")
    
    # Variables
    model.x = pyo.Var(range(num_products), domain=pyo.NonNegativeReals)
    
    # Objective (minimize negative profit)
    model.obj = pyo.Objective(
        expr=sum(-float(profits[j]) * model.x[j] for j in range(num_products)),
        sense=pyo.minimize
    )
    
    # Constraints
    model.capacity_con = pyo.ConstraintList()
    for i in range(num_resources):
        model.capacity_con.add(
            sum(float(usage[i, j]) * model.x[j] for j in range(num_products)) <= float(capacities[i])
        )
        
    return model


def create_parametric_lp(c, G, h, A=None, b=None, lb=None, ub=None):
    """
    Creates a Pyomo ConcreteModel from explicit matrices:
        min  c^T x
        s.t. G x <= h
             A x == b
             lb <= x <= ub
    
    Parameters
    ----------
    c : np.ndarray (n,)
    G : np.ndarray (m, n)
    h : np.ndarray (m,)
    A : np.ndarray (p, n), optional
    b : np.ndarray (p,), optional
    lb : np.ndarray (n,), optional
    ub : np.ndarray (n,), optional
    """
    n = len(c)
    model = pyo.ConcreteModel(name="Parametric_LP")
    
    # Define variables with bounds
    def var_bounds(m, i):
        lower = float(lb[i]) if (lb is not None and lb[i] is not None) else None
        upper = float(ub[i]) if (ub is not None and ub[i] is not None) else None
        return (lower, upper)
    
    model.x = pyo.Var(range(n), bounds=var_bounds)
    
    # Objective
    model.obj = pyo.Objective(
        expr=sum(float(c[i]) * model.x[i] for i in range(n)),
        sense=pyo.minimize
    )
    
    # Inequality constraints G x <= h
    model.ineq_con = pyo.ConstraintList()
    if G is not None and len(G) > 0:
        m_ineq = G.shape[0]
        for i in range(m_ineq):
            model.ineq_con.add(
                sum(float(G[i, j]) * model.x[j] for j in range(n)) <= float(h[i])
            )
            
    # Equality constraints A x == b
    model.eq_con = pyo.ConstraintList()
    if A is not None and len(A) > 0:
        p_eq = A.shape[0]
        for i in range(p_eq):
            model.eq_con.add(
                sum(float(A[i, j]) * model.x[j] for j in range(n)) == float(b[i])
            )
            
    return model
