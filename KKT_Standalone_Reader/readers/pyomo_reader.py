"""
pyomo_reader.py
===============
Converts a Pyomo ConcreteModel into the standardized KKTSystem.

Why is this useful?
-------------------
When you want to write a custom optimization problem in Python, Pyomo is the most
natural way to express mathematical equations:
    model.x1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=3*model.x1 + 2*model.x2)
    model.c1  = pyo.Constraint(expr=2*model.x1 + model.x2 <= 10)

This reader takes that high-level Pyomo model and directly extracts the canonical
matrices (c, G, h) for your KINN neural network!
"""

from typing import List, Dict, Optional
import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

from ..core.kkt_system import KKTSystem
from ..core.canonicalizer import build_canonical_kkt_system


def read_pyomo(model: pyo.ConcreteModel) -> KKTSystem:
    """
    Parses a Pyomo ConcreteModel and returns a standardized KKTSystem.
    """
    # 1. Index all active variables
    var_objects = list(model.component_data_objects(pyo.Var, active=True))
    n_vars = len(var_objects)
    var_names = [v.name for v in var_objects]
    var_map = {id(v): idx for idx, v in enumerate(var_objects)}

    # Extract variable bounds
    col_lower = np.array([float(v.lb) if v.lb is not None else -np.inf for v in var_objects], dtype=np.float64)
    col_upper = np.array([float(v.ub) if v.ub is not None else np.inf for v in var_objects], dtype=np.float64)

    # 2. Extract Objective
    obj_objects = list(model.component_data_objects(pyo.Objective, active=True))
    if not obj_objects:
        raise ValueError("The provided Pyomo model does not have an active Objective.")
    obj = obj_objects[0]
    
    sense = "maximize" if obj.sense == pyo.maximize else "minimize"
    repn_obj = generate_standard_repn(obj.expr)
    
    col_cost = np.zeros(n_vars, dtype=np.float64)
    if repn_obj.linear_vars:
        for v, coef in zip(repn_obj.linear_vars, repn_obj.linear_coefs):
            col_cost[var_map[id(v)]] += float(coef)
            
    offset = float(repn_obj.constant) if repn_obj.constant is not None else 0.0

    # 3. Extract Constraints
    row_names = []
    row_lower_list = []
    row_upper_list = []
    row_coefs = []

    for con in model.component_data_objects(pyo.Constraint, active=True):
        repn_con = generate_standard_repn(con.body)
        c_dict = {}
        if repn_con.linear_vars:
            for v, coef in zip(repn_con.linear_vars, repn_con.linear_coefs):
                c_dict[var_map[id(v)]] = float(coef)

        const = float(repn_con.constant) if repn_con.constant is not None else 0.0

        lo = float(con.lower) - const if con.lower is not None else -np.inf
        up = float(con.upper) - const if con.upper is not None else np.inf

        row_names.append(con.name)
        row_lower_list.append(lo)
        row_upper_list.append(up)
        row_coefs.append(c_dict)

    model_name = getattr(model, "name", "pyomo_model") or "pyomo_model"

    return build_canonical_kkt_system(
        name=model_name,
        n_vars=n_vars,
        var_names=var_names,
        col_cost=col_cost,
        sense=sense,
        row_names=row_names,
        row_lower=np.array(row_lower_list, dtype=np.float64),
        row_upper=np.array(row_upper_list, dtype=np.float64),
        row_coefs=row_coefs,
        col_lower=col_lower,
        col_upper=col_upper,
        offset=offset
    )
