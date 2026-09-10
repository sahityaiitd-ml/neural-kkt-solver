"""
json_reader.py
==============
Parses structured JSON files or Python dictionaries into a standardized KKTSystem.

Why is this useful?
-------------------
JSON is the universal language of modern software, web APIs, and databases.
If you have a web application, cloud service, or optimization pipeline that receives
problems via JSON, this reader directly converts that JSON into canonical (c, G, h)
without needing Pyomo, HiGHS, or any solver installed!

Example JSON structure:
-----------------------
{
    "name": "diet_sample",
    "sense": "minimize",
    "objective": {"bread": 2.0, "milk": 3.5, "eggs": 1.5},
    "constraints": [
        {
            "name": "calories",
            "expr": {"bread": 250, "milk": 150, "eggs": 70},
            "sense": ">=",
            "rhs": 2000
        },
        {
            "name": "protein",
            "expr": {"bread": 8, "milk": 8, "eggs": 6},
            "sense": ">=",
            "rhs": 55
        }
    ],
    "bounds": {
        "bread": [0.0, null],
        "milk": [0.0, null],
        "eggs": [0.0, 10.0]
    }
}
"""

import json
from typing import Dict, Any, Union, List
import numpy as np

from ..core.kkt_system import KKTSystem
from ..core.canonicalizer import build_canonical_kkt_system


def read_json(source: Union[str, Dict[str, Any]]) -> KKTSystem:
    """
    Parses a JSON file path or a Python dictionary into a KKTSystem.
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"Expected file path (str) or dictionary, got {type(source)}")

    name = data.get("name", "json_optimization_problem")
    sense = data.get("sense", "minimize")
    offset = float(data.get("offset", 0.0))

    # 1. Discover all variable names
    all_vars_set = set()
    obj_dict = data.get("objective", {})
    all_vars_set.update(obj_dict.keys())

    constraints_list = data.get("constraints", [])
    for con in constraints_list:
        expr = con.get("expr", {})
        all_vars_set.update(expr.keys())

    bounds_dict = data.get("bounds", {})
    all_vars_set.update(bounds_dict.keys())

    var_names = sorted(list(all_vars_set))
    var_map = {v: i for i, v in enumerate(var_names)}
    n_vars = len(var_names)

    # 2. Extract cost vector c
    col_cost = np.zeros(n_vars, dtype=np.float64)
    for v, coef in obj_dict.items():
        col_cost[var_map[v]] = float(coef)

    # 3. Extract constraints
    row_names = []
    row_lower_list = []
    row_upper_list = []
    row_coefs = []

    for i, con in enumerate(constraints_list):
        rname = con.get("name", f"con_{i+1}")
        expr = con.get("expr", {})
        c_dict = {var_map[v]: float(val) for v, val in expr.items()}

        rel = con.get("sense", "<=").strip()
        rhs = float(con.get("rhs", 0.0))

        if rel in ("<=", "=<", "<"):
            lo, up = -np.inf, rhs
        elif rel in (">=", "=>", ">"):
            lo, up = rhs, np.inf
        elif rel in ("==", "="):
            lo, up = rhs, rhs
        elif rel.lower() == "ranged":
            lo = float(con.get("lower", -np.inf))
            up = float(con.get("upper", np.inf))
        else:
            raise ValueError(f"Unrecognized constraint sense '{rel}' in constraint '{rname}'")

        row_names.append(rname)
        row_lower_list.append(lo)
        row_upper_list.append(up)
        row_coefs.append(c_dict)

    # 4. Extract variable bounds
    col_lower = np.zeros(n_vars, dtype=np.float64)
    col_upper = np.full(n_vars, np.inf, dtype=np.float64)

    for v, b_range in bounds_dict.items():
        if v in var_map:
            idx = var_map[v]
            lo = b_range[0] if len(b_range) > 0 and b_range[0] is not None else -np.inf
            up = b_range[1] if len(b_range) > 1 and b_range[1] is not None else np.inf
            col_lower[idx] = float(lo)
            col_upper[idx] = float(up)

    return build_canonical_kkt_system(
        name=name,
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
