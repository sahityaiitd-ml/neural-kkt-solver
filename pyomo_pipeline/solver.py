"""
solver.py
=========
Solves Pyomo models using available mathematical solvers (e.g., HiGHS, GLPK, CBC, Ipopt)
and extracts exact primal values, dual multipliers, and reduced costs for KKT verification.
"""

import pyomo.environ as pyo
import numpy as np


def solve_pyomo_model(model: pyo.ConcreteModel, solver_name: str = "appsi_highs", verbose: bool = False):
    """
    Solves a Pyomo ConcreteModel and imports dual variables and reduced costs.
    
    Parameters
    ----------
    model : pyo.ConcreteModel
    solver_name : str, default 'appsi_highs'
    verbose : bool, default False
    
    Returns
    -------
    dict:
        {
            "status": str ("optimal", "infeasible", etc.),
            "is_optimal": bool,
            "objective_value": float,
            "primal_vars": dict of {var_name: float},
            "primal_array": np.ndarray of x*,
            "dual_multipliers": dict of {con_name: float},
            "reduced_costs": dict of {var_name: float},
            "solver_name": str,
            "raw_results": SolverResults
        }
    """
    # Attach dual and reduced cost suffixes
    if not hasattr(model, "dual"):
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if not hasattr(model, "rc"):
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        for fallback in ["appsi_highs", "glpk", "cbc", "ipopt"]:
            fb_solver = pyo.SolverFactory(fallback)
            if fb_solver.available():
                solver = fb_solver
                solver_name = fallback
                break
        else:
            raise RuntimeError(f"No suitable solver found. Requested '{solver_name}', but it is not available.")
            
    # Solve
    results = solver.solve(model, tee=verbose)
    term_cond = str(results.solver.termination_condition)
    
    # Extract Primal values
    var_list = list(model.component_data_objects(pyo.Var, active=True))
    primal_dict = {}
    primal_vals = []
    rc_dict = {}
    for v in var_list:
        val = pyo.value(v)
        primal_dict[v.name] = float(val) if val is not None else 0.0
        primal_vals.append(primal_dict[v.name])
        try:
            rc_val = model.rc[v]
            rc_dict[v.name] = float(rc_val) if rc_val is not None else 0.0
        except KeyError:
            rc_dict[v.name] = 0.0
            
    primal_array = np.array(primal_vals, dtype=np.float64)
    
    # Extract Objective value
    obj_obj = next(model.component_data_objects(pyo.Objective, active=True))
    obj_val = float(pyo.value(obj_obj))
    
    # Extract Dual multipliers
    dual_dict = {}
    con_list = list(model.component_data_objects(pyo.Constraint, active=True))
    for con in con_list:
        try:
            dual_val = model.dual[con]
            dual_dict[con.name] = float(dual_val) if dual_val is not None else 0.0
        except KeyError:
            dual_dict[con.name] = 0.0
            
    return {
        "status": term_cond,
        "is_optimal": term_cond in ["optimal", "TerminationCondition.optimal", "locallyOptimal"],
        "objective_value": obj_val,
        "primal_vars": primal_dict,
        "primal_array": primal_array,
        "dual_multipliers": dual_dict,
        "reduced_costs": rc_dict,
        "solver_name": solver_name,
        "raw_results": results
    }


def extract_canonical_solution(extractor, solver_results):
    """
    Maps solver primal, dual, and reduced cost outputs directly to the
    canonical vector structures (x*, lambda*, nu*) matching PyomoKKTExtractor.
    """
    mat = extractor.get_canonical_matrices()
    n_vars = mat["n_vars"]
    n_ineq = mat["n_ineq"]
    n_eq = mat["n_eq"]
    
    # Primal x*
    x_star = np.array([solver_results["primal_vars"][name] for name in mat["var_names"]], dtype=np.float64)
    
    # Dual lambda* (inequalities)
    lam_star = np.zeros(n_ineq, dtype=np.float64)
    ineq_names = mat["con_info"]["ineq_names"]
    
    for idx, name in enumerate(ineq_names):
        if "var_bound:" in name:
            # e.g., "var_bound: x[0] >= 0.0" or "var_bound: x[0] <= 10.0"
            parts = name.replace("var_bound:", "").strip().split()
            var_name = parts[0]
            op = parts[1]
            rc = solver_results["reduced_costs"].get(var_name, 0.0)
            if op == ">=":
                lam_star[idx] = max(0.0, rc)
            elif op == "<=":
                lam_star[idx] = max(0.0, -rc)
        else:
            # Standard constraint name: e.g. "c1 (upper)" or "c1 (lower)"
            con_base = name.split(" (")[0]
            dual = solver_results["dual_multipliers"].get(con_base, 0.0)
            if "(upper)" in name:
                lam_star[idx] = max(0.0, -dual)
            elif "(lower)" in name:
                lam_star[idx] = max(0.0, dual)
            else:
                lam_star[idx] = max(0.0, abs(dual))
                
    # Dual nu* (equalities)
    nu_star = np.zeros(n_eq, dtype=np.float64)
    eq_names = mat["con_info"]["eq_names"]
    for idx, name in enumerate(eq_names):
        dual = solver_results["dual_multipliers"].get(name, 0.0)
        nu_star[idx] = -dual
        
    return {
        "x_star": x_star,
        "lambda_star": lam_star,
        "nu_star": nu_star,
    }
