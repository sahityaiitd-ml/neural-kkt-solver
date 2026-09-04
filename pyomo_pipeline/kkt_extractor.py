"""
kkt_extractor.py
================
Parses arbitrary Pyomo optimization models, extracts canonical matrices/vectors,
and compiles the complete Karush-Kuhn-Tucker (KKT) algebraic system.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn


class PyomoKKTExtractor:
    """
    Extracts canonical representation and KKT optimality system from a Pyomo ConcreteModel.
    
    Standard Canonical Form:
        min_x  1/2 x^T Q x + c^T x + r
        s.t.   G x <= h    (Inequality constraints, dual multipliers: lambda >= 0)
               A x == b    (Equality constraints,   dual multipliers: nu in R)
               
    Variable bounds (lb <= x <= ub) are converted to explicit inequality rows in G x <= h:
        -x <= -lb   and   x <= ub
    """
    
    def __init__(self, model: pyo.ConcreteModel, include_bounds_in_G: bool = True):
        self.model = model
        self.include_bounds_in_G = include_bounds_in_G
        
        # 1. Index active variables
        self.var_objects = list(model.component_data_objects(pyo.Var, active=True))
        self.var_names = [v.name for v in self.var_objects]
        self.var_map = {id(v): idx for idx, v in enumerate(self.var_objects)}
        self.n_vars = len(self.var_objects)
        
        # Variable bounds
        self.lb = np.array([float(v.lb) if v.lb is not None else -np.inf for v in self.var_objects])
        self.ub = np.array([float(v.ub) if v.ub is not None else np.inf for v in self.var_objects])
        
        # 2. Extract Objective (Q, c, r)
        self.Q, self.c, self.r, self.sense = self._extract_objective()
        
        # 3. Extract Constraints (G, h, A, b)
        self.G, self.h, self.A, self.b, self.con_info = self._extract_constraints()
        
        self.n_ineq = len(self.h)
        self.n_eq = len(self.b)

    def _extract_objective(self):
        obj_objs = list(self.model.component_data_objects(pyo.Objective, active=True))
        if len(obj_objs) == 0:
            raise ValueError("Model does not contain an active Objective component.")
        if len(obj_objs) > 1:
            raise ValueError("Multiple active Objectives found. Only single-objective models are supported.")
            
        obj = obj_objs[0]
        repn = generate_standard_repn(obj.expr)
        
        c = np.zeros(self.n_vars, dtype=np.float64)
        Q = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        r = float(repn.constant) if repn.constant is not None else 0.0
        
        # Linear terms
        if repn.linear_vars:
            for v, coef in zip(repn.linear_vars, repn.linear_coefs):
                c[self.var_map[id(v)]] += float(coef)
                
        # Quadratic terms: 1/2 x^T Q x
        if repn.quadratic_vars:
            for (v1, v2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
                i, j = self.var_map[id(v1)], self.var_map[id(v2)]
                if i == j:
                    Q[i, i] += 2.0 * float(coef)
                else:
                    Q[i, j] += float(coef)
                    Q[j, i] += float(coef)
                    
        sense = "minimize" if obj.sense == pyo.minimize else "maximize"
        if obj.sense == pyo.maximize:
            # max f(x) <=> min -f(x)
            c = -c
            Q = -Q
            r = -r
            
        return Q, c, r, sense

    def _extract_constraints(self):
        G_rows, h_vals = [], []
        A_rows, b_vals = [], []
        con_info = {"ineq_names": [], "eq_names": []}
        
        constraints = list(self.model.component_data_objects(pyo.Constraint, active=True))
        
        for con in constraints:
            repn = generate_standard_repn(con.body)
            row = np.zeros(self.n_vars, dtype=np.float64)
            if repn.linear_vars:
                for v, coef in zip(repn.linear_vars, repn.linear_coefs):
                    row[self.var_map[id(v)]] += float(coef)
            const = float(repn.constant) if repn.constant is not None else 0.0
            
            # Case 1: Equality constraint (body == rhs)
            if con.equality:
                rhs = float(con.lower)
                A_rows.append(row)
                b_vals.append(rhs - const)
                con_info["eq_names"].append(con.name)
            else:
                # Case 2: Upper bound (body <= upper)
                if con.upper is not None:
                    G_rows.append(row)
                    h_vals.append(float(con.upper) - const)
                    con_info["ineq_names"].append(f"{con.name} (upper)")
                    
                # Case 3: Lower bound (body >= lower  <=>  -body <= -lower)
                if con.lower is not None:
                    G_rows.append(-row)
                    h_vals.append(const - float(con.lower))
                    con_info["ineq_names"].append(f"{con.name} (lower)")
                    
        # Add variable bounds into G x <= h if requested
        if self.include_bounds_in_G:
            for i, v in enumerate(self.var_objects):
                if v.lb is not None and not np.isneginf(v.lb):
                    # x_i >= lb  <=>  -x_i <= -lb
                    row = np.zeros(self.n_vars, dtype=np.float64)
                    row[i] = -1.0
                    G_rows.append(row)
                    h_vals.append(-float(v.lb))
                    con_info["ineq_names"].append(f"var_bound: {v.name} >= {v.lb}")
                    
                if v.ub is not None and not np.isposinf(v.ub):
                    # x_i <= ub
                    row = np.zeros(self.n_vars, dtype=np.float64)
                    row[i] = 1.0
                    G_rows.append(row)
                    h_vals.append(float(v.ub))
                    con_info["ineq_names"].append(f"var_bound: {v.name} <= {v.ub}")
                    
        G = np.array(G_rows, dtype=np.float64) if len(G_rows) > 0 else np.zeros((0, self.n_vars))
        h = np.array(h_vals, dtype=np.float64) if len(h_vals) > 0 else np.zeros(0)
        A = np.array(A_rows, dtype=np.float64) if len(A_rows) > 0 else np.zeros((0, self.n_vars))
        b = np.array(b_vals, dtype=np.float64) if len(b_vals) > 0 else np.zeros(0)
        
        return G, h, A, b, con_info

    def get_canonical_matrices(self):
        """Returns the canonical matrices (Q, c, G, h, A, b) as a dictionary."""
        return {
            "Q": self.Q,
            "c": self.c,
            "r": self.r,
            "G": self.G,
            "h": self.h,
            "A": self.A,
            "b": self.b,
            "var_names": self.var_names,
            "con_info": self.con_info,
            "n_vars": self.n_vars,
            "n_ineq": self.n_ineq,
            "n_eq": self.n_eq,
            "sense": self.sense,
        }

    def print_kkt_system(self):
        """Prints a human-readable summary of the algebraic KKT system."""
        print("=" * 70)
        print(f"KKT Optimality System for Model: {self.model.name}")
        print("=" * 70)
        print(f"Dimensions: {self.n_vars} Primal Variables, {self.n_ineq} Inequality Multipliers (lambda), {self.n_eq} Equality Multipliers (nu)")
        print(f"Variables: {self.var_names}\n")
        
        print("1. STATIONARITY EQUATIONS (∇_x L = 0):")
        for i in range(self.n_vars):
            terms = []
            if np.any(self.Q[i] != 0):
                q_terms = [f"{self.Q[i, j]:+.3f}*{self.var_names[j]}" for j in range(self.n_vars) if self.Q[i, j] != 0]
                terms.append(" ".join(q_terms))
            if self.c[i] != 0:
                terms.append(f"{self.c[i]:+.3f}")
            if self.n_ineq > 0 and np.any(self.G[:, i] != 0):
                g_terms = [f"{self.G[k, i]:+.3f}*λ_{k+1}" for k in range(self.n_ineq) if self.G[k, i] != 0]
                terms.append(" ".join(g_terms))
            if self.n_eq > 0 and np.any(self.A[:, i] != 0):
                a_terms = [f"{self.A[j, i]:+.3f}*ν_{j+1}" for j in range(self.n_eq) if self.A[j, i] != 0]
                terms.append(" ".join(a_terms))
                
            eq_str = " ".join(terms) if terms else "0.0"
            print(f"   ∂L/∂{self.var_names[i]}:  {eq_str}  =  0")
            
        print("\n2. PRIMAL FEASIBILITY:")
        print(f"   Inequalities (G x <= h): {self.n_ineq} constraints")
        for k in range(min(5, self.n_ineq)):
            row_str = " ".join([f"{self.G[k, j]:+.3f}*{self.var_names[j]}" for j in range(self.n_vars) if self.G[k, j] != 0])
            print(f"   [λ_{k+1}]  {row_str}  <=  {self.h[k]:.3f}   ({self.con_info['ineq_names'][k]})")
        if self.n_ineq > 5:
            print(f"   ... ({self.n_ineq - 5} more inequality constraints)")
            
        if self.n_eq > 0:
            print(f"   Equalities (A x == b): {self.n_eq} constraints")
            for j in range(self.n_eq):
                row_str = " ".join([f"{self.A[j, k]:+.3f}*{self.var_names[k]}" for k in range(self.n_vars) if self.A[j, k] != 0])
                print(f"   [ν_{j+1}]  {row_str}  ==  {self.b[j]:.3f}   ({self.con_info['eq_names'][j]})")
                
        print("\n3. DUAL FEASIBILITY:")
        print(f"   λ_k >= 0, for k = 1, ..., {self.n_ineq}")
        print(f"   ν_j in R, for j = 1, ..., {self.n_eq}")
        
        print("\n4. COMPLEMENTARY SLACKNESS:")
        print(f"   λ_k * ( (G x)_k - h_k ) = 0, for k = 1, ..., {self.n_ineq}")
        print("=" * 70)
