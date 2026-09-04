# -*- coding: utf-8 -*-
"""B.Tech. Project: Neural Network Based KKT Solver
## Week 2: Batch-Building a (c, G, h, x*, lambda*) Training Dataset
##         from Standard Benchmark Libraries (Netlib, MIPLIB, ...)

**Students Name:** Sahitya Rankawat, Manav Gupta, Jeet Anand
**Supervisor:** Prof. Kartikey Sharma
**Department:** Mechanical Engineering

---

### Goal

Point this at a folder of real benchmark files (any mix of `.mps` / `.lp`,
from Netlib, MIPLIB, or anywhere else) and get back one `.npz` file per
problem containing everything the NN needs:

    c, G, h                -- the LP in canonical `min c^T x s.t. G x <= h` form
    x_star, lambda_star    -- the ground-truth primal/dual solution from HiGHS
    var_names, con_names   -- for debugging / interpretability

plus a `summary.json` logging solve status and exact KKT residuals per
problem, so you can immediately see which examples are clean, high-quality
training/verification labels.

### What changed vs. Week 1

Week 1's `extract_kkt_system()` only handled `<=` constraints and variable
LOWER bounds (fine for the hand-built 2-var toy LP, but real benchmark
files routinely use `>=`, `==`, and variable UPPER bounds too). Getting
the dual/reduced-cost sign convention right for each of these was the
main technical piece here — verified empirically below, not guessed.

### MIPLIB-specific handling (new)

- **Gzip-compressed files:** MIPLIB distributes instances as `.mps.gz`.
  HiGHS's reader decompresses these directly, so `load_benchmark_as_pyomo`
  needs no changes — the batch builder's file-search patterns now just
  include `*.mps.gz` / `*.lp.gz` alongside the plain versions.
- **`.solu` cross-check (optional):** MIPLIB publishes a `.solu` file with
  each instance's known optimal/best-known objective value. If you pass
  `solu_file=...` to `build_kkt_dataset`, every solved problem's HiGHS
  objective is compared against the published value and logged in
  `summary.json` — an extra sanity check independent of the KKT residual
  check. (Since integer variables are relaxed, an exact match isn't
  expected — the LP relaxation's objective should be *at least as good
  as* the integer-optimal one for minimize problems, i.e. its own kind of
  plausibility check, not a pass/fail test.)
"""

import os
import glob
import json
import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import highspy

print("All libraries imported successfully!")

"""---
## Part 1: Loading any benchmark file into Pyomo

(Same as Week 2a — reuses HiGHS's own robust MPS/LP parser via `highspy`,
then rebuilds an equivalent Pyomo `ConcreteModel`.)
"""


def load_benchmark_as_pyomo(filepath, relax_integers=True, verbose=True):
    """Load a .mps / .lp benchmark file into a Pyomo ConcreteModel."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(filepath)
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"HiGHS failed to read file '{filepath}' (status={status})")

    lp = h.getLp()
    n_vars, n_rows = lp.num_col_, lp.num_row_
    col_names = list(lp.col_names_) if lp.col_names_ else [f"x{i+1}" for i in range(n_vars)]
    col_lower = np.array(lp.col_lower_, dtype=float)
    col_upper = np.array(lp.col_upper_, dtype=float)
    col_cost = np.array(lp.col_cost_, dtype=float)
    integrality = list(lp.integrality_) if lp.integrality_ else []
    n_integer = sum(1 for it in integrality if int(it) != 0)

    if n_integer > 0 and not relax_integers:
        raise ValueError(
            f"Model has {n_integer} integer/binary variables. "
            "Set relax_integers=True to solve the LP relaxation."
        )

    model = pyo.ConcreteModel(name=lp.model_name_ or "benchmark_LP")
    model.VARSET = pyo.Set(initialize=range(n_vars))

    def _bounds_rule(m, i):
        lb = None if np.isneginf(col_lower[i]) else float(col_lower[i])
        ub = None if np.isposinf(col_upper[i]) else float(col_upper[i])
        return (lb, ub)

    model.x = pyo.Var(model.VARSET, domain=pyo.Reals, bounds=_bounds_rule)
    model.var_original_names = {i: col_names[i] for i in range(n_vars)}

    sense = pyo.minimize if lp.sense_ == highspy.ObjSense.kMinimize else pyo.maximize
    model.obj = pyo.Objective(
        expr=sum(float(col_cost[i]) * model.x[i] for i in range(n_vars)) + float(lp.offset_),
        sense=sense,
    )

    am = lp.a_matrix_
    row_coefs = [dict() for _ in range(n_rows)]
    for col in range(n_vars):
        for k in range(am.start_[col], am.start_[col + 1]):
            row_coefs[am.index_[k]][col] = float(am.value_[k])

    model.CONSET = pyo.Set(initialize=range(n_rows))
    model.con_original_names = {
        r: (lp.row_names_[r] if lp.row_names_ else f"row{r+1}") for r in range(n_rows)
    }

    def _con_rule(m, r):
        # Some real-world files (e.g. from certain LP writers) contain
        # degenerate empty rows -- no variable coefficients at all, just a
        # trivial "0 <= 0" bookkeeping row. Pyomo can't represent a
        # constant-only expression as a normal Constraint (it collapses to
        # a plain Python bool), so those rows are skipped here. They carry
        # no information for the KKT system anyway, since d(0)/dx = 0 for
        # every variable -- they contribute nothing to stationarity.
        if not row_coefs[r]:
            return pyo.Constraint.Skip
        expr = sum(coef * m.x[i] for i, coef in row_coefs[r].items())
        lo = lp.row_lower_[r]
        up = lp.row_upper_[r]
        lo = None if np.isneginf(lo) else float(lo)
        up = None if np.isposinf(up) else float(up)
        if lo is not None and up is not None and lo == up:
            return expr == lo
        elif up is not None and lo is None:
            return expr <= up
        elif lo is not None and up is None:
            return expr >= lo
        else:
            return (lo, expr, up)

    model.cons = pyo.Constraint(model.CONSET, rule=_con_rule)



    info = {"name": model.name, "n_vars": n_vars, "n_constraints": n_rows,
            "n_integer_relaxed": n_integer}
    if verbose:
        print(f"Loaded '{filepath}': {n_vars} vars, {n_rows} constraints"
              + (f", {n_integer} integer relaxed" if n_integer else ""))
    return model, info


"""---
## Part 2: Extended KKT extraction (now covers `>=`, `==`, and upper bounds too)
"""


def extract_kkt_system(model):
    """Convert model into canonical c, G, h with G x <= h (all constraint
    types + both variable bound directions folded in as extra rows)."""
    vars_list = list(model.component_data_objects(pyo.Var, active=True))
    var_names = [v.name for v in vars_list]
    var_map = {id(v): i for i, v in enumerate(vars_list)}
    n_vars = len(vars_list)

    obj = next(model.component_data_objects(pyo.Objective, active=True))
    repn_obj = generate_standard_repn(obj.expr)
    c = np.zeros(n_vars)
    if repn_obj.linear_vars:
        for v, coef in zip(repn_obj.linear_vars, repn_obj.linear_coefs):
            c[var_map[id(v)]] += float(coef)
    if obj.sense == pyo.maximize:
        c = -c

    G_rows, h_vals, con_names, con_tags = [], [], [], []
    for con in model.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(con.body)
        row = np.zeros(n_vars)
        if repn.linear_vars:
            for v, coef in zip(repn.linear_vars, repn.linear_coefs):
                row[var_map[id(v)]] += float(coef)
        const = float(repn.constant) if repn.constant is not None else 0.0
        if con.upper is not None:
            G_rows.append(row); h_vals.append(float(con.upper) - const)
            con_names.append(f"{con.name} (upper)"); con_tags.append(("con_upper", con))
        if con.lower is not None:
            G_rows.append(-row); h_vals.append(const - float(con.lower))
            con_names.append(f"{con.name} (lower)"); con_tags.append(("con_lower", con))

    for i, v in enumerate(vars_list):
        if v.lb is not None and not np.isneginf(v.lb):
            row = np.zeros(n_vars); row[i] = -1.0
            G_rows.append(row); h_vals.append(-float(v.lb))
            con_names.append(f"bound: {v.name} >= {v.lb}"); con_tags.append(("bound_lower", v))
        if v.ub is not None and not np.isposinf(v.ub):
            row = np.zeros(n_vars); row[i] = 1.0
            G_rows.append(row); h_vals.append(float(v.ub))
            con_names.append(f"bound: {v.name} <= {v.ub}"); con_tags.append(("bound_upper", v))

    G = np.array(G_rows, dtype=np.float64) if G_rows else np.zeros((0, n_vars))
    h = np.array(h_vals, dtype=np.float64) if h_vals else np.zeros(0)
    return {"var_names": var_names, "con_names": con_names, "c": c, "G": G, "h": h,
            "con_tags": con_tags}


def extract_lambda_star(model, kkt_data):
    """
    Build lambda* (one entry per row of G/h) from the solved model's
    dual/rc suffixes. Sign conventions below were verified empirically
    (not assumed) against hand-derived KKT stationarity on test problems
    covering every constraint type -- see the project notes for the
    derivation. They also flip for MAXIMIZE-sense models, since
    extract_kkt_system() always reports c in an equivalent MINIMIZE form
    but HiGHS/Pyomo reports duals relative to the model's original sense.

        MINIMIZE:  <=  -> lambda = max(0, -dual)
                   >=  -> lambda = max(0,  dual)
                   ==  -> both rows use the <=/>= formulas above (mu=-dual for upper row)
                   lower bound -> lambda = max(0,  rc)
                   upper bound -> lambda = max(0, -rc)
        MAXIMIZE:  same formulas with dual/rc negated first.
    """
    obj = next(model.component_data_objects(pyo.Objective, active=True))
    sign = -1.0 if obj.sense == pyo.maximize else 1.0

    lam = np.zeros(len(kkt_data["con_tags"]))
    for i, (tag, comp) in enumerate(kkt_data["con_tags"]):
        if tag == "con_upper":
            lam[i] = max(0.0, -sign * float(model.dual[comp]))
        elif tag == "con_lower":
            lam[i] = max(0.0, sign * float(model.dual[comp]))
        elif tag == "bound_lower":
            lam[i] = max(0.0, sign * float(model.rc[comp]))
        elif tag == "bound_upper":
            lam[i] = max(0.0, -sign * float(model.rc[comp]))
    return lam


def kkt_residuals(kkt_data, x_star, lambda_star):
    """The same 4 residual checks from Week 1, generalized to any G/h."""
    c, G, h = kkt_data["c"], kkt_data["G"], kkt_data["h"]
    stationarity = c + G.T @ lambda_star
    slack = G @ x_star - h
    return {
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "primal_infeas_norm": float(np.linalg.norm(np.maximum(0.0, slack))),
        "dual_infeas_norm": float(np.linalg.norm(np.maximum(0.0, -lambda_star))),
        "comp_slackness_norm": float(np.linalg.norm(lambda_star * slack)),
    }


"""---
## Part 3: Batch dataset builder

Point this at a directory of `.mps`/`.lp` files -- from Netlib, MIPLIB, or
anywhere else -- and it writes one `.npz` training/verification example
per solvable problem, plus a `summary.json` you can scan for solve
failures or numerically suspicious instances before training on them.
"""


def parse_solu_file(solu_path):
    """
    Parse a MIPLIB-style .solu file, mapping instance_name -> known
    objective value (or a status string for infeasible/unbounded cases).

    Format (one entry per line, whitespace-separated):
        =opt=  <instance_name>  <value>   -- proven optimal
        =best= <instance_name>  <value>   -- best known, not proven optimal
        =inf=  <instance_name>            -- infeasible
        =unbd= <instance_name>            -- unbounded

    Unrecognized/blank lines are skipped rather than raising, since these
    files can vary slightly across MIPLIB releases.
    """
    known = {}
    with open(solu_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            tag, instance = parts[0], parts[1]
            if tag in ("=opt=", "=best="):
                if len(parts) >= 3:
                    try:
                        known[instance] = {"status": tag.strip("="), "objective": float(parts[2])}
                    except ValueError:
                        continue
            elif tag in ("=inf=", "=unbd="):
                known[instance] = {"status": tag.strip("="), "objective": None}
    return known


def _instance_base_name(filepath):
    """Strip directory, extension(s), and .gz suffix to get the MIPLIB
    instance name used to look it up in a .solu file, e.g.
    'eil33-2.mps.gz' -> 'eil33-2'."""
    base = os.path.basename(filepath)
    if base.endswith(".gz"):
        base = base[:-3]
    base = os.path.splitext(base)[0]
    return base


def build_kkt_dataset(input_dir, output_dir,
                       patterns=("*.mps", "*.mps.gz", "*.lp", "*.lp.gz"),
                       relax_integers=True, solver_name="appsi_highs",
                       solu_file=None):
    """
    Parameters
    ----------
    solu_file : str or None
        Optional path to a MIPLIB .solu file. When given, each solved
        problem's HiGHS objective is cross-checked against the known
        published value (matched by instance name) and logged in
        summary.json as 'solu_match' -- an extra sanity check independent
        of the KKT residuals.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(input_dir, pat))))
    files = sorted(set(files))  # a plain .mps and .mps.gz of the same instance both matching is fine; just dedupe exact paths

    known_solutions = parse_solu_file(solu_file) if solu_file else {}

    solver = pyo.SolverFactory(solver_name)
    summary = []

    for filepath in files:
        base = os.path.basename(filepath)
        name = base.replace(".", "_")  # avoid collisions between e.g. foo.mps and foo.lp
        instance_name = _instance_base_name(filepath)
        entry = {"file": base, "name": name, "instance": instance_name}
        try:
            model, info = load_benchmark_as_pyomo(filepath, relax_integers=relax_integers, verbose=False)
            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            results = solver.solve(model)
            term = str(results.solver.termination_condition)
            entry.update(termination=term, **info)

            if term != "optimal":
                entry["status"] = "skipped_not_optimal"
                summary.append(entry)
                print(f"[SKIP] {name}: termination={term}")
                continue

            x_star = np.array([pyo.value(model.x[i]) for i in model.VARSET])
            kkt_data = extract_kkt_system(model)
            lambda_star = extract_lambda_star(model, kkt_data)
            res = kkt_residuals(kkt_data, x_star, lambda_star)

            np.savez(
                os.path.join(output_dir, f"{name}.npz"),
                c=kkt_data["c"], G=kkt_data["G"], h=kkt_data["h"],
                x_star=x_star, lambda_star=lambda_star,
                var_names=np.array(kkt_data["var_names"], dtype=object),
                con_names=np.array(kkt_data["con_names"], dtype=object),
            )

            entry["status"] = "ok"
            entry["objective"] = float(pyo.value(model.obj))
            entry.update(res)
            entry["total_kkt_error"] = sum(res.values())

            # Optional cross-check against MIPLIB's published solution value
            solu_note = ""
            if instance_name in known_solutions:
                ref = known_solutions[instance_name]
                entry["solu_status"] = ref["status"]
                if ref["objective"] is not None:
                    # Note: relaxing integers means our LP objective is expected
                    # to differ from MIPLIB's integer-optimal objective -- this
                    # is a sanity/plausibility check, not an exact-match test.
                    entry["solu_reference_objective"] = ref["objective"]
                    entry["solu_gap"] = entry["objective"] - ref["objective"]
                    solu_note = f", vs MIPLIB {ref['status']}={ref['objective']:.4g} (gap={entry['solu_gap']:+.4g})"
                else:
                    solu_note = f", MIPLIB marks this instance as {ref['status']}"

            summary.append(entry)
            print(f"[OK]   {name}: n_vars={info['n_vars']}, n_rows={kkt_data['G'].shape[0]}, "
                  f"total_kkt_error={entry['total_kkt_error']:.2e}{solu_note}")

        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            summary.append(entry)
            print(f"[ERROR] {name}: {e}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(1 for e in summary if e["status"] == "ok")
    print(f"\nDone. {n_ok}/{len(summary)} problems converted -> {output_dir}/")
    return summary


"""---
## Part 4: Demo -- run against real Netlib + MIPLIB files

`afiro.mps` (Netlib, pure LP) and `flugpl.mps` (MIPLIB, MIP -> LP
relaxation) are included alongside this script. `tiny_test.mps` /
`.lp` are the Week-1 hand-built 2-variable LP in both file formats, to
sanity-check the pipeline reproduces the known x1*=3.6, x2*=2.8 result.

**Folder layout:** all benchmark files (`.mps`/`.lp`/`.mps.gz`/`.lp.gz`)
must sit inside an `input/` folder next to this script. Output goes to a
separate `dataset/` folder -- keep these two folders distinct; don't put
benchmark files inside `dataset/`, and don't expect outputs in `input/`.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)

    demo_mps = """NAME          TINYLP
ROWS
 N  COST
 L  C1
 L  C2
COLUMNS
    X1        COST         -3.0   C1            2.0
    X1        C2            1.0
    X2        COST         -2.0   C1            1.0
    X2        C2            3.0
RHS
    RHS       C1           10.0   C2           12.0
BOUNDS
ENDATA
"""
    with open(os.path.join(INPUT_DIR, "tiny_test.mps"), "w") as f:
        f.write(demo_mps)

    demo_lp = """\\ Same problem, LP format, MAXIMIZE sense (tests the sign-convention fix)
Maximize
 obj: 3 x1 + 2 x2
Subject To
 c1: 2 x1 + x2 <= 10
 c2: x1 + 3 x2 <= 12
Bounds
 x1 >= 0
 x2 >= 0
End
"""
    with open(os.path.join(INPUT_DIR, "tiny_test.lp"), "w") as f:
        f.write(demo_lp)

    print(f"Looking for benchmark files in '{INPUT_DIR}/' ...")
    build_kkt_dataset(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

    # Quick sanity check: load one saved example back and confirm it's usable
    afiro_path = os.path.join(OUTPUT_DIR, "afiro_mps.npz")
    d = np.load(afiro_path, allow_pickle=True) if os.path.exists(afiro_path) else None
    if d is not None:
        print("\nExample loaded example (afiro): "
              f"c{d['c'].shape}, G{d['G'].shape}, h{d['h'].shape}, "
              f"x*{d['x_star'].shape}, lambda*{d['lambda_star'].shape}")