"""
file_reader.py
==============
High-performance standalone reader for industry-standard optimization files:
  - .mps and .mps.gz (Mathematical Programming System format)
  - .lp and .lp.gz (Algebraic CPLEX LP format)

Key Features:
-------------
1. 100% Pure Python + NumPy -- runs on Mac, Windows, Linux with zero C++ compiler needed.
2. Flat memory buffers -- avoids slow nested dictionaries by indexing variables and rows to integer IDs immediately.
3. Automatically decompresses .gz files on the fly.
4. Outputs the unified KKTSystem (c, G, h) ready for KINN neural solvers.
"""

import os
import re
import gzip
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core.kkt_system import KKTSystem
from ..core.canonicalizer import build_canonical_kkt_system


def _open_file(filepath: str):
    """Opens plain text or .gz compressed files seamlessly."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")


# ==============================================================================
# 1. FAST MPS PARSER
# ==============================================================================

def _parse_mps_highs(filepath: str) -> KKTSystem:
    """Uses HiGHS C++ reader for ultra-fast, 100% compliant MPS/MPS.GZ parsing."""
    import highspy
    import scipy.sparse as sp
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    st = h.readModel(filepath)
    if st != highspy.HighsStatus.kOk:
        raise ValueError(f"Highs could not read {filepath}: {st}")
    
    # Relax integrality for continuous KKT optimization
    num_cols = h.getNumCol()
    cols = np.arange(num_cols, dtype=np.int32)
    types = np.zeros(num_cols, dtype=np.uint8)
    h.changeColsIntegrality(num_cols, cols, types)
    
    lp = h.getLp()
    model_name = os.path.basename(filepath).split(".")[0]
    n_vars = lp.num_col_
    n_rows = lp.num_row_
    
    col_cost = np.array(lp.col_cost_, dtype=np.float64, copy=True)
    sense = "maximize" if lp.sense_ == highspy.ObjSense.kMaximize else "minimize"
    
    var_names = list(lp.col_names_) if lp.col_names_ else [f"x_{j}" for j in range(n_vars)]
    row_names = list(lp.row_names_) if lp.row_names_ else [f"row_{i}" for i in range(n_rows)]
    
    row_lower = np.array(lp.row_lower_, dtype=np.float64)
    row_upper = np.array(lp.row_upper_, dtype=np.float64)
    col_lower = np.array(lp.col_lower_, dtype=np.float64)
    col_upper = np.array(lp.col_upper_, dtype=np.float64)
    
    a_mat = lp.a_matrix_
    A_csc = sp.csc_matrix((a_mat.value_, a_mat.index_, a_mat.start_), shape=(n_rows, n_vars))
    A_csr = A_csc.tocsr()
    
    row_coefs = []
    for r in range(n_rows):
        row_slice = A_csr[r]
        row_dict = {int(c): float(v) for c, v in zip(row_slice.indices, row_slice.data)}
        row_coefs.append(row_dict)
        
    return build_canonical_kkt_system(
        name=model_name,
        n_vars=n_vars,
        var_names=var_names,
        col_cost=col_cost,
        sense=sense,
        row_names=row_names,
        row_lower=row_lower,
        row_upper=row_upper,
        row_coefs=row_coefs,
        col_lower=col_lower,
        col_upper=col_upper,
        offset=lp.offset_
    )


def parse_mps(filepath: str) -> KKTSystem:
    """
    Parses an MPS or MPS.GZ file into a standardized KKTSystem.
    Leverages HiGHS C++ parser if available, otherwise falls back to pure Python.
    """
    try:
        return _parse_mps_highs(filepath)
    except Exception:
        pass

    model_name = os.path.basename(filepath).split(".")[0]
    
    # 1. Tracking variables and rows
    var_names: List[str] = []
    var_index: Dict[str, int] = {}
    
    row_names: List[str] = []
    row_index: Dict[str, int] = {}
    row_types: List[str] = []
    obj_row_name: Optional[str] = None
    obj_row_idx: int = -1
    
    # Coefficients stored per row as {col_idx: value}
    row_coefs: List[Dict[int, float]] = []
    
    # Right-hand sides and ranges
    rhs_dict: Dict[str, float] = {}
    ranges_dict: Dict[str, float] = {}
    
    # Variable bounds
    bounds_lo: Dict[int, float] = {}
    bounds_up: Dict[int, float] = {}
    
    section = None
    in_integer_block = False

    with _open_file(filepath) as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line.strip() or line.startswith("*"):
                continue  # skip empty lines and comments

            # In standard MPS, section headers start in column 1 (no leading space)
            if not line[0].isspace():
                tokens = line.split()
                header = tokens[0].upper()
                if header == "NAME":
                    if len(tokens) > 1:
                        model_name = tokens[1]
                    section = None
                elif header in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
                    section = header
                elif header == "ENDATA":
                    break
                else:
                    section = None
                continue

            tokens = line.split()
            if not tokens:
                continue

            # --- ROWS SECTION ---
            if section == "ROWS":
                rtype, rname = tokens[0].upper(), tokens[1]
                if rname not in row_index:
                    ridx = len(row_names)
                    row_index[rname] = ridx
                    row_names.append(rname)
                    row_types.append(rtype)
                    row_coefs.append({})
                    # The first 'N' row is traditionally the objective function
                    if rtype == "N" and obj_row_name is None:
                        obj_row_name = rname
                        obj_row_idx = ridx

            # --- COLUMNS SECTION ---
            elif section == "COLUMNS":
                # Check for integer markers
                if len(tokens) >= 3 and "'MARKER'" in tokens[1].upper():
                    marker_type = tokens[2].upper() if len(tokens) > 2 else tokens[-1].upper()
                    if "INTORG" in marker_type:
                        in_integer_block = True
                    elif "INTEND" in marker_type:
                        in_integer_block = False
                    continue

                cname = tokens[0]
                if cname not in var_index:
                    cidx = len(var_names)
                    var_index[cname] = cidx
                    var_names.append(cname)
                else:
                    cidx = var_index[cname]

                # Pairs of (row_name, coefficient_value)
                rest = tokens[1:]
                for k in range(0, len(rest) - 1, 2):
                    rname = rest[k]
                    val = float(rest[k + 1])
                    if rname in row_index:
                        ridx = row_index[rname]
                        row_coefs[ridx][cidx] = row_coefs[ridx].get(cidx, 0.0) + val

            # --- RHS SECTION ---
            elif section == "RHS":
                # rest tokens come in (row_name, value) pairs
                rest = tokens[1:]
                for k in range(0, len(rest) - 1, 2):
                    rhs_dict[rest[k]] = float(rest[k + 1])

            # --- RANGES SECTION ---
            elif section == "RANGES":
                rest = tokens[1:]
                for k in range(0, len(rest) - 1, 2):
                    ranges_dict[rest[k]] = float(rest[k + 1])

            # --- BOUNDS SECTION ---
            elif section == "BOUNDS":
                btype = tokens[0].upper()
                cname = tokens[2]
                val = float(tokens[3]) if len(tokens) > 3 else None
                if cname in var_index:
                    cidx = var_index[cname]
                    if btype == "UP":
                        bounds_up[cidx] = val
                        if val < 0 and cidx not in bounds_lo:
                            bounds_lo[cidx] = -np.inf
                    elif btype == "LO":
                        bounds_lo[cidx] = val
                    elif btype == "FX":
                        bounds_lo[cidx] = val
                        bounds_up[cidx] = val
                    elif btype == "FR":
                        bounds_lo[cidx] = -np.inf
                        bounds_up[cidx] = np.inf
                    elif btype == "MI":
                        bounds_lo[cidx] = -np.inf
                    elif btype == "PL":
                        bounds_up[cidx] = np.inf
                    elif btype == "BV":
                        bounds_lo[cidx] = 0.0
                        bounds_up[cidx] = 1.0

    n_vars = len(var_names)
    
    # 2. Extract cost vector c (from the objective row)
    col_cost = np.zeros(n_vars, dtype=np.float64)
    if obj_row_idx >= 0:
        for cidx, val in row_coefs[obj_row_idx].items():
            col_cost[cidx] = val
    offset = -rhs_dict.get(obj_row_name, 0.0) if obj_row_name else 0.0

    # 3. Build structural row bounds (row_lower, row_upper)
    filtered_row_names = []
    filtered_row_lower = []
    filtered_row_upper = []
    filtered_row_coefs = []

    for rname, rtype, coefs in zip(row_names, row_types, row_coefs):
        if rtype == "N":
            continue  # skip free/objective rows

        b = rhs_dict.get(rname, 0.0)
        if rtype == "L":
            lo, up = -np.inf, b
        elif rtype == "G":
            lo, up = b, np.inf
        elif rtype == "E":
            lo, up = b, b
        else:
            lo, up = -np.inf, np.inf

        # Apply range modifier if present
        if rname in ranges_dict:
            rg = ranges_dict[rname]
            if rtype == "L":
                lo, up = up - abs(rg), up
            elif rtype == "G":
                lo, up = lo, lo + abs(rg)
            elif rtype == "E":
                lo, up = (b, b + rg) if rg >= 0 else (b + rg, b)

        filtered_row_names.append(rname)
        filtered_row_lower.append(lo)
        filtered_row_upper.append(up)
        filtered_row_coefs.append(coefs)

    # 4. Variable bounds (default in MPS is [0, +inf) for unmentioned variables)
    col_lower = np.zeros(n_vars, dtype=np.float64)
    col_upper = np.full(n_vars, np.inf, dtype=np.float64)
    for cidx in range(n_vars):
        if cidx in bounds_lo:
            col_lower[cidx] = bounds_lo[cidx]
        if cidx in bounds_up:
            col_upper[cidx] = bounds_up[cidx]

    # 5. Compile into standardized KKTSystem
    return build_canonical_kkt_system(
        name=model_name,
        n_vars=n_vars,
        var_names=var_names,
        col_cost=col_cost,
        sense="minimize",
        row_names=filtered_row_names,
        row_lower=np.array(filtered_row_lower, dtype=np.float64),
        row_upper=np.array(filtered_row_upper, dtype=np.float64),
        row_coefs=filtered_row_coefs,
        col_lower=col_lower,
        col_upper=col_upper,
        offset=offset
    )


# ==============================================================================
# 2. FAST LP PARSER
# ==============================================================================

_LP_TOKEN_RE = re.compile(r"""
    (?P<num>   [-+]?\d+\.?\d*(?:[eE][-+]?\d+)?  ) |
    (?P<relop> <=|>=|=<|=>|<|>|=                ) |
    (?P<colon> :                                 ) |
    (?P<sign>  [-+]                              ) |
    (?P<ident> [A-Za-z_][A-Za-z0-9_.\[\]]*        )
""", re.VERBOSE)

_LP_KEYWORDS = {
    "max": "obj_max", "maximize": "obj_max", "maximise": "obj_max",
    "min": "obj_min", "minimize": "obj_min", "minimise": "obj_min",
    "subject": "cons", "st": "cons", "s.t.": "cons", "such": "cons",
    "bounds": "bounds", "bound": "bounds",
    "bin": "bin", "binary": "bin", "binaries": "bin",
    "gen": "gen", "general": "gen", "generals": "gen", "integer": "gen", "integers": "gen",
    "end": "end"
}


def parse_lp(filepath: str) -> KKTSystem:
    """
    Parses an algebraic LP or LP.GZ file into a standardized KKTSystem.
    
    Handles:
      - Maximize / Minimize objectives
      - Multi-line wrapped constraint expressions
      - Ranged constraints (e.g. 5 <= 2*x1 + x2 <= 20)
      - Variable bounds and binary declarations
    """
    model_name = os.path.basename(filepath).split(".")[0]
    
    # Strip comments (\ starts a comment in CPLEX LP format)
    clean_lines = []
    with _open_file(filepath) as f:
        for line in f:
            clean_lines.append(line.split("\\", 1)[0])
    text = "".join(clean_lines)

    # Tokenize the text stream
    tokens = []
    for m in _LP_TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind == "relop":
            val = {"=<": "<=", "=>": ">="}.get(val, val)
        tokens.append((kind, val))

    n_tokens = len(tokens)
    i = 0

    def parse_single_number(idx: int) -> Tuple[float, int]:
        sign = 1.0
        if idx < n_tokens and tokens[idx][0] == "sign":
            sign = 1.0 if tokens[idx][1] == "+" else -1.0
            idx += 1
        if idx < n_tokens and tokens[idx][0] == "num":
            val = sign * float(tokens[idx][1])
            return val, idx + 1
        return 0.0, idx

    def parse_linear_terms(idx: int) -> Tuple[Dict[str, float], float, int]:
        coefs: Dict[str, float] = {}
        const = 0.0
        sign = 1.0
        pending_num = None

        while idx < n_tokens:
            kind, val = tokens[idx]
            if kind == "sign":
                if pending_num is not None:
                    break
                sign = 1.0 if val == "+" else -1.0
                idx += 1
            elif kind == "num":
                pending_num = float(val)
                idx += 1
            elif kind == "ident":
                low = val.lower()
                if low in _LP_KEYWORDS or low in ("to", "that"):
                    break
                c = sign * (pending_num if pending_num is not None else 1.0)
                coefs[val] = coefs.get(val, 0.0) + c
                pending_num, sign = None, 1.0
                idx += 1
            else:
                break
        if pending_num is not None:
            const += sign * pending_num
        return coefs, const, idx

    # 1. Find Objective
    sense = "minimize"
    while i < n_tokens:
        kind, val = tokens[i]
        low = val.lower() if kind == "ident" else None
        if low in ("max", "maximize", "maximise"):
            sense = "maximize"
            i += 1
            break
        elif low in ("min", "minimize", "minimise"):
            sense = "minimize"
            i += 1
            break
        i += 1

    # Skip optional "obj:" label
    if i + 1 < n_tokens and tokens[i][0] == "ident" and tokens[i + 1][0] == "colon":
        i += 2

    # Parse objective expression
    obj_coefs, obj_const, i = parse_linear_terms(i)

    # Skip until "subject to" / "s.t." / "st"
    while i < n_tokens:
        kind, val = tokens[i]
        low = val.lower() if kind == "ident" else None
        if low in ("subject", "st", "s.t.", "such"):
            i += 1
            if i < n_tokens and tokens[i][0] == "ident" and tokens[i][1].lower() in ("to", "that"):
                i += 1
            break
        i += 1

    # 2. Parse Constraints and Bounds
    all_vars: List[str] = list(obj_coefs.keys())
    seen_vars = set(all_vars)

    con_names: List[str] = []
    con_lower_list: List[float] = []
    con_upper_list: List[float] = []
    con_coefs_list: List[Dict[str, float]] = []

    bounds_lo: Dict[str, float] = {}
    bounds_up: Dict[str, float] = {}
    bin_vars: set = set()

    section = "cons"
    con_idx = 0

    while i < n_tokens:
        kind, val = tokens[i]
        low = val.lower() if kind == "ident" else None

        # Check for section change keyword
        if kind == "ident" and low in _LP_KEYWORDS:
            sec = _LP_KEYWORDS[low]
            i += 1
            if sec in ("bounds", "bin", "gen"):
                section = sec
                continue
            elif sec == "end":
                break
            else:
                continue

        if section == "cons":
            cname = f"con_{con_idx + 1}"
            if kind == "ident" and i + 1 < n_tokens and tokens[i + 1][0] == "colon":
                cname = val
                i += 2

            # Check if this constraint starts with a number (ranged constraint: lo <= expr <= up)
            is_ranged = tokens[i][0] in ("num", "sign") and i + 1 < n_tokens and tokens[i + 1][0] == "relop"
            if is_ranged:
                lo_val, i = parse_single_number(i)
                rel1 = tokens[i][1]; i += 1
                expr_c, expr_k, i = parse_linear_terms(i)
                rel2 = tokens[i][1]; i += 1
                up_val, i = parse_single_number(i)
                con_names.append(cname)
                con_lower_list.append(lo_val - expr_k)
                con_upper_list.append(up_val - expr_k)
                con_coefs_list.append(expr_c)
                for v in expr_c:
                    if v not in seen_vars:
                        seen_vars.add(v); all_vars.append(v)
            else:
                expr_c, expr_k, i = parse_linear_terms(i)
                if i >= n_tokens or tokens[i][0] != "relop":
                    break
                rel = tokens[i][1]; i += 1
                rhs_val, i = parse_single_number(i)
                rhs = rhs_val - expr_k
                con_names.append(cname)
                con_coefs_list.append(expr_c)
                if rel == "<=":
                    con_lower_list.append(-np.inf)
                    con_upper_list.append(rhs)
                elif rel == ">=":
                    con_lower_list.append(rhs)
                    con_upper_list.append(np.inf)
                else:  # "==" or "="
                    con_lower_list.append(rhs)
                    con_upper_list.append(rhs)
                for v in expr_c:
                    if v not in seen_vars:
                        seen_vars.add(v); all_vars.append(v)
            con_idx += 1

        elif section == "bounds":
            if kind == "ident":
                v = val; i += 1
                if i < n_tokens and tokens[i][0] == "ident" and tokens[i][1].lower() == "free":
                    bounds_lo[v] = -np.inf
                    bounds_up[v] = np.inf
                    i += 1
                elif i < n_tokens and tokens[i][0] == "relop":
                    rel = tokens[i][1]; i += 1
                    rhs, i = parse_single_number(i)
                    if rel == "<=":
                        bounds_up[v] = rhs
                    elif rel == ">=":
                        bounds_lo[v] = rhs
                    else:
                        bounds_lo[v] = rhs; bounds_up[v] = rhs
            else:
                i += 1

        elif section == "bin":
            if kind == "ident":
                bin_vars.add(val)
                if val not in seen_vars:
                    seen_vars.add(val); all_vars.append(val)
            i += 1
        else:
            i += 1

    # Map variable names to integer indices
    var_index = {v: idx for idx, v in enumerate(all_vars)}
    n_vars = len(all_vars)

    col_cost = np.zeros(n_vars, dtype=np.float64)
    for v, c in obj_coefs.items():
        col_cost[var_index[v]] = c

    # Convert sparse dicts to integer-indexed dicts
    row_coefs = []
    for c_dict in con_coefs_list:
        row_coefs.append({var_index[v]: val for v, val in c_dict.items()})

    col_lower = np.zeros(n_vars, dtype=np.float64)
    col_upper = np.full(n_vars, np.inf, dtype=np.float64)
    for v, lo in bounds_lo.items():
        if v in var_index:
            col_lower[var_index[v]] = lo
    for v, up in bounds_up.items():
        if v in var_index:
            col_upper[var_index[v]] = up
    for v in bin_vars:
        if v in var_index and v not in bounds_up:
            col_lower[var_index[v]] = 0.0
            col_upper[var_index[v]] = 1.0

    return build_canonical_kkt_system(
        name=model_name,
        n_vars=n_vars,
        var_names=all_vars,
        col_cost=col_cost,
        sense=sense,
        row_names=con_names,
        row_lower=np.array(con_lower_list, dtype=np.float64),
        row_upper=np.array(con_upper_list, dtype=np.float64),
        row_coefs=row_coefs,
        col_lower=col_lower,
        col_upper=col_upper,
        offset=obj_const
    )


def read_file(filepath: str) -> KKTSystem:
    """
    Auto-detects whether the file is MPS or LP format and loads it.
    """
    lower = filepath.lower()
    if ".mps" in lower:
        return parse_mps(filepath)
    elif ".lp" in lower:
        return parse_lp(filepath)
    else:
        raise ValueError(f"Unrecognized file format for '{filepath}'. Expected .mps, .lp, or .gz versions.")
