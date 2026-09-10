"""
KKT_Standalone_Reader
=====================
A high-performance, solver-agnostic problem reader and KKT canonicalization engine
built specifically for KKT-Informed Neural Networks (KINNs).

Key Capabilities:
-----------------
1. Unified interface: load_problem(...) accepts:
   - Industry files (.mps, .lp, .mps.gz, .lp.gz)
   - Pyomo models (pyo.ConcreteModel)
   - Structured data (.json files or Python dicts)
   - Direct NumPy arrays (c, G, h)
2. Always outputs a standardized KKTSystem containing:
   - c : cost vector normalized to minimize
   - G : inequality constraint matrix (G x <= h)
   - h : RHS boundary vector
   - sparse_coo : (rows, cols, vals) for Graph Neural Networks (GNNs)
   - to_torch() : direct helper to create PyTorch tensors

Usage Example:
--------------
    from KKT_Standalone_Reader import load_problem

    # 1. From an MPS file:
    problem = load_problem("benchmarks/input/afiro.mps")
    print(problem.summary())

    # 2. Convert to PyTorch tensors for KINN training:
    tensors = problem.to_torch()
    c, G, h = tensors["c"], tensors["G"], tensors["h"]
"""

from typing import Any, Optional, Dict, Union
import os

from .core.kkt_system import KKTSystem
from .core.canonicalizer import build_canonical_kkt_system
from .readers.file_reader import read_file, parse_mps, parse_lp
from .readers.pyomo_reader import read_pyomo
from .readers.json_reader import read_json
from .readers.matrix_reader import read_matrices


def load_problem(source: Any, format: Optional[str] = None, **kwargs) -> KKTSystem:
    """
    Universal entry point to load an optimization problem from any format.
    
    Parameters:
    -----------
    source : Any
        - str: path to .mps, .lp, .mps.gz, .lp.gz, or .json file
        - pyo.ConcreteModel: Pyomo optimization model
        - dict: Python dictionary with JSON problem schema
    format : Optional[str]
        Optional format override ('mps', 'lp', 'json', 'pyomo', 'matrices').
        If None, automatically detected from the source type and extension.
    """
    # 1. Pyomo Model
    if hasattr(source, "component_data_objects") or type(source).__name__ == "ConcreteModel":
        return read_pyomo(source)

    # 2. Dictionary (JSON schema)
    if isinstance(source, dict):
        return read_json(source)

    # 3. File path (string)
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Optimization problem file not found: {source}")

        fmt = format.lower() if format else None
        ext = os.path.splitext(source.lower().rstrip(".gz"))[1]

        if fmt == "json" or ext == ".json":
            return read_json(source)
        elif fmt in ("mps", "lp") or ext in (".mps", ".lp"):
            return read_file(source)
        else:
            # Try file reader by default
            return read_file(source)

    raise TypeError(
        f"Unsupported source type '{type(source)}'. Supported types: "
        f"file path string (.mps, .lp, .json), Pyomo ConcreteModel, or dictionary."
    )


__all__ = [
    "load_problem",
    "KKTSystem",
    "build_canonical_kkt_system",
    "read_file",
    "parse_mps",
    "parse_lp",
    "read_pyomo",
    "read_json",
    "read_matrices"
]
