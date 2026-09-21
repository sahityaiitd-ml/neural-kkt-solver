"""
solver.py
=========
Two-Stage Hybrid Optimization Engine for KKT_Solver_Iteration_5.

Two-Stage Pipeline:
Stage 1: Neural Scout
  - Decoupled KINN architecture (SingleInstanceKINN)
  - Powell-Hestenes-Rockafellar (PHR) Augmented Lagrangian with row-scaled rho_i
  - Dual-aware objective pull (breaking profit-driven & demand-driven traps)
  - Adaptive preconditioning ("auto": Row L2 vs Ruiz)
  - Diameter-adaptive annealed objective pull
Stage 2: Closed-Form Active-Set Linear Snap
  - SVD pseudoinverse projection onto violated hyperplanes (G_viol^+)
  - Dual reduced-cost basis sparsity snapping
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.optim as optim

from .model import SingleInstanceKINN
from .preconditioning import preconditioned_system
from .loss import iteration_5_kkt_loss
from .snap import apply_active_set_linear_snap, apply_basis_sparsity_snapping


def solve_kkt_instance(
    kkt_system,
    max_epochs: int = 1000,
    lr: float = 0.015,
    alm_rho_init: float = 10.0,
    precond_method: str = "auto",
    apply_snap: bool = True,
    apply_sparsity: bool = True,
    sparsity_threshold: float = 0.05,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solves a KKT linear program instance using Iteration 5 Two-Stage Hybrid Solver.
    """
    t0 = time.perf_counter()

    # 1. Problem matrices in float64
    c_raw = np.array(kkt_system.c, dtype=np.float64)
    h_raw = np.array(kkt_system.h, dtype=np.float64)
    if hasattr(kkt_system.G, "toarray"):
        G_dense = kkt_system.G.toarray().astype(np.float64)
    else:
        G_dense = np.array(kkt_system.G, dtype=np.float64)

    n_vars = kkt_system.n_vars
    n_cons = kkt_system.n_constraints

    # 2. Adaptive Preconditioning
    precond = preconditioned_system(G_dense, h_raw, c_raw, method=precond_method)

    c_norm = precond.c_norm
    G_norm = precond.G_norm
    h_norm = precond.h_norm

    c_tensor = torch.tensor(c_norm, dtype=torch.float32)
    G_tensor = torch.tensor(G_norm, dtype=torch.float32)
    h_tensor = torch.tensor(h_norm, dtype=torch.float32)

    # 3. Row-scaled adaptive penalty parameter rho_i = rho_0 / ||G_i||_2
    row_norms = np.linalg.norm(G_norm, axis=1)
    row_norms = np.maximum(row_norms, 1e-4)
    rho_per_row = torch.tensor(alm_rho_init / row_norms, dtype=torch.float32)

    # 4. Detect cost direction: profit-driven (c < 0) vs demand-driven (c > 0)
    net_c_sign = float(np.sum(c_norm))
    is_positive_cost = net_c_sign > 0.01

    # 5. Span-adaptive annealing rate
    scale_factor = max(1.0, (precond.estimated_diameter / 2.0) ** 0.5)
    effective_gamma = 1.0 - (1.0 - 0.996) / scale_factor

    # 6. Initialize Decoupled Model and Optimizer
    torch.manual_seed(seed)
    model = SingleInstanceKINN(n_vars, n_cons, hidden_dim=64, dual_bias_init=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-5
    )

    if verbose:
        print(f"\n[STAGE 1: NEURAL SCOUT] Initializing Decoupled ALM KINN...")
        print(f"   Variables: {n_vars} | Cons: {n_cons} | Precond: {precond.method} | Span: {precond.estimated_diameter:.2e}")
        print(f"   Cost Orientation: {'Demand-Driven (c > 0)' if is_positive_cost else 'Profit-Driven (c < 0)'}")
        print(f"   {'Epoch':<8} | {'Total Loss':<12} | {'KKT Resid':<12} | {'Stationarity':<13} | {'ALM Penalty':<12} | {'Prim Viol':<10}")
        print("   " + "-" * 85)

    best_res = float("inf")
    best_x = np.zeros(n_vars)
    best_lam = np.zeros(n_cons)

    # Stage 1: Neural Scout Optimization
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        x_hat, lambda_hat = model()

        total_loss, metrics = iteration_5_kkt_loss(
            x_hat=x_hat,
            lambda_hat=lambda_hat,
            c_norm=c_tensor,
            G_norm=G_tensor,
            h_norm=h_tensor,
            rho_per_row=rho_per_row,
            epoch=epoch,
            effective_gamma=effective_gamma,
            is_positive_cost=is_positive_cost,
            w_obj_init=0.5,
            w_stat=1.0,
            w_gap=1.0,
            w_fb=2.0,
            w_alm=1.0,
            w_x_pos=5.0
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step(total_loss.item())

        kkt_res = metrics["kkt_residual"]
        if kkt_res < best_res:
            best_res = kkt_res
            best_x = x_hat.detach().cpu().numpy().copy()
            best_lam = lambda_hat.detach().cpu().numpy().copy()

        if verbose and (epoch in [1, 100, 250, 500, 750, 1000] or epoch == max_epochs):
            print(
                f"   {epoch:<8d} | {metrics['loss_total']:<12.4e} | {metrics['kkt_residual']:<12.4e} | "
                f"{metrics['loss_stat']:<13.4e} | {metrics['loss_alm']:<12.4e} | {metrics['max_primal_violation']:<10.4e}"
            )

    # Unscale variables back to original problem coordinate space
    x_unscaled = precond.unscale_primal(best_x)
    lam_unscaled = precond.unscale_dual(best_lam)
    neural_scout_ms = (time.perf_counter() - t0) * 1000.0

    if verbose:
        print(f"   [STAGE 1 COMPLETE] Neural Scout converged in {neural_scout_ms:.1f} ms. Best KKT Resid: {best_res:.4e}")

    # Stage 2: Boundary Snapping
    snap_ms = 0.0
    x_snapped = x_unscaled.copy()
    if apply_snap:
        x_snapped, snap_ms = apply_active_set_linear_snap(kkt_system, x_unscaled)
        if verbose:
            print(f"\n[STAGE 2: ACTIVE-SET SNAP] Closed-form G_viol^+ projection completed in {snap_ms:.2f} ms.")

    x_final = x_snapped.copy()
    if apply_sparsity:
        x_final = apply_basis_sparsity_snapping(kkt_system, x_snapped, lam_unscaled, threshold=sparsity_threshold)
        if verbose:
            n_zeroed = np.sum((x_snapped > 0) & (x_final == 0))
            print(f"[STAGE 2: SPARSITY SNAP] Snapped {n_zeroed} non-basic coordinates to exact zero.")

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "x_opt": x_final,
        "lambda_opt": lam_unscaled,
        "x_scout": x_unscaled,
        "best_kkt_residual": best_res,
        "neural_scout_ms": neural_scout_ms,
        "snap_ms": snap_ms,
        "total_time_ms": total_time_ms
    }
