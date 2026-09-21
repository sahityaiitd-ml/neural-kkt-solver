"""
solver.py
=========
The Iteration 4.2 Optimization Engine.

Key Highlights:
1. Decoupled neural architecture with GELU & anti-saturation Softplus dual head.
2. Dynamic annealed objective pull (w_obj * gamma^t) expels the solver from origin traps
   without exerting persistent distorting gravity.
3. One-sided duality gap penalty ReLU(c_tilde^T x + h_tilde^T lambda)^2 eliminates
   the false equilibrium trap while respecting weak duality.
4. Checkpoints best solution on pure KKT residual satisfaction rather than raw loss.
5. Canonical row & cost preconditioning ensures O(1) slacks across all constraints.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.optim as optim

from .model import SingleInstanceKINN, DirectParameterKINN
from .loss import iteration_4_2_kkt_loss


def solve_kkt_instance(
    kkt_system,
    model_type: str = "neural",
    max_epochs: int = 1500,
    lr: float = 0.015,
    tol: float = 1e-4,
    w_obj_init: float = 0.5,
    gamma: float = 0.996,
    w_stat: float = 1.0,
    w_gap: float = 1.0,
    w_fb: float = 2.0,
    w_prim: float = 10.0,
    w_x_pos: float = 5.0,
    eps: float = 1e-6,
    dual_bias_init: float = 1.0,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solves a KKT linear program instance using Iteration 4.2 architecture.
    """
    n_vars = kkt_system.n_vars
    n_cons = kkt_system.n_constraints

    # 1. Problem matrices in float64
    c_raw = np.array(kkt_system.c, dtype=np.float64)
    h_raw = np.array(kkt_system.h, dtype=np.float64)
    if hasattr(kkt_system.G, "toarray"):
        G_dense = kkt_system.G.toarray().astype(np.float64)
    else:
        G_dense = np.array(kkt_system.G, dtype=np.float64)

    # 2. Canonical Preconditioning (Row & Cost Normalization)
    row_norms = np.linalg.norm(G_dense, axis=1)
    row_scales = np.where(row_norms < 1e-8, 1.0, row_norms)
    G_norm = G_dense / row_scales[:, np.newaxis]
    h_norm = h_raw / row_scales

    c_norm_val = np.linalg.norm(c_raw)
    cost_scale = float(max(1.0, c_norm_val))
    c_norm = c_raw / cost_scale

    # 3. Convert to PyTorch Tensors
    c_tensor = torch.tensor(c_norm, dtype=torch.float32, device=device)
    G_tensor = torch.tensor(G_norm, dtype=torch.float32, device=device)
    h_tensor = torch.tensor(h_norm, dtype=torch.float32, device=device)

    # 4. Instantiate Model
    if model_type.lower() == "direct":
        model = DirectParameterKINN(
            n_vars=n_vars,
            n_cons=n_cons,
            dual_bias_init=dual_bias_init
        ).to(device)
    else:
        model = SingleInstanceKINN(
            n_vars=n_vars,
            n_cons=n_cons,
            hidden_dim=64,
            dual_bias_init=dual_bias_init
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-5
    )

    history = {
        "epoch": [],
        "loss_total": [],
        "kkt_residual": [],
        "loss_obj": [],
        "w_obj_current": [],
        "loss_stat": [],
        "loss_gap": [],
        "loss_fb": [],
        "loss_prim": [],
        "lr": []
    }

    best_kkt_residual = float("inf")
    best_loss = float("inf")
    best_x = np.zeros(n_vars)
    best_lambda_norm = np.zeros(n_cons)
    converged = False

    t_start = time.perf_counter()

    if verbose:
        print(f"\n[SOLVER] Solving '{kkt_system.name}' with Iteration 4.2 (Annealed Pull + One-Sided Gap)...")
        print(f"   Variables (n): {n_vars} | Constraints (m): {n_cons} | Model: {model_type}")
        print(f"   Cost Scale: {cost_scale:.2e} | Mean Row Scale: {np.mean(row_scales):.2e}")
        print("   " + "-" * 90)
        print(f"   {'Epoch':<8} | {'Total Loss':<12} | {'KKT Resid':<12} | {'w_obj(t)':<10} | {'Stationarity':<13} | {'OneSided Gap':<12} | {'Prim Viol':<10}")
        print("   " + "-" * 90)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        x_hat, lambda_hat = model()

        total_loss, metrics = iteration_4_2_kkt_loss(
            x_hat=x_hat,
            lambda_hat=lambda_hat,
            c_norm=c_tensor,
            G_norm=G_tensor,
            h_norm=h_tensor,
            epoch=epoch,
            w_obj_init=w_obj_init,
            gamma=gamma,
            w_stat=w_stat,
            w_gap=w_gap,
            w_fb=w_fb,
            w_prim=w_prim,
            w_x_pos=w_x_pos,
            eps=eps
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(total_loss.item())

        cur_kkt_res = metrics["kkt_residual"]
        cur_loss = total_loss.item()

        # Checkpoint based on pure KKT residual satisfaction
        if cur_kkt_res < best_kkt_residual:
            best_kkt_residual = cur_kkt_res
            best_loss = cur_loss
            best_x = x_hat.detach().cpu().numpy().copy()
            best_lambda_norm = lambda_hat.detach().cpu().numpy().copy()

        # Log history
        history["epoch"].append(epoch)
        history["loss_total"].append(cur_loss)
        history["kkt_residual"].append(cur_kkt_res)
        history["loss_obj"].append(metrics["loss_obj"])
        history["w_obj_current"].append(metrics["w_obj_current"])
        history["loss_stat"].append(metrics["loss_stat"])
        history["loss_gap"].append(metrics["loss_gap"])
        history["loss_fb"].append(metrics["loss_fb"])
        history["loss_prim"].append(metrics["loss_prim"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and (epoch == 1 or epoch % 200 == 0 or epoch == max_epochs):
            line = (
                f"   {epoch:<8d} | "
                f"{metrics['loss_total']:<12.4e} | "
                f"{metrics['kkt_residual']:<12.4e} | "
                f"{metrics['w_obj_current']:<10.4f} | "
                f"{metrics['loss_stat']:<13.4e} | "
                f"{metrics['loss_gap']:<12.4e} | "
                f"{metrics['max_primal_violation']:<10.4e}"
            )
            print(line)

    solve_time_sec = time.perf_counter() - t_start
    solve_time_ms = solve_time_sec * 1000.0

    # 5. Dual Multiplier Unscaling to Original Space
    best_lambda_unscaled = cost_scale * (best_lambda_norm / row_scales)

    if verbose:
        print("   " + "-" * 90)
        print(f"   [COMPLETE] Solve completed in {solve_time_ms:.2f} ms ({len(history['epoch'])} epochs).")
        print(f"   Best KKT Residual: {best_kkt_residual:.4e} | Best Loss: {best_loss:.4e}")

    return {
        "name": kkt_system.name,
        "x_opt": best_x,
        "lambda_opt": best_lambda_unscaled,
        "best_loss": best_loss,
        "best_kkt_residual": best_kkt_residual,
        "converged": converged,
        "iterations": len(history["epoch"]),
        "solve_time_sec": solve_time_sec,
        "pure_solve_time_ms": solve_time_ms,
        "history": history
    }
