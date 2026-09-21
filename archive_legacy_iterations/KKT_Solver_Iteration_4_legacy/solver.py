"""
solver.py
=========
The Iteration 4 KINN Optimization Engine with Preconditioning & Strong Duality.

Key Advances:
1. Canonical row & cost normalization ensures every slack sits on O(1) scale.
2. Strong Duality Gap loss actively pulls primal x toward lower cost.
3. Dual head initialized with positive bias (+1.0) to prevent Softplus saturation.
4. Automatic dual multiplier unscaling guarantees exact compatibility with standard KKT residuals.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.optim as optim

from .model import SingleInstanceKINN
from .loss import iteration_4_kkt_loss


def solve_kkt_instance(
    kkt_system,
    model_type: str = "neural",
    max_epochs: int = 1500,
    lr: float = 0.015,
    tol: float = 1e-4,
    w_stat: float = 1.0,
    w_gap: float = 2.0,
    w_fb: float = 2.0,
    w_prim: float = 2.0,
    w_x_pos: float = 2.0,
    eps: float = 1e-6,
    dual_bias_init: float = 1.0,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solves a KKT linear program instance using Iteration 4 architecture.
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
        "loss_stat": [],
        "loss_gap": [],
        "loss_fb": [],
        "loss_prim": [],
        "lr": []
    }

    best_loss = float("inf")
    best_x = np.zeros(n_vars)
    best_lambda_norm = np.zeros(n_cons)
    converged = False

    # Timing: Pure algorithmic solve timer (zero I/O, zero file parsing)
    t_start = time.perf_counter()

    if verbose:
        print(f"\n[SOLVER] Solving '{kkt_system.name}' with Iteration 4 KINN...")
        print(f"   Variables (n): {n_vars} | Constraints (m): {n_cons}")
        print(f"   Cost Scale: {cost_scale:.2e} | Mean Row Scale: {np.mean(row_scales):.2e}")
        print("   " + "-" * 80)
        print(f"   {'Epoch':<8} | {'Total Loss':<12} | {'Stationarity':<13} | {'Duality Gap':<12} | {'FB Loss':<12} | {'Prim Viol':<10}")
        print("   " + "-" * 80)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        x_hat, lambda_hat = model()

        total_loss, metrics = iteration_4_kkt_loss(
            x_hat=x_hat,
            lambda_hat=lambda_hat,
            c_norm=c_tensor,
            G_norm=G_tensor,
            h_norm=h_tensor,
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

        cur_loss = total_loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_x = x_hat.detach().cpu().numpy().copy()
            best_lambda_norm = lambda_hat.detach().cpu().numpy().copy()

        # Log history
        history["epoch"].append(epoch)
        history["loss_total"].append(cur_loss)
        history["loss_stat"].append(metrics["loss_stat"])
        history["loss_gap"].append(metrics["loss_gap"])
        history["loss_fb"].append(metrics["loss_fb"])
        history["loss_prim"].append(metrics["loss_prim"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and (epoch == 1 or epoch % 200 == 0 or epoch == max_epochs):
            line = (
                f"   {epoch:<8d} | "
                f"{metrics['loss_total']:<12.4e} | "
                f"{metrics['loss_stat']:<13.4e} | "
                f"{metrics['duality_gap']:<12.4e} | "
                f"{metrics['loss_fb']:<12.4e} | "
                f"{metrics['max_primal_violation']:<10.4e}"
            )
            print(line)

        if cur_loss < tol:
            converged = True
            if verbose:
                print(f"   [CONVERGED] Early stopping triggered at epoch {epoch} (loss < {tol}).")
            break

    solve_time_sec = time.perf_counter() - t_start
    solve_time_ms = solve_time_sec * 1000.0

    # 5. Dual Multiplier Unscaling to Original Space
    # lambda_orig = cost_scale * (lambda_tilde / row_scales)
    best_lambda_unscaled = cost_scale * (best_lambda_norm / row_scales)

    if verbose:
        print("   " + "-" * 80)
        print(f"   [COMPLETE] Solve completed in {solve_time_ms:.2f} ms ({len(history['epoch'])} epochs).")
        print(f"   Best Loss: {best_loss:.4e} | Converged: {converged}")

    return {
        "name": kkt_system.name,
        "x_opt": best_x,
        "lambda_opt": best_lambda_unscaled,
        "best_loss": best_loss,
        "converged": converged,
        "iterations": len(history["epoch"]),
        "solve_time_sec": solve_time_sec,
        "pure_solve_time_ms": solve_time_ms,
        "history": history
    }
