"""
solver.py
=========
The Single-Instance KINN Optimizer Engine (Iteration 3 - Path B).

Key Innovations:
----------------
1. Solves the KKT system using the unified Fischer-Burmeister C-function.
2. Dual variables lambda_hat >= 0 are enforced structurally via Softplus.
3. Tracks historical best-loss solution snapshot to avoid late-epoch noise.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.optim as optim

from .model import SingleInstanceKINN, DirectParameterKINN
from .loss import basic_kkt_loss


def solve_kkt_instance(
    kkt_system,
    model_type: str = "neural",
    max_epochs: int = 2500,
    lr: float = 0.01,
    tol: float = 1e-4,
    w_stat: float = 1.0,
    w_fb: float = 5.0,
    w_prim: float = 1.0,
    w_primal_pos: float = 5.0,
    eps: float = 1e-6,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solves a single linear program using the Fischer-Burmeister KKT formulation.
    """
    n_vars = kkt_system.n_vars
    n_cons = kkt_system.n_constraints

    # Convert problem matrices to tensors
    c_tensor = torch.tensor(kkt_system.c, dtype=torch.float32, device=device)
    h_tensor = torch.tensor(kkt_system.h, dtype=torch.float32, device=device)

    # Handle dense or sparse constraint matrix G
    if hasattr(kkt_system.G, "toarray"):
        G_dense = kkt_system.G.toarray()
    else:
        G_dense = np.array(kkt_system.G)
    G_tensor = torch.tensor(G_dense, dtype=torch.float32, device=device)

    # Instantiate model
    if model_type == "neural":
        model = SingleInstanceKINN(n_vars=n_vars, n_cons=n_cons, hidden_dim=64).to(device)
    elif model_type == "direct":
        model = DirectParameterKINN(n_vars=n_vars, n_cons=n_cons).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=150, min_lr=1e-5
    )

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_stat": [],
        "loss_fb": [],
        "loss_prim": [],
        "loss_slack": [],
        "lr": []
    }

    best_loss = float("inf")
    best_x = np.zeros(n_vars)
    best_lambda = np.zeros(n_cons)
    converged = False

    start_time = time.perf_counter()

    if verbose:
        print(f"\nSolving '{kkt_system.name}' with Iteration 3 KINN (Path B: Fischer-Burmeister)...")
        print(f"   Variables (n): {n_vars} | Constraints (m): {n_cons}")
        print("   " + "-" * 72)
        print(f"   {'Epoch':<8} | {'Total Loss':<12} | {'Stationarity':<13} | {'FB Loss':<12} | {'Primal Viol':<11}")
        print("   " + "-" * 72)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        x_hat, lambda_hat = model()

        total_loss, metrics = basic_kkt_loss(
            x_hat=x_hat,
            lambda_hat=lambda_hat,
            c=c_tensor,
            G=G_tensor,
            h=h_tensor,
            w_stat=w_stat,
            w_fb=w_fb,
            w_prim=w_prim,
            w_primal_pos=w_primal_pos,
            eps=eps
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_val = float(total_loss.item())
        scheduler.step(loss_val)

        if epoch % 50 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss_val)
            history["loss_stat"].append(metrics["loss_stat"])
            history["loss_fb"].append(metrics["loss_fb"])
            history["loss_prim"].append(metrics["loss_prim"])
            history["loss_slack"].append(metrics["loss_slack"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

        # Track best solution snapshot
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x_hat.detach().cpu().numpy().copy()
            best_lambda = lambda_hat.detach().cpu().numpy().copy()

        if verbose and (epoch % 250 == 0 or epoch == 1):
            print(
                f"   {epoch:<8} | {loss_val:<12.3e} | {metrics['loss_stat']:<13.3e} | "
                f"{metrics['loss_fb']:<12.3e} | {metrics['max_primal_violation']:<11.3e}"
            )

        # Early stopping check
        if loss_val < tol:
            converged = True
            if verbose:
                print(f"\n   Converged at epoch {epoch} with KKT Loss = {loss_val:.2e} < tolerance {tol:.2e}!")
            break

    elapsed_time = time.perf_counter() - start_time

    if not converged and verbose:
        print(f"\n   Reached max epochs ({max_epochs}) with best KKT Loss = {best_loss:.2e}")

    return {
        "name": kkt_system.name,
        "model_type": model_type,
        "x_opt": best_x,
        "lambda_opt": best_lambda,
        "converged": converged,
        "iterations": epoch,
        "solve_time_sec": elapsed_time,
        "best_kkt_loss": best_loss,
        "history": history
    }
