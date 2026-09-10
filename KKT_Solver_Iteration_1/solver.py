"""
solver.py
=========
The Single-Instance KINN Optimizer Engine.

Workflow:
---------
1. Ingests a standardized KKTSystem from KKT_Standalone_Reader.
2. Dynamically builds the SingleInstanceKINN neural network for (n vars, m cons).
3. Optimizes the network weights using Adam to drive the KKT Loss to zero.
4. Tracks convergence history (Stationarity, Feasibility, Slackness) at each epoch.
5. Returns the final predicted (x*, lambda*), training history, and diagnostics.
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
    w_prim: float = 5.0,
    w_slack: float = 2.0,
    w_dual_pos: float = 5.0,
    w_primal_pos: float = 5.0,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solves a single optimization problem using Physics-Informed KKT training.

    Parameters:
    -----------
    kkt_system : KKTSystem
        The problem instance from KKT_Standalone_Reader.
    model_type : str, default "neural"
        "neural" uses SingleInstanceKINN with hidden layers.
        "direct" uses DirectParameterKINN without hidden layers.
    max_epochs : int, default 2500
        Maximum gradient descent iterations.
    lr : float, default 0.01
        Initial learning rate for Adam optimizer.
    tol : float, default 1e-4
        Convergence tolerance on total KKT loss.
    w_stat, w_prim, w_slack : float
        Weights for the 3 KKT physical loss components.
    verbose : bool, default True
        Whether to print epoch progress to terminal.

    Returns:
    --------
    dict containing:
      - "x_opt": final predicted decision vector (NumPy array)
      - "lambda_opt": final predicted dual multipliers (NumPy array)
      - "converged": bool
      - "iterations": int
      - "solve_time_sec": float
      - "history": dict of training curves
      - "final_metrics": dict
    """
    start_time = time.perf_counter()
    n_vars = kkt_system.n_vars
    n_cons = kkt_system.n_constraints

    # 1. Convert problem parameters to PyTorch tensors
    c_tensor = torch.tensor(kkt_system.c, dtype=torch.float32, device=device)
    G_tensor = torch.tensor(kkt_system.G, dtype=torch.float32, device=device)
    h_tensor = torch.tensor(kkt_system.h, dtype=torch.float32, device=device)

    # 2. Instantiate the neural network dynamically for this problem
    if model_type.lower() == "neural":
        model = SingleInstanceKINN(n_vars=n_vars, n_constraints=n_cons, hidden_dim=64).to(device)
    elif model_type.lower() == "direct":
        model = DirectParameterKINN(n_vars=n_vars, n_constraints=n_cons).to(device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'neural' or 'direct'.")

    # 3. Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=150, min_lr=1e-5)

    # 4. History loggers
    history = {
        "epoch": [],
        "loss_total": [],
        "loss_stat": [],
        "loss_prim": [],
        "loss_slack": [],
        "lr": []
    }

    converged = False
    best_loss = float("inf")
    best_x = None
    best_lambda = None

    if verbose:
        print(f"\n⚡ Solving '{kkt_system.name}' with Single-Instance KINN ({model_type.upper()} Mode)...")
        print(f"   Variables (n): {n_vars} | Constraints (m): {n_cons}")
        print("   " + "-" * 62)
        print(f"   {'Epoch':<8} | {'Total Loss':<12} | {'Stationarity':<13} | {'Primal Viol':<13} | {'Slackness':<11}")
        print("   " + "-" * 62)

    # 5. Training Loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: guess x and lambda
        x_hat, lambda_hat = model()

        # Calculate KKT Loss (with positive primal and dual conditions)
        total_loss, metrics = basic_kkt_loss(
            c=c_tensor,
            G=G_tensor,
            h=h_tensor,
            x_hat=x_hat,
            lambda_hat=lambda_hat,
            w_stat=w_stat,
            w_prim=w_prim,
            w_slack=w_slack,
            w_dual_pos=w_dual_pos,
            w_primal_pos=w_primal_pos,
            normalize=True
        )

        # Backward pass: compute gradients and update weights
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step(total_loss.detach())

        # Record history
        loss_val = metrics["loss_total"]
        if epoch % 10 == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss_val)
            history["loss_stat"].append(metrics["loss_stat"])
            history["loss_prim"].append(metrics["loss_prim"])
            history["loss_slack"].append(metrics["loss_slack"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

        # Track best solution
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x_hat.detach().cpu().numpy().copy()
            best_lambda = lambda_hat.detach().cpu().numpy().copy()

        # Print progress periodically
        if verbose and (epoch % 250 == 0 or epoch == 1):
            print(
                f"   {epoch:<8} | {loss_val:<12.3e} | {metrics['loss_stat']:<13.3e} | "
                f"{metrics['max_primal_violation']:<13.3e} | {metrics['loss_slack']:<11.3e}"
            )

        # Early stopping check
        if loss_val < tol:
            converged = True
            if verbose:
                print(f"\n   🎯 Converged at epoch {epoch} with KKT Loss = {loss_val:.2e} < tolerance {tol:.2e}!")
            break

    elapsed_time = time.perf_counter() - start_time

    if not converged and verbose:
        print(f"\n   ⏹ Reached max epochs ({max_epochs}) with best KKT Loss = {best_loss:.2e}")

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
