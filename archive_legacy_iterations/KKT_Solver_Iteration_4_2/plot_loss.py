"""
plot_loss.py
============
Visualizer for Iteration 4.2 training convergence.
"""

import os
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_iteration_4_2_convergence(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Iteration 4.2 (Annealed Pull + One-Sided Gap) Convergence"
):
    """
    Plots convergence profiles for Iteration 4.2 loss components.
    """
    epochs = history.get("epoch", [])
    if not epochs:
        print("[WARN] No history available to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Total Loss & KKT Residual
    axes[0, 0].plot(epochs, history["loss_total"], label="Total Loss", color="#1f77b4", linewidth=1.5)
    if "kkt_residual" in history:
        axes[0, 0].plot(epochs, history["kkt_residual"], label="KKT Residual", color="#2ca02c", linewidth=1.5, linestyle="--")
    axes[0, 0].set_title("Total Optimization Loss & KKT Residual")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # 2. Annealed Objective Pull & Weight
    axes[0, 1].plot(epochs, history["loss_obj"], label="Normalized Primal Obj (c_norm^T x)", color="#d62728", linewidth=1.5)
    if "w_obj_current" in history:
        ax2 = axes[0, 1].twinx()
        ax2.plot(epochs, history["w_obj_current"], label="w_obj(t) Decay", color="#ff7f0e", linestyle=":")
        ax2.set_ylabel("w_obj Weight", color="#ff7f0e")
    axes[0, 1].set_title("Primal Objective & Annealed Pull Weight")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Objective Value")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper left")

    # 3. Stationarity & One-Sided Duality Gap
    axes[1, 0].plot(epochs, history["loss_stat"], label="Stationarity", color="#9467bd", linewidth=1.5)
    axes[1, 0].plot(epochs, history["loss_gap"], label="One-Sided Gap", color="#8c564b", linewidth=1.5)
    axes[1, 0].set_title("Stationarity & One-Sided Duality Gap")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # 4. Fischer-Burmeister & Primal Violation
    axes[1, 1].plot(epochs, history["loss_fb"], label="Fischer-Burmeister", color="#e377c2", linewidth=1.5)
    axes[1, 1].plot(epochs, history["loss_prim"], label="Primal Infeasibility", color="#17becf", linewidth=1.5)
    axes[1, 1].set_title("Complementarity & Feasibility Penalties")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Convergence plot saved to: {save_path}")
    plt.close()
