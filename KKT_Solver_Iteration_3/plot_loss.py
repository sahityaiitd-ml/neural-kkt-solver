"""
plot_loss.py
============
Plots multi-term KKT loss convergence for Iteration 3.
"""

import os
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_iteration_3_convergence(history: Dict[str, Any], save_path: str, title: str = "Iteration 3 Convergence"):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = history["epoch"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, history["loss_total"], label="Total KKT Loss", color="#0284c7", lw=2)
    ax1.plot(epochs, history["loss_stat"], label="Stationarity Loss", color="#dc2626", lw=1.5, ls="--")
    ax1.plot(epochs, history["loss_fb"], label="Fischer-Burmeister Loss", color="#16a34a", lw=1.5, ls="-.")
    ax1.plot(epochs, history["loss_prim"], label="Primal Violation Loss", color="#ea580c", lw=1.2, ls=":")

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss Magnitude", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
