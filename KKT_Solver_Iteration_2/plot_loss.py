"""
plot_loss.py
============
Utility to plot and save KINN convergence curves (Epoch vs Loss) on a log scale.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_kinn_convergence(history: Dict[str, Any], title: str = "KINN Convergence", save_path: str = None):
    """
    Plots the decay of the 3 individual KKT loss components over epochs.
    """
    epochs = history["epoch"]
    loss_tot = history["loss_total"]
    loss_stat = history["loss_stat"]
    loss_prim = history["loss_prim"]
    loss_slack = history["loss_slack"]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, loss_tot, label="Total KKT Loss", color="black", linewidth=2.0)
    plt.plot(epochs, loss_stat, label="Stationarity Error ||c + G^T λ||", color="#0288d1", linestyle="--")
    plt.plot(epochs, loss_prim, label="Primal Infeasibility ||max(0, Gx - h)||", color="#d32f2f", linestyle="-.")
    plt.plot(epochs, loss_slack, label="Slackness ||λ ⊙ (Gx - h)||", color="#388e3c", linestyle=":")

    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Log Scale)", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📈 Convergence plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()
