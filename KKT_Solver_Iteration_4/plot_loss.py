"""
plot_loss.py
============
Visualizes convergence trajectories for Iteration 4.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_iteration_4_convergence(history: Dict[str, Any], save_path: str, title: str):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["loss_total"], label="Total Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["loss_stat"], label="Stationarity", color="#ff7f0e", linestyle="--")
    plt.plot(epochs, history["loss_obj"], label="Objective Pull", color="#2ca02c", linestyle="-.")
    plt.plot(epochs, history["loss_gap"], label="Duality Gap", color="#d62728", linestyle=":")
    plt.plot(epochs, history["loss_prim"], label="Primal Feasibility", color="#9467bd", alpha=0.7)

    plt.yscale("log")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss Magnitude (Log Scale)")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Backward compatibility alias
plot_iteration_4_1_convergence = plot_iteration_4_convergence
