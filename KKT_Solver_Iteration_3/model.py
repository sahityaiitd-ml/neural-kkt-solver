"""
model.py
========
Neural Network Architecture for KKT_Solver_Iteration_3.

Backbone:
- 2 hidden layers with GELU activation functions to ensure continuous gradient flow.

Output Heads:
- Primal Head: Linear projection to R^n (decision variables x_hat).
- Dual Head: Linear projection followed by Softplus to strictly guarantee lambda_hat >= 0.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SingleInstanceKINN(nn.Module):
    """
    Single-instance physics-informed neural network for LP solving.
    Maps a fixed problem latent embedding to optimal primal (x) and dual (lambda) vectors.
    """
    def __init__(self, n_vars: int, n_cons: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.n_vars = n_vars
        self.n_cons = n_cons
        self.latent_dim = latent_dim

        # Fixed latent problem representation seed
        self.latent_seed = nn.Parameter(torch.randn(1, latent_dim), requires_grad=False)

        # Shared representation backbone with GELU activations
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Primal Head (Linear mapping to decision variables)
        self.primal_head = nn.Linear(hidden_dim, n_vars)

        # Dual Head (Softplus to strictly enforce lambda >= 0)
        self.dual_head = nn.Sequential(
            nn.Linear(hidden_dim, n_cons),
            nn.Softplus(beta=1.0)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(self.latent_seed)
        x_hat = self.primal_head(feat).squeeze(0)
        lambda_hat = self.dual_head(feat).squeeze(0)
        return x_hat, lambda_hat


class DirectParameterKINN(nn.Module):
    """
    Direct parameter representation baseline (optimizing decision variables directly).
    """
    def __init__(self, n_vars: int, n_cons: int):
        super().__init__()
        self.x_raw = nn.Parameter(torch.zeros(n_vars))
        self.lambda_raw = nn.Parameter(torch.zeros(n_cons))
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.x_raw
        lambda_hat = self.softplus(self.lambda_raw)
        return x_hat, lambda_hat


class KINNSolverModel(SingleInstanceKINN):
    pass
