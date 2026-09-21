"""
model.py
========
Neural Network Architecture for KKT_Solver_Iteration_4.

Design Principles:
------------------
1. Shared representation: 2-layer MLP with GELU activations to preserve gradient flow.
2. Primal Head: Direct linear projection to decision variables (x_hat).
3. Dual Head: Linear projection with Softplus to guarantee non-negativity (lambda_hat >= 0).
4. Anti-Saturation Bias: Dual head linear bias initialized to +1.0 so multipliers begin
   in the high-gradient linear regime rather than the dead zero-gradient flat regime.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SingleInstanceKINN(nn.Module):
    """
    Single-instance physics-informed neural network for LP solving.
    Maps a fixed problem latent seed to primal variables (x) and normalized duals (lambda).
    """
    def __init__(
        self,
        n_vars: int,
        n_cons: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        dual_bias_init: float = 1.0
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_cons = n_cons
        self.latent_dim = latent_dim

        # Fixed problem latent seed
        self.latent_seed = nn.Parameter(torch.randn(1, latent_dim), requires_grad=False)

        # Shared representation backbone with smooth GELU activations
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Primal Head: outputs primal decision variables x_hat in R^n
        self.primal_head = nn.Linear(hidden_dim, n_vars)

        # Dual Head: outputs normalized non-negative dual multipliers lambda_hat >= 0
        self.dual_linear = nn.Linear(hidden_dim, n_cons)
        self.dual_act = nn.Softplus(beta=1.0)

        self._init_weights(dual_bias_init=dual_bias_init)

    def _init_weights(self, dual_bias_init: float):
        # Initialize backbone with Xavier uniform
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize primal head
        nn.init.xavier_uniform_(self.primal_head.weight, gain=0.5)
        if self.primal_head.bias is not None:
            nn.init.zeros_(self.primal_head.bias)

        # Initialize dual head: positive bias prevents Softplus saturation
        nn.init.xavier_uniform_(self.dual_linear.weight, gain=0.5)
        if self.dual_linear.bias is not None:
            nn.init.constant_(self.dual_linear.bias, dual_bias_init)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(self.latent_seed)
        x_hat = self.primal_head(feat).squeeze(0)
        lambda_hat = self.dual_act(self.dual_linear(feat)).squeeze(0)
        return x_hat, lambda_hat
