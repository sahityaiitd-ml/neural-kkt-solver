"""
model.py
========
Decoupled Architecture for KKT_Solver_Iteration_4.

Key Architectural Principles:
-----------------------------
1. Fully Decoupled Networks: Primal and Dual branches have completely separate
   hidden layers. This eliminates gradient cross-talk between primal cost minimization
   and dual multiplier stationarity.
2. Smooth Activations: GELU activations throughout to ensure non-zero gradient flow.
3. Anti-Saturation Dual Head: Linear bias initialized to +1.0 so dual multipliers
   start in the high-gradient linear regime (sigma ~ 0.73) of Softplus.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SingleInstanceKINN(nn.Module):
    """
    Decoupled single-instance KINN solver.
    Maps fixed problem latent seeds to primal variables x and dual multipliers lambda
    using independent, non-interfering neural pathways.
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

        # Fixed latent problem representation seeds
        self.primal_seed = nn.Parameter(torch.randn(1, latent_dim), requires_grad=False)
        self.dual_seed = nn.Parameter(torch.randn(1, latent_dim), requires_grad=False)

        # 1. Independent Primal Network (specialized for polytope navigation)
        self.primal_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_vars)
        )

        # 2. Independent Dual Network (specialized for multiplier alignment)
        self.dual_backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.dual_out = nn.Linear(hidden_dim, n_cons)
        self.dual_act = nn.Softplus(beta=1.0)

        self._init_weights(dual_bias_init=dual_bias_init)

    def _init_weights(self, dual_bias_init: float):
        # Initialize primal network
        for m in self.primal_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize dual network
        for m in self.dual_backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.dual_out.weight, gain=0.5)
        if self.dual_out.bias is not None:
            nn.init.constant_(self.dual_out.bias, dual_bias_init)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.primal_net(self.primal_seed).squeeze(0)
        dual_feat = self.dual_backbone(self.dual_seed)
        lambda_hat = self.dual_act(self.dual_out(dual_feat)).squeeze(0)
        return x_hat, lambda_hat
