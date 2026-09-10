"""
model.py
========
Defines the dynamic Neural Network architecture for the Single-Instance KINN Solver.

How it works:
-------------
When you pass any optimization problem from KKT_Standalone_Reader:
  - It reads the number of primal variables (n)
  - It reads the number of inequality constraints (m)
  - It automatically shapes the Primal Output Head to size (n)
  - It automatically shapes the Dual Output Head to size (m) with a Softplus layer

Why Softplus?
-------------
In optimization, Dual Feasibility strictly requires:
    lambda >= 0
Instead of adding a penalty or clipping in code, Softplus mathematically guarantees
that every predicted multiplier is strictly non-negative:
    lambda_hat = ln(1 + e^z) >= 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleInstanceKINN(nn.Module):
    """
    A dual-head neural network designed to solve an individual optimization problem instance.
    
    Parameters:
    -----------
    n_vars : int
        Number of primal decision variables in the problem (e.g. 8 for diet problem).
    n_constraints : int
        Number of inequality constraints in the canonical G x <= h system.
    hidden_dim : int, default 64
        Width of the hidden feature-mixing backbone.
    latent_dim : int, default 16
        Dimension of the latent input vector.
    """
    def __init__(self, n_vars: int, n_constraints: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.n_vars = n_vars
        self.n_constraints = n_constraints
        self.latent_dim = latent_dim

        # 1. Latent seed input (learnable or fixed trigger)
        self.register_buffer("seed_input", torch.ones(1, latent_dim, dtype=torch.float32))

        # 2. Shared Hidden Backbone (uses basic ReLU for Iteration 1)
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 3. Head 1: Primal Decision Head (Outputs raw x_hat in R^n)
        self.primal_head = nn.Linear(hidden_dim, n_vars)

        # 4. Head 2: Dual Multiplier Head (Outputs raw lambda_hat in R^m)
        self.dual_head = nn.Linear(hidden_dim, n_constraints)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self):
        """
        Forward pass through the basic network.
        
        Returns:
        --------
        x_hat : torch.Tensor of shape (n_vars,)
            Raw predicted primal decision variables.
        lambda_hat : torch.Tensor of shape (n_constraints,)
            Raw predicted dual multipliers (positivity enforced via loss function).
        """
        features = self.backbone(self.seed_input)

        x_hat = self.primal_head(features).squeeze(0)
        lambda_hat = self.dual_head(features).squeeze(0)

        return x_hat, lambda_hat


class DirectParameterKINN(nn.Module):
    """
    Direct Learnable Parameter version of KINN (No hidden layers).
    
    Treats x and lambda directly as PyTorch parameters, optimized purely via autograd.
    Useful for comparing whether deep neural representations offer an advantage over
    direct gradient descent on the loss landscape!
    """
    def __init__(self, n_vars: int, n_constraints: int):
        super().__init__()
        # Direct primal variables x in R^n
        self.x = nn.Parameter(torch.zeros(n_vars, dtype=torch.float32))
        
        # Raw dual multipliers (will be passed through Softplus)
        self.raw_lambda = nn.Parameter(torch.zeros(n_constraints, dtype=torch.float32))

    def forward(self):
        lambda_hat = F.softplus(self.raw_lambda)
        return self.x, lambda_hat
