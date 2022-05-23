# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/lpd/lpd.py
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn


class DualNet(nn.Module):
    """Dual Network for Learned Primal Dual Network."""

    def __init__(self, num_dual, **kwargs):
        """
        Inits DualNet.

        Parameters
        ----------
        num_dual: Number of dual for LPD algorithm.
        kwargs: Keyword arguments.
        """
        super().__init__()

        if kwargs.get("dual_architecture") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("n_hidden is required for DualNet")

            self.dual_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_dual + 2), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_dual, kernel_size=3, padding=1),
                ]
            )
        else:
            self.dual_block = kwargs.get("dual_architecture")

    @staticmethod
    def compute_model_per_coil(model, data):
        """
        Computes model per coil.

        Parameters
        ----------
        model: Model to compute.
        data: Multi-coil input.

        Returns
        -------
        Multi-coil output.
        """
        output = []
        for idx in range(data.size(1)):
            subselected_data = data.select(1, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=1)
        return output

    def forward(self, h, forward_f, g):
        """Forward pass."""
        inp = torch.cat([h, forward_f, g], dim=-1).permute(0, 1, 4, 2, 3)
        return self.compute_model_per_coil(self.dual_block, inp).permute(0, 1, 3, 4, 2)


class PrimalNet(nn.Module):
    """Primal Network for Learned Primal Dual Network."""

    def __init__(self, num_primal, **kwargs):
        """
        Inits PrimalNet.

        Parameters
        ----------
        num_primal: Number of primal for LPD algorithm.
        """
        super().__init__()

        if kwargs.get("primal_architecture") is None:
            n_hidden = kwargs.get("n_hidden")
            if n_hidden is None:
                raise ValueError("Missing argument n_hidden.")
            self.primal_block = nn.Sequential(
                *[
                    nn.Conv2d(2 * (num_primal + 1), n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(n_hidden, 2 * num_primal, kernel_size=3, padding=1),
                ]
            )
        else:
            self.primal_block = kwargs.get("primal_architecture")

    def forward(self, f, backward_h):
        """
        Forward pass of primal network.

        Parameters
        ----------
        f: Forward function.
        backward_h: Backward function.

        Returns
        -------
        Primal function.
        """
        inp = torch.cat([f, backward_h], dim=-1).permute(0, 3, 1, 2)
        return self.primal_block(inp).permute(0, 2, 3, 1)
