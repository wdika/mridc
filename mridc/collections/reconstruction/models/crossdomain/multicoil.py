# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/crossdomain/multicoil.py
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn


class MultiCoil(nn.Module):
    """This makes the forward pass of multi-coil data of shape (N, N_coils, H, W, C) to a model.
    If coil_to_batch is set to True, coil dimension is moved to the batch dimension. Otherwise, it passes to the model
    each coil-data individually.
    """

    def __init__(self, model: nn.Module, coil_dim: int = 1, coil_to_batch: bool = False):
        """Inits MultiCoil.
        Parameters
        ----------
        model: nn.Module
            Any nn.Module that takes as input with 4D data (N, H, W, C). Typically a convolutional-like model.
        coil_dim: int
            Coil dimension. Default: 1.
        coil_to_batch: bool
            If True batch and coil dimensions are merged when forwarded by the model and unmerged when outputted.
            Otherwise, input is forwarded to the model per coil.
        """
        super().__init__()

        self.model = model
        self.coil_to_batch = coil_to_batch
        self._coil_dim = coil_dim

    def _compute_model_per_coil(self, data: torch.Tensor) -> torch.Tensor:
        """Computes the model per coil."""
        output = []

        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.model(subselected_data.unsqueeze(1)).squeeze(1))
        output = torch.stack(output, dim=self._coil_dim)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of MultiCoil.

        Parameters
        ----------
        x: torch.Tensor
            Multi-coil input of shape (N, coil, height, width, in_channels).
        Returns
        -------
        out: torch.Tensor
            Multi-coil output of shape (N, coil, height, width, out_channels).
        """
        if self.coil_to_batch:
            x = x.clone()

            batch, coil, channels, height, width = x.size()
            x = x.reshape(batch * coil, channels, height, width).contiguous()
            x = self.model(x).permute(0, 2, 3, 1)
            x = x.reshape(batch, coil, height, width, -1).permute(0, 1, 4, 2, 3)
        else:
            x = self._compute_model_per_coil(x).contiguous()

        return x
