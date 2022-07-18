# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/wustl-cig/DeCoLearn/blob/main/decolearn/torch_util/losses.py

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedCrossCorrelationLoss(nn.Module):
    """Normalized Cross Correlation loss module."""

    def __init__(self, win_size: Optional[int] = None):
        """
        Parameters
        ----------
        win_size: Window size for Normalized Cross Correlation calculation.
        """
        super().__init__()
        self.win_size = win_size

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        I: First input tensor.
        J: Second input tensor.

        Returns
        -------
        Normalized Cross Correlation loss.
        """
        num_dims = I.dim() - 2
        assert num_dims in [1, 2, 3], "Only 1D, 2D, and 3D data are supported."

        if self.win_size is None:
            win_size = [9] * num_dims
        else:
            win_size = [self.win_size] * num_dims

        sum_filter = torch.ones([1, 1, *win_size]).to(I)

        pad = math.floor(win_size[0] / 2)

        if num_dims == 1:
            stride = (1,)  # type: ignore
            padding = (pad,)  # type: ignore
        elif num_dims == 2:
            stride = (1, 1)  # type: ignore
            padding = (pad, pad)  # type: ignore
        elif num_dims == 3:
            stride = (1, 1, 1)  # type: ignore
            padding = (pad, pad, pad)  # type: ignore

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filter, stride, padding, win_size)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.tensor([1]).to(cc) * torch.mean(cc)

    @staticmethod
    def compute_local_sums(I, J, sum_filter, stride, padding, win_size):
        """
        Compute local sums of I and J.
        """
        # Compute CC squares
        I2 = I**2
        J2 = J**2
        IJ = I * J

        I_sum = F.conv2d(I.float(), sum_filter, stride=stride, padding=padding).to(I)
        J_sum = F.conv2d(J.float(), sum_filter, stride=stride, padding=padding).to(I)
        I2_sum = F.conv2d(I2.float(), sum_filter, stride=stride, padding=padding).to(I)
        J2_sum = F.conv2d(J2.float(), sum_filter, stride=stride, padding=padding).to(I)
        IJ_sum = F.conv2d(IJ.float(), sum_filter, stride=stride, padding=padding).to(I)

        win_size = np.prod(win_size)

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross


class GradientLoss(nn.Module):
    """Gradient loss module."""

    def __init__(self, penalty: str = "l2", reg: float = 2.0, dimensionality: int = 2):
        """
        Parameters
        ----------
        penalty: Penalty to use for the loss.
        reg: Regularization parameter.
        dimensionality: Dimensionality of the input.
        """
        super().__init__()
        self.penalty = penalty
        self.reg = reg
        self.dimensionality = dimensionality

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        s: input tensor.

        Returns
        -------
        Gradient loss.
        """
        X = torch.abs(s[:, :1, :, :])
        Y = torch.abs(s[:, 2:4, :, :])
        if self.dimensionality == 3:
            Z = torch.abs(s[:, 4:6, :, :])

        if self.penalty == "l2":
            if self.dimensionality == 2:
                X = X**2
                Y = Y**2
            elif self.dimensionality == 3:
                Z = Z**2
            else:
                raise ValueError("Dimensionality must be 2 or 3.")

        loss = torch.mean(X) + torch.mean(Y)
        if self.dimensionality == 3:
            loss += torch.mean(Z)

        return loss / self.reg
