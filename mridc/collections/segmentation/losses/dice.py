# coding=utf-8
__author__ = "Dimitrios Karkalousos, Lysander de Jong"

import warnings

import torch
from torch import nn as nn


class Dice(nn.Module):
    """Dice score for multi-class segmentation."""

    def __init__(
        self,
        batched: bool = False,
        include_background: bool = True,
        squared_pred: bool = False,
        normalization: str = "softmax",
        epsilon: float = 1e-5,
    ):
        """
        Parameters
        ----------
        batched : bool, optional
            Whether to reduce batch dimension, by default False
        include_background : bool, optional
            Whether to include background class, by default True
        squared_pred : bool, optional
            Whether to square prediction, by default False
        normalization : str, optional
            Normalization method, by default "softmax"
        epsilon : float, optional
            Epsilon for numerical stability, by default 1e-5
        """
        super().__init__()
        self.batched = batched
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.epsilon = epsilon

        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Dice loss normalization should be either sigmoid or softmax. Found {normalization}")

    def weights(self, x):
        return torch.reciprocal(x)

    def forward(self, target, input):
        """Forward pass of Dice computation."""
        if target.shape != input.shape:
            raise AssertionError(f"Ground truth has different shape ({target.shape}) from input ({input.shape})")

        target = self.normalization(target)
        input = self.normalization(input)

        n_pred_ch = input.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("Single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        if self.batched:
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o

        dice = (2.0 * intersection * self.weights(ground_o) + self.epsilon) / (
            denominator * self.weights(ground_o) + self.epsilon
        )

        return dice
