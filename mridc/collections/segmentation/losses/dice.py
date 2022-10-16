# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py

import warnings
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

import mridc.collections.common.parts.utils as utils
from mridc.collections.segmentation.losses.utils import do_metric_reduction


class Dice(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        flatten: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        include_background: bool
            whether to skip Dice computation on the first channel of the predicted output. Defaults to True.
        to_onehot_y: bool
            Whether to convert `y` into the one-hot format. Defaults to False.
        sigmoid: bool
            Whether to add sigmoid function to the input data. Defaults to True.
        softmax: bool
            Whether to add softmax function to the input data. Defaults to False.
        other_act: Callable
            Use this parameter if you want to apply another type of activation layer.
            Defaults to None.
        squared_pred: bool
            Whether to square the prediction before calculating Dice. Defaults to False.
        jaccard: bool
            Whether to compute Jaccard Index as a loss. Defaults to False.
        flatten: bool
            Whether to flatten input data. Defaults to False.
        reduction: str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied.
            'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed.
            Default: 'mean'
        smooth_nr: float
            A small constant added to the numerator to avoid `nan` when all items are 0.
        smooth_dr: float
            A small constant added to the denominator to avoid `nan` when all items are 0.
        batch: bool
            If True, compute Dice loss for each batch and return a tensor with shape (batch_size,).
            If False, compute Dice loss for the whole batch and return a tensor with shape (1,).
        """
        super().__init__()
        other_act = None if utils.is_none(other_act) else other_act
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.flatten = flatten
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, target: torch.Tensor, input: torch.Tensor) -> Tuple[Union[Tensor, Any], Tensor]:
        """
        Parameters
        ----------
        input: torch.Tensor
            the prediction of shape [BNHW[D]].
        target: torch.Tensor
            the ground truth of shape [BNHW[D]].
        """
        if self.flatten:
            target = target.view(target.shape[0], 1, -1)
            input = input.view(input.shape[0], 1, -1)

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        dice_score = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        dice_score = torch.where(denominator > 0, dice_score, torch.tensor(1.0).to(pred_o.device))
        dice_score, _ = do_metric_reduction(dice_score, reduction=self.reduction)

        f: torch.Tensor = 1.0 - dice_score

        return dice_score, f


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    Convert labels to one-hot representation.

    Parameters
    ----------
    labels: torch.Tensor
        the labels of shape [BNHW[D]].
    num_classes: int
        number of classes.
    dtype: torch.dtype
        the data type of the returned tensor.
    dim: int
        the dimension to expand the one-hot tensor.
    """
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels
