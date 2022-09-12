# coding=utf-8
__author__ = "Dimitrios Karkalousos, Lysander de Jong"

import monai.metrics as MM
import torch


def binarize_input(x):
    """Binarize input tensor."""
    tensor_label = x.argmax(1).byte()
    binary_x = torch.zeros_like(x, device=x.device)
    for i in range(x.shape[1]):
        binary_x[:, i, ...] = tensor_label == i
    return binary_x.byte()


def hausdorff_distance(target, pred, include_background=True):
    """Compute Hausdorff distance."""
    return MM.compute_hausdorff_distance(
        torch.softmax(pred, dim=1),
        torch.softmax(target, dim=1),
        include_background=include_background,
        percentile=95,
    )


def average_surface_distance(target, pred, include_background=True):
    """Compute average surface distance."""
    return MM.compute_average_surface_distance(
        torch.softmax(pred, dim=1), torch.softmax(target, dim=1), include_background=include_background
    )
