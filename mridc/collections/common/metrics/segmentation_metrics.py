# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import warnings
from typing import Any, List, Union

import numpy as np
import torch
from runstats import Statistics
from scipy.ndimage import _ni_support, binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff
from torch import Tensor
from torchmetrics import functional as F

from mridc.collections.segmentation.losses import Dice
from mridc.collections.segmentation.losses.utils import do_metric_reduction


def binary_cross_entropy_with_logits_metric(gt: torch.Tensor, pred: torch.Tensor, reduction: str = "mean") -> float:
    """Compute Binary Cross Entropy with Logits"""
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, gt, reduction=reduction).item()


def dice_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    other_act: Union[str, None] = None,
    squared_pred: bool = False,
    jaccard: bool = False,
    flatten: bool = False,
    reduction: Union[str, None] = "mean_batch",
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    batch: bool = True,
) -> float:
    """
    Compute Dice Score

    Parameters
    ----------
    gt : torch.Tensor
        Ground Truth Tensor
    pred : torch.Tensor
        Prediction Tensor
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

    Returns
    -------
    float
        Dice Score
    """
    custom_dice = Dice(
        include_background=include_background,
        to_onehot_y=to_onehot_y,
        sigmoid=sigmoid,
        softmax=softmax,
        other_act=other_act,  # type: ignore
        squared_pred=squared_pred,
        jaccard=jaccard,
        flatten=flatten,
        reduction=reduction,  # type: ignore
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        batch=batch,
    )
    dice_score, _ = custom_dice(gt, pred)
    return dice_score.item()


def generalized_dice_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    include_background: bool = True,
    weight_type: Union[str, None] = "simple",
    reduction: Union[str, None] = "mean_batch",
) -> torch.Tensor:
    """Computes the Generalized Dice Score and returns a tensor with its per image values."""
    # gt = gt.type(torch.uint8)
    # pred = pred.type(torch.uint8)

    # Ensure tensors have at least 3 dimensions and have the same shape
    dims = pred.dim()
    if dims < 3:
        raise ValueError(f"Prediction should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
    if gt.shape != pred.shape:
        raise ValueError(f"Prediction - {pred.shape} - and ground truth - {gt.shape} - should have the same shapes.")

    # Ignore background, if needed
    if not include_background:
        if pred.shape[1] == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            gt = gt[:, 1:]
            pred = pred[:, 1:]

    # Reducing only spatial dimensions (not batch nor channels), compute the intersection and non-weighted denominator
    reduce_axis = list(range(2, pred.dim()))
    intersection = torch.sum(gt * pred, dim=reduce_axis)
    y_o = torch.sum(gt, dim=reduce_axis)
    y_pred_o = torch.sum(pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    # Set the class weights
    if weight_type == "simple":
        w = torch.reciprocal(y_o.float())
    elif weight_type == "square":
        w = torch.reciprocal(y_o.float() * y_o.float())
    else:
        w = torch.ones_like(y_o.float())

    # Replace infinite values for non-appearing classes by the maximum weight
    for b in w:
        infs = torch.isinf(b)
        b[infs] = 0
        b[infs] = torch.max(b)

    # Compute the weighted numerator and denominator, summing along the class axis
    numer = 2.0 * (intersection * w).sum(dim=1)
    denom = (denominator * w).sum(dim=1)

    # Compute the score
    generalized_dice_score = numer / denom
    generalized_dice_score = generalized_dice_score.float()

    # Handle zero division. Where denom == 0 and the prediction volume is 0, score is 1.
    # Where denom == 0 but the prediction volume is not 0, score is 0
    y_pred_o = y_pred_o.sum(dim=-1)
    denom_zeros = denom == 0

    generalized_dice_score[denom_zeros] = torch.where(
        (y_pred_o == 0)[denom_zeros], torch.tensor(1.0), torch.tensor(0.0)
    )
    generalized_dice_score, _ = do_metric_reduction(generalized_dice_score, reduction=reduction)  # type: ignore

    return generalized_dice_score.item()


def f1_per_class_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    beta: float = 1e-5,
    average: str = "none",
    mdmc_average: str = "samplewise",
    threshold: float = 0.0,
) -> List[Any]:
    """
    Compute F1 Score per Class

    Parameters
    ----------
    gt : torch.Tensor
        Ground Truth Tensor
    pred : torch.Tensor
        Prediction Tensor
    beta: float
        Beta value for F1 score. Defaults to 1e-5.
    average: str
        Defines the averaging performed in the binary case:
        'micro' calculates metrics globally
        'macro' calculates metrics for each label, and finds their unweighted mean.
        'weighted' calculates metrics for each label, and finds their average, weighted by support
        (the number of true instances for each label).
        'none' returns the score for each class.
        Default: 'none'
    mdmc_average: str
        Defines the averaging performed in the multiclass case:
        'samplewise' calculates metrics for each sample, and finds their unweighted mean.
        'global' calculates metrics globally, across all samples.
        Default: 'samplewise'

    Returns
    -------
    float
        F1 Score per Class
    """
    f1_per_class = F.fbeta_score(
        pred.to(torch.uint8),
        gt.to(torch.uint8),
        beta=beta,
        average=average,
        mdmc_average=mdmc_average,
        num_classes=gt.shape[1],
        threshold=threshold,
    )
    return [f1_per_class[i].item() for i in range(gt.shape[1])]


def hausdorff_distance_metric(gt: torch.Tensor, pred: torch.Tensor, batched: bool = True) -> float:
    """Compute Hausdorff Distance"""
    if batched:
        hd = []
        for sl in range(gt.shape[0]):
            hd.append(
                max(
                    directed_hausdorff(gt[sl].argmax(0).numpy(), pred[sl].argmax(0).numpy())[0],
                    directed_hausdorff(pred[sl].argmax(0).numpy(), gt[sl].argmax(0).numpy())[0],
                )
            )
        return sum(hd) / len(hd)
    return max(
        directed_hausdorff(gt.argmax(0).numpy(), pred.argmax(0).numpy())[0],
        directed_hausdorff(pred.argmax(0).numpy(), gt.argmax(0).numpy())[0],
    )


def hausdorff_distance_95_metric(gt: torch.Tensor, pred: torch.Tensor, batched: bool = True) -> float:
    """Compute 95th percentile of the  Hausdorff Distance"""
    if batched:
        hd = []
        for sl in range(gt.shape[0]):
            hd1 = directed_hausdorff(gt[sl].argmax(0).numpy(), pred[sl].argmax(0).numpy())[0]
            hd2 = directed_hausdorff(pred[sl].argmax(0).numpy(), gt[sl].argmax(0).numpy())[0]
            hd.append(np.percentile(np.hstack((hd1, hd2)), 95))
        return sum(hd) / len(hd)
    hd1 = directed_hausdorff(gt.argmax(0).numpy(), pred.argmax(0).numpy())[0]
    hd2 = directed_hausdorff(pred.argmax(0).numpy(), gt.argmax(0).numpy())[0]
    return np.percentile(np.hstack((hd1, hd2)), 95)


def iou_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    include_background: bool = True,
    ignore_empty: bool = True,
    reduction: Union[str, None] = "mean",
) -> Tensor:
    """Compute Intersection over Union"""
    if not include_background:
        if pred == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            gt = gt[:, 1:]
            pred = pred[:, 1:]

    # gt = gt.type(torch.uint8)
    # pred = pred.type(torch.uint8)

    if gt.shape != pred.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {pred.shape} and {gt.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(gt * pred, dim=reduce_axis)

    y_o = torch.sum(gt, reduce_axis)
    y_pred_o = torch.sum(pred, dim=reduce_axis)
    union = y_o + y_pred_o - intersection

    max = 1.0 if not ignore_empty else float("nan")
    iou_score = torch.where(union > 0, (intersection) / union, torch.tensor(max, device=y_o.device))
    iou_score, _ = do_metric_reduction(iou_score, reduction=reduction)  # type: ignore

    return iou_score.item()


def precision_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    include_background: bool = True,
    average="none",
    mdmc_average="samplewise",
    reduction: Union[str, None] = "mean_batch",
) -> Tensor:
    """Compute Surface Distance"""
    if not include_background:
        if pred == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            gt = gt[:, 1:]
            pred = pred[:, 1:]

    gt = gt.type(torch.uint8)
    pred = pred.type(torch.uint8)

    if gt.shape != pred.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {pred.shape} and {gt.shape}.")

    precision_score = F.precision(
        pred,
        gt,
        average=average,
        mdmc_average=mdmc_average,
        num_classes=pred.shape[1],
    )
    precision_score, _ = do_metric_reduction(precision_score, reduction=reduction)  # type: ignore
    return precision_score.item()


def recall_metric(
    gt: torch.Tensor,
    pred: torch.Tensor,
    include_background: bool = True,
    average="none",
    mdmc_average="samplewise",
    reduction: Union[str, None] = "mean_batch",
) -> Tensor:
    """Compute Surface Distance"""
    if not include_background:
        if pred == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            gt = gt[:, 1:]
            pred = pred[:, 1:]

    gt = gt.type(torch.uint8)
    pred = pred.type(torch.uint8)

    if gt.shape != pred.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {pred.shape} and {gt.shape}.")

    recall_score = F.recall(
        pred,
        gt,
        average=average,
        mdmc_average=mdmc_average,
        num_classes=pred.shape[1],
    )
    recall_score, _ = do_metric_reduction(recall_score, reduction=reduction)  # type: ignore
    return recall_score.item()


def asd(reference, result, voxelspacing=None, connectivity=1):
    """Compute Average Symmetric Surface Distance (ASD) between a binary object and its reference."""
    sd1 = np.mean(__surface_distances(result, reference, voxelspacing, connectivity))
    sd2 = np.mean(__surface_distances(reference, result, voxelspacing, connectivity))
    return (sd1 + sd2) / 2.0


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their nearest partner surface voxel of a
    binary object in reference.

    Taken from: https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L458
    """
    result = result.numpy()
    reference = reference.numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if 0 == np.count_nonzero(reference):
        raise RuntimeError("The second supplied array does not contain any binary object.")

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the foreground objects,
    # therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs, output_path, method):
        """
        Parameters
        ----------
        metric_funcs (dict): A dict where the keys are metric names and the values are Python functions for evaluating
        that metric.
        output_path: path to the output directory
        method: reconstruction method
        """
        self.metric_funcs = metric_funcs
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}
        self.output_path = output_path
        self.method = method

    def push(self, target, recons):
        """
        Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        target: target image
        recons: reconstructed image

        Returns
        -------
        dict: A dict where the keys are metric names and the values are
        """
        for metric, func in self.metric_funcs.items():
            score = func(target, recons)
            if isinstance(score, list):
                for i in range(len(score)):
                    if metric == f"F1_{i}":
                        self.metrics_scores[metric].push(score[i])
            else:
                self.metrics_scores[metric].push(score)

    def means(self):
        """Mean of the means of each metric."""
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """Standard deviation of the means of each metric."""
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """Representation of the metrics."""
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        with open(f"{self.output_path}metrics.txt", "a") as output:
            output.write(f"{self.method}: {res}")

        return res
