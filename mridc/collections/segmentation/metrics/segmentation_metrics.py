# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import warnings
from typing import Any, List, Union

import numpy as np
import torch
from runstats import Statistics
from scipy.ndimage import _ni_support, binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff
from torchmetrics import functional as F

from mridc.collections.segmentation.losses import Dice
from mridc.collections.segmentation.losses.dice import one_hot
from mridc.collections.segmentation.losses.utils import do_metric_reduction


def binary_cross_entropy_with_logits_metric(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> float:
    """
    Compute Binary Cross Entropy with Logits.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    reduction : str
        Specifies the reduction to apply to the output: ``none`` | ``mean`` | ``sum``.
        ``none``: no reduction will be applied. ``mean``: the sum of the output will be divided by the number of
        elements. ``sum``: the output will be summed. Default is``mean``.

    Returns
    -------
    float
        Binary Cross Entropy with Logits.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import binary_cross_entropy_with_logits_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> binary_cross_entropy_with_logits_metric(datax, datay)
    0.7518648505210876

    .. note::
        This function is equivalent to `torch.nn.functional.binary_cross_entropy_with_logits` with `reduction='mean'`.
        Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(x.float(), y.float(), reduction=reduction).item()


def dice_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    other_act: Union[str, None] = None,
    squared_y: bool = False,
    jaccard: bool = False,
    flatten: bool = False,
    reduction: Union[str, None] = "mean_batch",
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    batch: bool = True,
) -> float:
    """
    Compute Dice Score.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    include_background : bool
        Whether to skip Dice computation on the first channel of the predicted output. Default is ``True``.
    to_onehot_y : bool
        Whether to convert `y` into the one-hot format. Default is ``False``.
    sigmoid : bool
        Whether to add sigmoid function to the input data. Default is ``True``.
    softmax : bool
        Whether to add softmax function to the input data. Default is ``False``.
    other_act : Union[str, None]
        Use this parameter if you want to apply another type of activation layer. Default is ``None``.
    squared_y : bool
        Whether to square the prediction before calculating Dice. Default is ``False``.
    jaccard : bool
        Whether to compute Jaccard Index as a loss. Default is ``False``.
    flatten : bool
        Whether to flatten input data. Default is ``False``.
    reduction : Union[str, None]
        Specifies the reduction to apply to the output: ``none`` | ``mean`` | ``sum``.
        ``none``: no reduction will be applied. ``mean``: the sum of the output will be divided by the number of
        elements. ``sum``: the output will be summed. Default is ``mean``.
    smooth_nr : float
        A small constant added to the numerator to avoid ``nan`` when all items are 0.
    smooth_dr : float
        A small constant added to the denominator to avoid ``nan`` when all items are 0.
    batch : bool
        If True, compute Dice loss for each batch and return a tensor with shape (batch_size,).
        If False, compute Dice loss for the whole batch and return a tensor with shape (1,).
        Default is ``True``.
    Returns
    -------
    float
        Dice Score.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import dice_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> dice_metric(datax, datay)
    0.5016108751296997
    """
    custom_dice = Dice(
        include_background=include_background,
        to_onehot_y=to_onehot_y,
        sigmoid=sigmoid,
        softmax=softmax,
        other_act=other_act,  # type: ignore
        squared_pred=squared_y,
        jaccard=jaccard,
        flatten=flatten,
        reduction=reduction,  # type: ignore
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        batch=batch,
    )
    dice_score, _ = custom_dice(x, y)
    return dice_score.item()


def f1_per_class_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1e-5,
    average: str = "mean",
    mdmc_average: str = "samplewise",
    threshold: float = 0.0,
) -> List[Any]:
    """
    Compute F1 Score per Class. If the input has only one class, the output will be a list with one element.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    beta : float
        Beta value for F1 score. Default is ``1e-5``.
    average : str
        Defines the averaging performed in the binary case:
        ``micro`` calculates metrics globally,
        ``macro`` calculates metrics for each label, and finds their unweighted mean,
        ``weighted`` calculates metrics for each label, and finds their average, weighted by support
        (the number of true instances for each label),
        ``none`` returns the score for each class.
        Default is ``none``.
    mdmc_average : str
        Defines the averaging performed in the multiclass case:
        ``samplewise`` calculates metrics for each sample, and finds their unweighted mean,
        ``global`` calculates metrics globally, across all samples.
        Default is ``samplewise``.
    threshold : float
        Threshold value for binarization. Default is ``0.0``.

    Returns
    -------
    List[Any]
        F1 Score per Class.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import f1_per_class_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> f1_per_class_metric(datax, datay)
    [0.49855247139930725, 0.49478909373283386]

    .. note::
        This function is a wrapper for `torchmetrics.functional.classification.fbeta_score`.
    """
    f1_per_class = F.fbeta_score(
        y.to(torch.uint8),
        x.to(torch.uint8),
        task="binary",
        beta=beta,
        average=average,
        multidim_average=mdmc_average,
        num_classes=x.shape[1],
        threshold=threshold,
    )
    if f1_per_class.dim() == 0:
        f1_per_class = torch.stack([f1_per_class] * x.shape[1])
    return [f1_per_class[i].item() for i in range(x.shape[1])]


def hausdorff_distance_metric(x: torch.Tensor, y: torch.Tensor, batched: bool = True) -> float:
    """
    Compute Hausdorff Distance.

    The Hausdorff distance is computed as the maximum between x to y and y to x.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    batched : bool
        If True, compute Hausdorff distance for each batch and return a tensor with shape (batch_size,).
        If False, compute Hausdorff distance for the whole batch and return a tensor with shape (1,).
        Default is ``True``.

    Returns
    -------
    float
        Hausdorff Distance.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import hausdorff_distance_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> hausdorff_distance_metric(datax, datay)
    5.858907404245753
    """
    if batched:
        hd = []
        for sl in range(x.shape[0]):
            hd.append(
                max(
                    directed_hausdorff(x[sl].argmax(0).numpy(), y[sl].argmax(0).numpy())[0],
                    directed_hausdorff(y[sl].argmax(0).numpy(), x[sl].argmax(0).numpy())[0],
                )
            )
        return sum(hd) / len(hd)
    return max(
        directed_hausdorff(x.argmax(0).numpy(), y.argmax(0).numpy())[0],
        directed_hausdorff(y.argmax(0).numpy(), x.argmax(0).numpy())[0],
    )


def hausdorff_distance_95_metric(x: torch.Tensor, y: torch.Tensor, batched: bool = True) -> float:
    """
    Compute 95th percentile of the Hausdorff Distance.

    The Hausdorff distance is computed as the maximum between x to y and y to x.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    batched : bool
        If True, compute Hausdorff distance for each batch and return a tensor with shape (batch_size,).
        If False, compute Hausdorff distance for the whole batch and return a tensor with shape (1,).
        Default is ``True``.

    Returns
    -------
    float
        95th percentile of the Hausdorff Distance.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import hausdorff_distance_95_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> hausdorff_distance_95_metric(datax, datay)
    5.853190166360368
    """
    if batched:
        hd = []
        for sl in range(x.shape[0]):
            hd1 = directed_hausdorff(x[sl].argmax(0).numpy(), y[sl].argmax(0).numpy())[0]
            hd2 = directed_hausdorff(y[sl].argmax(0).numpy(), x[sl].argmax(0).numpy())[0]
            hd.append(np.percentile(np.hstack((hd1, hd2)), 95))
        return sum(hd) / len(hd)
    hd1 = directed_hausdorff(x.argmax(0).numpy(), y.argmax(0).numpy())[0]
    hd2 = directed_hausdorff(y.argmax(0).numpy(), x.argmax(0).numpy())[0]
    return np.percentile(np.hstack((hd1, hd2)), 95)


def iou_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    ignore_empty: bool = True,
    reduction: Union[str, None] = "mean",
) -> float:
    """
    Compute Intersection over Union.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    include_background : bool
        If True, include background in the computation. Default is ``True``.
    ignore_empty : bool
        If True, ignore empty slices. Default is ``True``.
    reduction : str or None
        If None, return a tensor with shape (batch_size,).
        If ``mean``, return the mean of the tensor.
        If ``sum``, return the sum of the tensor.
        Default is ``mean``.

    Returns
    -------
    float
        Intersection over Union.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import iou_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> iou_metric(datax, datay)
    0.33478260040283203
    """
    if not include_background:
        if y.dim() == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            x = x[:, 1:]
            y = y[:, 1:]

    if x.shape != y.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {y.shape} and {x.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(x * y, dim=reduce_axis)

    y_o = torch.sum(x, reduce_axis)
    y_y_o = torch.sum(y, dim=reduce_axis)
    union = y_o + y_y_o - intersection

    max = 1.0 if not ignore_empty else float("nan")
    iou_score = torch.where(union > 0, (intersection) / union, torch.tensor(max, device=y_o.device))
    iou_score, _ = do_metric_reduction(iou_score, reduction=reduction)  # type: ignore

    return iou_score.item()


def precision_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    average="none",
    mdmc_average="samplewise",
    reduction: Union[str, None] = "mean_batch",
) -> float:
    """
    Compute Precision Score.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    include_background : bool
        If True, include background in the computation. Default is ``True``.
    average : str
        If ``none``, return a tensor with shape (batch_size,).
        If ``mean``, return the mean of the tensor.
        If ``sum``, return the sum of the tensor.
        Default is ``mean``.
    mdmc_average : str
        If ``samplewise``, return a tensor with shape (batch_size,).
        If ``global``, return the mean of the tensor.
        If ``none``, return the sum of the tensor.
        Default is ``samplewise``.
    reduction : str or None
        If None, return a tensor with shape (batch_size,).
        If ``mean``, return the mean of the tensor.
        If ``sum``, return the sum of the tensor.
        Default is ``mean_batch``.

    Returns
    -------
    float
        Precision Score.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import precision_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> precision_metric(datax, datay)
    0.5005333423614502
    """
    if not include_background:
        if y.dim() == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            x = x[:, 1:]
            y = y[:, 1:]

    x = x.type(torch.uint8)
    y = y.type(torch.uint8)

    if x.shape != y.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {y.shape} and {x.shape}.")

    # to one hot per class
    pr = []
    for i in range(y.shape[1]):
        precision_score = F.precision(
            one_hot(y[:, i].unsqueeze(1), num_classes=2),
            one_hot(x[:, i].unsqueeze(1), num_classes=2),
            task="binary",
            average=average,
            multidim_average=mdmc_average,
            num_classes=y.shape[1],
        )
        precision_score, _ = do_metric_reduction(precision_score, reduction=reduction)  # type: ignore
        pr.append(precision_score.item())
    return torch.mean(torch.tensor(pr)).item()


def recall_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    average="none",
    mdmc_average="samplewise",
    reduction: Union[str, None] = "mean_batch",
) -> float:
    """
    Compute Recall Score.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    include_background : bool
        If True, include background in the computation. Default is ``True``.
    average : str
        If ``none``, return a tensor with shape (batch_size,).
        If ``mean``, return the mean of the tensor.
        If ``sum``, return the sum of the tensor.
        Default is ``mean``.
    mdmc_average : str
        If ``samplewise``, return a tensor with shape (batch_size,).
        If ``global``, return the mean of the tensor.
        If ``none``, return the sum of the tensor.
        Default is ``samplewise``.
    reduction : str or None
        If None, return a tensor with shape (batch_size,).
        If ``mean``, return the mean of the tensor.
        If ``sum``, return the sum of the tensor.
        Default is ``mean_batch``.

    Returns
    -------
    float
        Recall Score.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import recall_metric
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> recall_metric(datax, datay)
    0.5005333423614502
    """
    if not include_background:
        if y.dim() == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            x = x[:, 1:]
            y = y[:, 1:]

    x = x.type(torch.uint8)
    y = y.type(torch.uint8)

    if x.shape != y.shape:
        raise ValueError(f"Prediction and ground truth should have same shapes, got {y.shape} and {x.shape}.")

    # to one hot per class
    rec = []
    for i in range(y.shape[1]):
        recall_score = F.recall(
            one_hot(y[:, i].unsqueeze(1), num_classes=2),
            one_hot(x[:, i].unsqueeze(1), num_classes=2),
            task="binary",
            average=average,
            multidim_average=mdmc_average,
            num_classes=y.shape[1],
        )
        recall_score, _ = do_metric_reduction(recall_score, reduction=reduction)  # type: ignore
        rec.append(recall_score.item())
    return torch.mean(torch.tensor(rec)).item()


def asd(x, y, voxelspacing=None, connectivity=1):
    """
    Compute Average Symmetric Surface Distance (ASD) between a binary object and its reference.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    voxelspacing : Voxel Spacing. Defaults to ``None``.
    connectivity : int
        Connectivity. Defaults to ``1``.

    Returns
    -------
    float
        Average Symmetric Surface Distance (ASD) between a binary object and its reference.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import asd
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> asd(datax, datay)
    0.5010349308997433
    """
    sd1 = np.mean(surface_distances(y, x, voxelspacing, connectivity))
    sd2 = np.mean(surface_distances(x, y, voxelspacing, connectivity))
    return (sd1 + sd2) / 2.0


def surface_distances(x, y, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their nearest partner surface voxel of a
    binary object in reference.

    Parameters
    ----------
    x : torch.Tensor
        Ground Truth Tensor.
    y : torch.Tensor
        Prediction Tensor.
    voxelspacing : Voxel Spacing. Defaults to ``None``.
    connectivity : int
        Connectivity. Defaults to ``1``.

    Returns
    -------
    np.ndarray
        The distances between the surface voxel of binary objects in result and their nearest partner surface voxel of
        a binary object in reference.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import surface_distances
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> surface_distances(datax, datay)
    array([0., 0., 1., ..., 0., 0., 0.])
    >>> surface_distances(datax, datay).mean()
    0.5083586894950562

    .. note::
        This function is based on the medpy implementation of the Average Symmetric Surface Distance (ASD) metric.
        Source: https://github.com/loli/medpy/blob/master/medpy/metric/binary.py#L458
    """
    x = np.atleast_1d(x.numpy().astype(np.bool))
    y = np.atleast_1d(y.numpy().astype(np.bool))

    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, y.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(y.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(x):
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if 0 == np.count_nonzero(y):
        raise RuntimeError("The second supplied array does not contain any binary object.")

    # extract only 1-pixel border line of objects
    x_border = x ^ binary_erosion(x, structure=footprint, iterations=1)
    y_border = y ^ binary_erosion(y, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipy distance transform is calculated only inside the borders of the foreground objects,
    # therefore the input has to be reversed
    dt = distance_transform_edt(~x_border, sampling=voxelspacing)
    sds = dt[y_border]

    return sds


class SegmentationMetrics:
    """
    Maintains running statistics for a given collection of segmentation metrics.

    Examples
    --------
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import SegmentationMetrics
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import binary_cross_entropy_with_logits_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import dice_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import f1_per_class_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import hausdorff_distance_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import hausdorff_distance_95_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import iou_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import precision_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import recall_metric
    >>> from mridc.collections.segmentation.metrics.segmentation_metrics import asd
    >>> import torch
    >>> datax = torch.randint(0, 2, (3, 2, 100, 100))
    >>> datay = torch.randint(0, 2, (3, 2, 100, 100))
    >>> metric_funcs = {
    ...     "binary_cross_entropy_with_logits": binary_cross_entropy_with_logits_metric,
    ...     "dice": dice_metric,
    ...     "f1_per_class": f1_per_class_metric,
    ...     "hausdorff_distance": hausdorff_distance_metric,
    ...     "hausdorff_distance_95": hausdorff_distance_95_metric,
    ...     "iou": iou_metric,
    ...     "precision": precision_metric,
    ...     "recall": recall_metric,
    ...     "asd": asd,
    ... }
    >>> metrics = SegmentationMetrics(metric_funcs, "output", "method")
    >>> metrics.push(datax, datay)
    >>> metrics.means()
    {'binary_cross_entropy_with_logits': 0.7527344822883606,
     'dice': 0.4993175268173218,
     'f1_per_class': 0.0,
     'hausdorff_distance': 5.8873012632302,
     'hausdorff_distance_95': 5.881561144720858,
     'iou': 0.3327365219593048,
     'precision': 0.5005833506584167,
     'recall': 0.5005833506584167,
     'asd': 0.503373912220483}
    >>> metrics.stddevs()
    {'binary_cross_entropy_with_logits': 0.0,
        'dice': 0.0,
        'f1_per_class': array([0., 0.]),
        'hausdorff_distance': 0.0,
        'hausdorff_distance_95': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'asd': 0.0}
    >>> metrics.__repr__()
    'asd = 0.5034 +/- 0 binary_cross_entropy_with_logits = 0.7527 +/- 0 dice = 0.4993 +/- 0 f1_per_class = 0 +/- 0 \
    hausdorff_distance = 5.887 +/- 0 hausdorff_distance_95 = 5.882 +/- 0 iou = 0.3327 +/- 0 precision = 0.5006 +/- 0 \
    recall = 0.5006 +/- 0\n'
    """

    def __init__(self, metric_funcs, output_path, method):
        """
        Parameters
        ----------
        metric_funcs : dict
            A dict where the keys are metric names and the values are Python functions for evaluating that metric.
        output_path : str
            Path to the output directory.
        method : str
            Segmentation method.
        """
        self.metric_funcs = metric_funcs
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}
        self.output_path = output_path
        self.method = method

    def push(self, x, y):
        """
        Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        x : np.ndarray
            Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.
        y : np.ndarray
            Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.

        Returns
        -------
        dict
            A dict where the keys are metric names and the values are the computed metric scores.
        """
        for metric, func in self.metric_funcs.items():
            score = func(x, y)
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
