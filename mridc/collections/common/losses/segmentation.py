import warnings
from cmath import nan

import monai.metrics as MM
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background=True,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        batch=False,
    ):
        super().__init__()
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, _input, target):
        if target.shape != _input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({_input.shape})")

        _input = torch.softmax(_input, dim=1)
        n_pred_ch = _input.shape[1]

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                _input = _input[:, 1:]

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = torch.arange(2, len(_input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * _input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            _input = torch.pow(_input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(_input, dim=reduce_axis)

        denominator = ground_o + pred_o

        dice = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        loss = 1.0 - dice

        return loss.mean(1), dice.mean(1)


class ContourLoss(nn.Module):
    """https://github.com/rosanajurdi/Perimeter_loss"""

    def forward(self, pred, target):
        pred_contour = self.contour(pred)
        target_contour = self.contour(target)

        loss = (pred_contour.flatten(2).sum(-1) - target_contour.flatten(2).sum(-1)).pow(2)
        loss /= pred.shape[-2] * pred.shape[-1]
        return loss.mean()

    @staticmethod
    def contour(x):
        min_pool = F.max_pool2d(x * -1, kernel_size=(3, 3), stride=1, padding=1) * -1
        max_pool = F.max_pool2d(min_pool, kernel_size=(3, 3), stride=1, padding=1)
        x = F.relu(max_pool - min_pool)
        return x


class MC_CrossEntropy(nn.Module):
    def __init__(
        self,
        num_samples=50,
        weight=None,
        ignore_index=-100,
        reduction="none",
        label_smoothing=0.0,
    ) -> None:
        super().__init__()

        self.mc_samples = num_samples

        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        pred_mean,
        target,
        pred_log_var=None,
    ):
        if self.mc_samples == 1 or pred_log_var is None:
            return self.cross_entropy(pred_mean, target).mean()

        pred_shape = [self.mc_samples, *pred_mean.shape]

        noise = torch.randn(pred_shape, device=pred_mean.device)
        noisy_pred = pred_mean.unsqueeze(0) + torch.sqrt(torch.exp(pred_log_var)).unsqueeze(0) * noise
        del noise
        noisy_pred = noisy_pred.view(-1, *pred_mean.shape[1:])
        tiled_target = target.unsqueeze(0).tile((self.mc_samples,)).view(-1, *target.shape[1:])

        loss = self.cross_entropy(noisy_pred, tiled_target)
        loss = loss.view(self.mc_samples, -1, *pred_mean.shape[-2:]).mean(0)
        return loss.mean()


def hausdorf_distance(preds, target, include_background=True):
    binairy_preds = binerize_segmentation(torch.softmax(preds, dim=1))
    binairy_target = binerize_segmentation(target)
    pred_class = binairy_preds.sum((0, 2, 3))[1:].bool()
    target_class = binairy_target.sum((0, 2, 3))[1:].bool()
    if pred_class.float().sum() < 2 or target_class.float().sum() < 2:
        return torch.tensor([nan], device=preds.device)
    if pred_class.float().sum() == 2 and target_class.float().sum() == 2:
        binairy_preds = binairy_preds[:, :3, ...]
        binairy_target = binairy_target[:, :3, ...]
    return MM.compute_hausdorff_distance(
        binairy_preds,
        binairy_target,
        include_background=include_background,
        percentile=95,
    )


def average_surface_distance(preds, target, include_background=True):
    binairy_preds = binerize_segmentation(torch.softmax(preds, dim=1))
    binairy_target = binerize_segmentation(target)
    class_size = binairy_target.sum((0, 2, 3))[1:].bool()
    if class_size.float().sum() < 2:
        return torch.tensor([nan], device=preds.device)
    if class_size.float().sum() == 2:
        binairy_preds = binairy_preds[:, :3, ...]
        binairy_target = binairy_target[:, :3, ...]
    return MM.compute_average_surface_distance(binairy_preds, binairy_target, include_background=include_background)


def binerize_segmentation(tensor):
    tensor_label = tensor.argmax(1).byte()
    binairy_tensor = torch.zeros_like(tensor, device=tensor.device)
    for i in range(tensor.shape[1]):
        binairy_tensor[:, i, ...] = tensor_label == i
    return binairy_tensor.byte()
