# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from mridc.collections.common.parts.utils import is_none
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.metrics.evaluate import mse, nmse, psnr, ssim
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, DistributedMetricSum
from mridc.collections.segmentation.data.mri_data import JRSMRISliceDataset
from mridc.collections.segmentation.losses.cross_entropy import MC_CrossEntropyLoss
from mridc.collections.segmentation.losses.dice import Dice
from mridc.collections.segmentation.parts.transforms import JRSMRIDataTransforms
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["BaseMRIJointReconstructionSegmentationModel"]


class BaseMRIJointReconstructionSegmentationModel(BaseMRIReconstructionModel, ABC):
    """Base class of all MRI Segmentation and Joint Reconstruction & Segmentation models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.acc = 1.0

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.segmentation_loss_fn = {"cross_entropy": None, "dice": None}
        segmentation_loss_fn = cfg_dict.get("segmentation_loss_fn")
        if "cross_entropy" in segmentation_loss_fn:
            cross_entropy_loss_weight = cfg_dict.get("cross_entropy_loss_weight", None)
            if not is_none(cross_entropy_loss_weight):
                cross_entropy_loss_weight = torch.tensor(cross_entropy_loss_weight)
            else:
                cross_entropy_loss_weight = None
            self.segmentation_loss_fn["cross_entropy"] = MC_CrossEntropyLoss(  # type: ignore
                num_samples=cfg_dict.get("cross_entropy_loss_num_samples", 50),
                ignore_index=cfg_dict.get("cross_entropy_loss_ignore_index", -100),
                reduction=cfg_dict.get("cross_entropy_loss_reduction", "none"),
                label_smoothing=cfg_dict.get("cross_entropy_loss_label_smoothing", 0.0),
                normalization=cfg_dict.get("cross_entropy_loss_normalization", "softmax"),
                weight=cross_entropy_loss_weight,
            )
        if "dice" in segmentation_loss_fn:
            self.segmentation_loss_fn["dice"] = Dice(  # type: ignore
                batched=cfg_dict.get("dice_loss_batched", False),
                include_background=cfg_dict.get("dice_loss_include_background", False),
                squared_pred=cfg_dict.get("dice_loss_squared_pred", False),
                normalization=cfg_dict.get("dice_loss_normalization", "softmax"),
                epsilon=cfg_dict.get("dice_loss_epsilon", 1e-5),
            )

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        cross_entropy_metric_weight = cfg_dict.get("cross_entropy_metric_weight", None)
        if not is_none(cross_entropy_metric_weight):
            cross_entropy_metric_weight = torch.tensor(cross_entropy_metric_weight)
        else:
            cross_entropy_metric_weight = None
        self.cross_entropy_metric = MC_CrossEntropyLoss(  # type: ignore
            num_samples=cfg_dict.get("cross_entropy_metric_num_samples", 50),
            ignore_index=cfg_dict.get("cross_entropy_metric_ignore_index", -100),
            reduction=cfg_dict.get("cross_entropy_metric_reduction", "none"),
            label_smoothing=cfg_dict.get("cross_entropy_metric_label_smoothing", 0.0),
            normalization=cfg_dict.get("cross_entropy_metric_normalization", "softmax"),
            weight=cross_entropy_metric_weight,
        )
        self.dice_coefficient_metric = Dice(  # type: ignore
            batched=cfg_dict.get("dice_coefficient_batched", False),
            include_background=cfg_dict.get("dice_coefficient_include_background", False),
            squared_pred=cfg_dict.get("dice_coefficient_squared_pred", False),
            normalization=cfg_dict.get("dice_coefficient_normalization", "softmax"),
            epsilon=cfg_dict.get("dice_coefficient_epsilon", 1e-5),
        )

        self.CROSS_ENTROPY = DistributedMetricSum()
        self.DICE = DistributedMetricSum()

        self.dice_vals: Dict = defaultdict(dict)
        self.cross_entropy_vals: Dict = defaultdict(dict)

        self.mse_vals_segmentation: Dict = defaultdict(dict)
        self.nmse_vals_segmentation: Dict = defaultdict(dict)
        self.ssim_vals_segmentation: Dict = defaultdict(dict)
        self.psnr_vals_segmentation: Dict = defaultdict(dict)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.train_loss_fn = cfg_dict.get("reconstruction_loss_fn")
            self.val_loss_fn = cfg_dict.get("reconstruction_loss_fn")

            self.mse_vals_reconstruction: Dict = defaultdict(dict)
            self.nmse_vals_reconstruction: Dict = defaultdict(dict)
            self.ssim_vals_reconstruction: Dict = defaultdict(dict)
            self.psnr_vals_reconstruction: Dict = defaultdict(dict)

    def process_segmentation_loss(self, target, pred):
        """
        Processes the segmentation loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        pred: Final prediction.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        loss_dict = {"cross_entropy_loss": 0.0, "dice_loss": 0.0}
        if self.segmentation_loss_fn["cross_entropy"] is not None:
            loss_dict["cross_entropy_loss"] = self.segmentation_loss_fn["cross_entropy"].to(target.device)(
                target, pred
            )
        if self.segmentation_loss_fn["dice"] is not None:
            loss_dict["dice_loss"] = torch.tensor(1.0) - torch.mean(self.segmentation_loss_fn["dice"](target, pred))
        loss_dict["segmentation_loss"] = loss_dict["cross_entropy_loss"] + loss_dict["dice_loss"]
        return loss_dict

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Brain & sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_reconstruction_pred': Initial reconstruction prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target_reconstruction': Target reconstruction.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
            'segmentation': Target segmentation.
                torch.Tensor, shape [batch_size, nr_classes, n_x, n_y] or None
            'fname': File name.
                str
            'slice_num': Slice number.
                int
            'acc': Acceleration factor of the sampling mask.
                float
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acc,
        ) = batch

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.consecutive_slices > 1:
            target_segmentation = target_segmentation.reshape(  # type: ignore
                target_segmentation.shape[0] * target_segmentation.shape[1],  # type: ignore
                target_segmentation.shape[2],  # type: ignore
                target_segmentation.shape[3],  # type: ignore
                target_segmentation.shape[4],  # type: ignore
            )

        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        if self.use_reconstruction_module:
            if self.accumulate_estimates:
                try:
                    pred_reconstruction = next(pred_reconstruction)
                except StopIteration:
                    pass
                reconstruction_loss = sum(
                    self.process_loss(target_reconstruction, pred_reconstruction, _loss_fn=self.val_loss_fn, mask=None)
                )
            else:
                reconstruction_loss = self.process_loss(
                    target_reconstruction, pred_reconstruction, _loss_fn=self.train_loss_fn, mask=None
                )
            train_loss = 1e-4 * segmentation_loss + 1e-4 * reconstruction_loss
        else:
            train_loss = segmentation_loss

        self.acc = r if r != 0 else acc
        tensorboard_logs = {
            f"train_loss_{self.acc}x": train_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """
        Performs a validation step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Brain & sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_reconstruction_pred': Initial reconstruction prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target_reconstruction': Target reconstruction.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
            'segmentation': Target segmentation.
                torch.Tensor, shape [batch_size, nr_classes, n_x, n_y] or None
            'fname': File name.
                str
            'slice_num': Slice number.
                int
            'acc': Acceleration factor of the sampling mask.
                float
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acc,
        ) = batch

        key = f"{fname[0]}_images_idx_{int(slice_idx)}"  # type: ignore

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.consecutive_slices > 1:
            target_reconstruction = target_reconstruction.reshape(  # type: ignore
                target_reconstruction.shape[0] * target_reconstruction.shape[1],  # type: ignore
                target_reconstruction.shape[2],  # type: ignore
                target_reconstruction.shape[3],  # type: ignore
            )
            target_segmentation = target_segmentation.reshape(  # type: ignore
                target_segmentation.shape[0] * target_segmentation.shape[1],  # type: ignore
                target_segmentation.shape[2],  # type: ignore
                target_segmentation.shape[3],  # type: ignore
                target_segmentation.shape[4],  # type: ignore
            )
        target_reconstruction = torch.abs(target_reconstruction).detach().cpu()
        target_reconstruction = target_reconstruction / target_reconstruction.max()  # type: ignore

        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        if self.use_reconstruction_module:
            if self.accumulate_estimates:
                try:
                    pred_reconstruction = next(pred_reconstruction)
                except StopIteration:
                    pass

                reconstruction_loss = sum(
                    self.process_loss(target_reconstruction, pred_reconstruction, _loss_fn=self.val_loss_fn, mask=None)
                )

                # Cascades
                if isinstance(pred_reconstruction, list):
                    pred_reconstruction = pred_reconstruction[-1]

                # Time-steps
                if isinstance(pred_reconstruction, list):
                    pred_reconstruction = pred_reconstruction[-1]
            else:
                reconstruction_loss = self.process_loss(
                    target_reconstruction, pred_reconstruction, _loss_fn=self.train_loss_fn, mask=None
                )

            val_loss = 1e-4 * segmentation_loss + 1e-4 * reconstruction_loss

            if self.consecutive_slices > 1:
                pred_reconstruction = pred_reconstruction.reshape(
                    pred_reconstruction.shape[0] * pred_reconstruction.shape[1],
                    pred_reconstruction.shape[2],
                    pred_reconstruction.shape[3],
                )

            output_reconstruction = torch.abs(pred_reconstruction).detach().cpu()
            output_reconstruction = output_reconstruction / output_reconstruction.max()
            error_reconstruction = torch.abs(target_reconstruction - output_reconstruction)
            self.log_image(f"{key}/reconstruction/target", target_reconstruction)
            self.log_image(f"{key}/reconstruction/prediction", output_reconstruction)
            self.log_image(f"{key}/reconstruction/error", error_reconstruction)

            target_reconstruction = target_reconstruction.numpy()  # type: ignore
            output_reconstruction = output_reconstruction.numpy()
            self.mse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                mse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.nmse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                nmse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.ssim_vals_reconstruction[fname][slice_idx] = torch.tensor(
                ssim(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
            self.psnr_vals_reconstruction[fname][slice_idx] = torch.tensor(
                psnr(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
        else:
            val_loss = segmentation_loss

        output_segmentation = torch.abs(pred_segmentation).detach().cpu()
        output_segmentation = output_segmentation / output_segmentation.max()
        target_segmentation = torch.abs(target_segmentation).detach().cpu()
        target_segmentation = target_segmentation / target_segmentation.max()  # type: ignore
        error_segmentation = torch.abs(target_segmentation - output_segmentation)

        # remove all zero classes to avoid division by zero
        non_zero_target_classes = []
        non_zero_output_classes = []
        for class_idx in range(target_segmentation.shape[0]):  # type: ignore
            if not torch.all(target_segmentation[class_idx] == 0):  # type: ignore
                non_zero_target_classes.append(target_segmentation[class_idx])  # type: ignore
                non_zero_output_classes.append(output_segmentation[class_idx])
        target_segmentation = torch.stack(non_zero_target_classes, dim=0)
        output_segmentation = torch.stack(non_zero_output_classes, dim=0)

        mses = []
        nmses = []
        ssims = []
        psnrs = []

        for class_idx in range(output_segmentation.shape[1]):
            target_segmentation_class = target_segmentation[:, class_idx]  # type: ignore
            output_segmentation_class = output_segmentation[:, class_idx]
            error_segmentation_class = error_segmentation[:, class_idx]

            self.log_image(
                f"{key}/segmentation_classes/target_class_{class_idx}",
                target_segmentation_class,  # type: ignore
            )
            self.log_image(f"{key}/segmentation_classes/prediction_class_{class_idx}", output_segmentation_class)
            self.log_image(f"{key}/segmentation_classes/error_class_{class_idx}", error_segmentation_class)

            target_segmented_class = target_reconstruction * target_segmentation_class
            target_segmented_class = target_segmented_class / target_segmented_class.max()
            output_segmented_class = target_reconstruction * output_segmentation_class
            output_segmented_class = output_segmented_class / output_segmented_class.max()
            error_segmented_class = torch.abs(target_reconstruction - error_segmentation_class)
            error_segmented_class = error_segmented_class / error_segmented_class.max()

            self.log_image(f"{key}/segmentations/target_class_{class_idx}", target_segmented_class)
            self.log_image(f"{key}/segmentations/prediction_class_{class_idx}", output_segmented_class)
            self.log_image(f"{key}/segmentations/error_class_{class_idx}", error_segmented_class)

            target_segmented_class = target_segmented_class.numpy()
            output_segmented_class = output_segmented_class.numpy()

            mses.append(torch.tensor(mse(target_segmented_class, output_segmented_class)).view(1))
            nmses.append(torch.tensor(nmse(target_segmented_class, output_segmented_class)).view(1))
            ssims.append(
                torch.tensor(
                    ssim(
                        target_segmented_class,
                        output_segmented_class,
                        maxval=output_segmented_class.max() - output_segmented_class.min(),
                    )
                ).view(1)
            )
            psnrs.append(
                torch.tensor(
                    psnr(
                        target_segmented_class,
                        output_segmented_class,
                        maxval=output_segmented_class.max() - output_segmented_class.min(),
                    )
                ).view(1)
            )

        target_segmentation = target_segmentation.float()  # type: ignore
        output_segmentation = output_segmentation.float()
        self.cross_entropy_vals[fname][slice_idx] = torch.mean(
            self.cross_entropy_metric(target_segmentation, output_segmentation)
        )
        self.dice_vals[fname][slice_idx] = torch.mean(
            self.dice_coefficient_metric(target_segmentation, output_segmentation)
        )

        self.mse_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(mses))
        self.nmse_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(nmses))
        self.ssim_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(ssims))
        self.psnr_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(psnrs))

        return {"val_loss": val_loss}

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Brain & sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_reconstruction_pred': Initial reconstruction prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target_reconstruction': Target reconstruction.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
            'segmentation': Target segmentation.
                torch.Tensor, shape [batch_size, nr_classes, n_x, n_y] or None
            'fname': File name.
                str
            'slice_num': Slice number.
                int
            'acc': Acceleration factor of the sampling mask.
                float
        batch_idx: Batch index.
            int

        Returns
        -------
        fname: File name.
            str
        slice_num: Slice number.
            int
        predictions: Stacked predictions for reconstruction (if reconstruction module is used) and segmentation.
            tuple of torch.Tensor, shape [batch_size, n_x, n_y, 2] and [batch_size, nr_classes, n_x, n_y]
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acc,
        ) = batch

        key = f"{fname[0]}_images_idx_{int(slice_idx)}"  # type: ignore

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.consecutive_slices > 1:
            target_reconstruction = target_reconstruction.reshape(  # type: ignore
                target_reconstruction.shape[0] * target_reconstruction.shape[1],  # type: ignore
                target_reconstruction.shape[2],  # type: ignore
                target_reconstruction.shape[3],  # type: ignore
            )
            target_segmentation = target_segmentation.reshape(  # type: ignore
                target_segmentation.shape[0] * target_segmentation.shape[1],  # type: ignore
                target_segmentation.shape[2],  # type: ignore
                target_segmentation.shape[3],  # type: ignore
                target_segmentation.shape[4],  # type: ignore
            )
        target_reconstruction = torch.abs(target_reconstruction).detach().cpu()
        target_reconstruction = target_reconstruction / target_reconstruction.max()  # type: ignore

        if self.use_reconstruction_module:
            if self.accumulate_estimates:
                try:
                    pred_reconstruction = next(pred_reconstruction)
                except StopIteration:
                    pass

                # Cascades
                if isinstance(pred_reconstruction, list):
                    pred_reconstruction = pred_reconstruction[-1]

                # Time-steps
                if isinstance(pred_reconstruction, list):
                    pred_reconstruction = pred_reconstruction[-1]

            if self.consecutive_slices > 1:
                pred_reconstruction = pred_reconstruction.reshape(
                    pred_reconstruction.shape[0] * pred_reconstruction.shape[1],
                    pred_reconstruction.shape[2],
                    pred_reconstruction.shape[3],
                )

            output_reconstruction = torch.abs(pred_reconstruction).detach().cpu()
            output_reconstruction = output_reconstruction / output_reconstruction.max()
            error_reconstruction = torch.abs(target_reconstruction - output_reconstruction)
            self.log_image(f"{key}/reconstruction/target", target_reconstruction)
            self.log_image(f"{key}/reconstruction/prediction", output_reconstruction)
            self.log_image(f"{key}/reconstruction/error", error_reconstruction)

            target_reconstruction = target_reconstruction.numpy()  # type: ignore
            output_reconstruction = output_reconstruction.numpy()
            self.mse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                mse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.nmse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                nmse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.ssim_vals_reconstruction[fname][slice_idx] = torch.tensor(
                ssim(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
            self.psnr_vals_reconstruction[fname][slice_idx] = torch.tensor(
                psnr(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)

        output_segmentation = torch.abs(pred_segmentation).detach().cpu()
        output_segmentation = output_segmentation / output_segmentation.max()
        target_segmentation = torch.abs(target_segmentation).detach().cpu()
        target_segmentation = target_segmentation / target_segmentation.max()  # type: ignore
        error_segmentation = torch.abs(target_segmentation - output_segmentation)

        # remove all zero classes to avoid division by zero
        non_zero_target_classes = []
        non_zero_output_classes = []
        for class_idx in range(target_segmentation.shape[0]):  # type: ignore
            if not torch.all(target_segmentation[class_idx] == 0):  # type: ignore
                non_zero_target_classes.append(target_segmentation[class_idx])  # type: ignore
                non_zero_output_classes.append(output_segmentation[class_idx])
        target_segmentation = torch.stack(non_zero_target_classes, dim=0)
        output_segmentation = torch.stack(non_zero_output_classes, dim=0)

        mses = []
        nmses = []
        ssims = []
        psnrs = []

        for class_idx in range(output_segmentation.shape[1]):
            target_segmentation_class = target_segmentation[:, class_idx]  # type: ignore
            output_segmentation_class = output_segmentation[:, class_idx]
            error_segmentation_class = error_segmentation[:, class_idx]

            self.log_image(
                f"{key}/segmentation_classes/target_class_{class_idx}",
                target_segmentation_class,  # type: ignore
            )
            self.log_image(f"{key}/segmentation_classes/prediction_class_{class_idx}", output_segmentation_class)
            self.log_image(f"{key}/segmentation_classes/error_class_{class_idx}", error_segmentation_class)

            target_segmented_class = target_reconstruction * target_segmentation_class
            target_segmented_class = target_segmented_class / target_segmented_class.max()
            output_segmented_class = target_reconstruction * output_segmentation_class
            output_segmented_class = output_segmented_class / output_segmented_class.max()
            error_segmented_class = torch.abs(target_reconstruction - error_segmentation_class)
            error_segmented_class = error_segmented_class / error_segmented_class.max()

            self.log_image(f"{key}/segmentations/target_class_{class_idx}", target_segmented_class)
            self.log_image(f"{key}/segmentations/prediction_class_{class_idx}", output_segmented_class)
            self.log_image(f"{key}/segmentations/error_class_{class_idx}", error_segmented_class)

            target_segmented_class = target_segmented_class.numpy()
            output_segmented_class = output_segmented_class.numpy()

            mses.append(torch.tensor(mse(target_segmented_class, output_segmented_class)).view(1))
            nmses.append(torch.tensor(nmse(target_segmented_class, output_segmented_class)).view(1))
            ssims.append(
                torch.tensor(
                    ssim(
                        target_segmented_class,
                        output_segmented_class,
                        maxval=output_segmented_class.max() - output_segmented_class.min(),
                    )
                ).view(1)
            )
            psnrs.append(
                torch.tensor(
                    psnr(
                        target_segmented_class,
                        output_segmented_class,
                        maxval=output_segmented_class.max() - output_segmented_class.min(),
                    )
                ).view(1)
            )

        target_segmentation = target_segmentation.float()  # type: ignore
        output_segmentation = output_segmentation.float()
        self.cross_entropy_vals[fname][slice_idx] = torch.mean(
            self.cross_entropy_metric(target_segmentation, output_segmentation)
        )
        self.dice_vals[fname][slice_idx] = torch.mean(
            self.dice_coefficient_metric(target_segmentation, output_segmentation)
        )

        self.mse_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(mses))
        self.nmse_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(nmses))
        self.ssim_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(ssims))
        self.psnr_vals_segmentation[fname][slice_idx] = torch.mean(torch.stack(psnrs))

        predictions = (
            (pred_segmentation.detach().cpu().numpy(), pred_reconstruction.detach().cpu().numpy())
            if self.use_reconstruction_module
            else (pred_segmentation.detach().cpu().numpy(),)
        )

        return (str(fname[0]), slice_idx, predictions)  # type: ignore

    def train_epoch_end(self, outputs):
        """
        Called at the end of train epoch to aggregate the loss values.

        Parameters
        ----------
        outputs: List of outputs from the train step.
            list of dicts
        """
        self.log("train_loss", torch.stack([x["train_loss"] for x in outputs]).mean())
        self.log(
            f"train_loss_{self.acc}x",
            torch.stack([x[f"train_loss_{self.acc}x"] for x in outputs]).mean(),
        )

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the validation batches.
            list of dicts

        Returns
        -------
        metrics: Dictionary of metrics.
            dict
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

        # Log metrics.
        cross_entropy_vals = defaultdict(dict)
        dice_vals = defaultdict(dict)

        mse_vals_segmentation = defaultdict(dict)
        nmse_vals_segmentation = defaultdict(dict)
        ssim_vals_segmentation = defaultdict(dict)
        psnr_vals_segmentation = defaultdict(dict)

        for k in self.cross_entropy_vals.keys():
            cross_entropy_vals[k].update(self.cross_entropy_vals[k])
        for k in self.dice_vals.keys():
            dice_vals[k].update(self.dice_vals[k])

        for k in self.mse_vals_segmentation.keys():
            mse_vals_segmentation[k].update(self.mse_vals_segmentation[k])
        for k in self.nmse_vals_segmentation.keys():
            nmse_vals_segmentation[k].update(self.nmse_vals_segmentation[k])
        for k in self.ssim_vals_segmentation.keys():
            ssim_vals_segmentation[k].update(self.ssim_vals_segmentation[k])
        for k in self.psnr_vals_segmentation.keys():
            psnr_vals_segmentation[k].update(self.psnr_vals_segmentation[k])

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k in self.mse_vals_reconstruction.keys():
                mse_vals_reconstruction[k].update(self.mse_vals_reconstruction[k])
            for k in self.nmse_vals_reconstruction.keys():
                nmse_vals_reconstruction[k].update(self.nmse_vals_reconstruction[k])
            for k in self.ssim_vals_reconstruction.keys():
                ssim_vals_reconstruction[k].update(self.ssim_vals_reconstruction[k])
            for k in self.psnr_vals_R2star.keys():
                psnr_vals_reconstruction[k].update(self.psnr_vals_reconstruction[k])

        # apply means across image volumes
        metrics = {
            "Cross_Entropy": {
                "segmentation": 0,
            },
            "DICE": {
                "segmentation": 0,
            },
            "MSE": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "NMSE": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "SSIM": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "PSNR": {
                "reconstruction": 0,
                "segmentation": 0,
            },
        }
        local_examples = 0
        for fname in mse_vals_segmentation:
            local_examples += 1

            metrics["Cross_Entropy"]["segmentation"] = metrics["Cross_Entropy"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics["DICE"]["segmentation"] = metrics["DICE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

            metrics["MSE"]["segmentation"] = metrics["MSE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_segmentation[fname].items()])
            )
            metrics["NMSE"]["segmentation"] = metrics["NMSE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_segmentation[fname].items()])
            )
            metrics["SSIM"]["segmentation"] = metrics["SSIM"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_segmentation[fname].items()])
            )
            metrics["PSNR"]["segmentation"] = metrics["PSNR"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_segmentation[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics["MSE"]["reconstruction"] = metrics["MSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics["NMSE"]["reconstruction"] = metrics["NMSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics["SSIM"]["reconstruction"] = metrics["SSIM"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics["PSNR"]["reconstruction"] = metrics["PSNR"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in psnr_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics["Cross_Entropy"]["segmentation"] = self.CROSS_ENTROPY(metrics["Cross_Entropy"]["segmentation"])
        metrics["DICE"]["segmentation"] = self.DICE(metrics["DICE"]["segmentation"])

        metrics["MSE"]["segmentation"] = self.MSE(metrics["MSE"]["segmentation"])
        metrics["NMSE"]["segmentation"] = self.NMSE(metrics["NMSE"]["segmentation"])
        metrics["SSIM"]["segmentation"] = self.SSIM(metrics["SSIM"]["segmentation"])
        metrics["PSNR"]["segmentation"] = self.PSNR(metrics["PSNR"]["segmentation"])

        if self.use_reconstruction_module:
            metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
            metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
            metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
            metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}_Segmentation", value["segmentation"] / tot_examples)
            if self.use_reconstruction_module:
                self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the test batches.
            list of dicts

        Returns
        -------
        Saves the reconstructed images to .h5 files.
        """
        # Log metrics.
        cross_entropy_vals = defaultdict(dict)
        dice_vals = defaultdict(dict)

        mse_vals_segmentation = defaultdict(dict)
        nmse_vals_segmentation = defaultdict(dict)
        ssim_vals_segmentation = defaultdict(dict)
        psnr_vals_segmentation = defaultdict(dict)

        for k in self.cross_entropy_vals.keys():
            cross_entropy_vals[k].update(self.cross_entropy_vals[k])
        for k in self.dice_vals.keys():
            dice_vals[k].update(self.dice_vals[k])

        for k in self.mse_vals_segmentation.keys():
            mse_vals_segmentation[k].update(self.mse_vals_segmentation[k])
        for k in self.nmse_vals_segmentation.keys():
            nmse_vals_segmentation[k].update(self.nmse_vals_segmentation[k])
        for k in self.ssim_vals_segmentation.keys():
            ssim_vals_segmentation[k].update(self.ssim_vals_segmentation[k])
        for k in self.psnr_vals_segmentation.keys():
            psnr_vals_segmentation[k].update(self.psnr_vals_segmentation[k])

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k in self.mse_vals_reconstruction.keys():
                mse_vals_reconstruction[k].update(self.mse_vals_reconstruction[k])
            for k in self.nmse_vals_reconstruction.keys():
                nmse_vals_reconstruction[k].update(self.nmse_vals_reconstruction[k])
            for k in self.ssim_vals_reconstruction.keys():
                ssim_vals_reconstruction[k].update(self.ssim_vals_reconstruction[k])
            for k in self.psnr_vals_R2star.keys():
                psnr_vals_reconstruction[k].update(self.psnr_vals_reconstruction[k])

        # apply means across image volumes
        metrics = {
            "Cross_Entropy": {
                "segmentation": 0,
            },
            "DICE": {
                "segmentation": 0,
            },
            "MSE": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "NMSE": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "SSIM": {
                "reconstruction": 0,
                "segmentation": 0,
            },
            "PSNR": {
                "reconstruction": 0,
                "segmentation": 0,
            },
        }
        local_examples = 0
        for fname in mse_vals_segmentation:
            local_examples += 1

            metrics["Cross_Entropy"]["segmentation"] = metrics["Cross_Entropy"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics["DICE"]["segmentation"] = metrics["DICE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

            metrics["MSE"]["segmentation"] = metrics["MSE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_segmentation[fname].items()])
            )
            metrics["NMSE"]["segmentation"] = metrics["NMSE"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_segmentation[fname].items()])
            )
            metrics["SSIM"]["segmentation"] = metrics["SSIM"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_segmentation[fname].items()])
            )
            metrics["PSNR"]["segmentation"] = metrics["PSNR"]["segmentation"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_segmentation[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics["MSE"]["reconstruction"] = metrics["MSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics["NMSE"]["reconstruction"] = metrics["NMSE"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics["SSIM"]["reconstruction"] = metrics["SSIM"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics["PSNR"]["reconstruction"] = metrics["PSNR"]["reconstruction"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in psnr_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics["Cross_Entropy"]["segmentation"] = self.CROSS_ENTROPY(metrics["Cross_Entropy"]["segmentation"])
        metrics["DICE"]["segmentation"] = self.DICE(metrics["DICE"]["segmentation"])

        metrics["MSE"]["segmentation"] = self.MSE(metrics["MSE"]["segmentation"])
        metrics["NMSE"]["segmentation"] = self.NMSE(metrics["NMSE"]["segmentation"])
        metrics["SSIM"]["segmentation"] = self.SSIM(metrics["SSIM"]["segmentation"])
        metrics["PSNR"]["segmentation"] = self.PSNR(metrics["PSNR"]["segmentation"])

        if self.use_reconstruction_module:
            metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
            metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
            metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
            metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}_Segmentation", value["segmentation"] / tot_examples)
            if self.use_reconstruction_module:
                self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples)

        predictions = defaultdict(list)
        for fname, slice_num, output in outputs:
            predictions[fname].append((slice_num, output))

        for fname in predictions:
            predictions[fname] = np.stack([out for _, out in sorted(predictions[fname])])

        out_dir = Path(os.path.join(self.logger.log_dir, "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)
        for fname, preds in predictions.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("segmentation", data=preds[0])
                if self.use_reconstruction_module:
                    hf.create_dataset("reconstruction", data=preds[1])

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg: Configuration file.
            dict

        Returns
        -------
        dataloader: DataLoader.
            torch.utils.data.DataLoader
        """
        mask_root = cfg.get("mask_path")
        mask_args = cfg.get("mask_args")
        shift_mask = mask_args.get("shift_mask")
        mask_type = mask_args.get("type")

        mask_func = None  # type: ignore
        mask_center_scale = 0.02

        if is_none(mask_root) and not is_none(mask_type):
            accelerations = mask_args.get("accelerations")
            center_fractions = mask_args.get("center_fractions")
            mask_center_scale = mask_args.get("scale")

            mask_func = (
                [
                    create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) >= 2
                else [create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            )

        dataset = JRSMRISliceDataset(
            root=cfg.get("data_path"),
            sense_root=cfg.get("sense_path"),
            mask_root=cfg.get("mask_path"),
            sample_rate=cfg.get("sample_rate", 1.0),
            volume_sample_rate=cfg.get("volume_sample_rate", None),
            use_dataset_cache=cfg.get("use_dataset_cache", None),
            dataset_cache_file=cfg.get("dataset_cache_file", None),
            num_cols=cfg.get("num_cols", None),
            consecutive_slices=cfg.get("consecutive_slices", 1),
            segmentation_classes=cfg.get("segmentation_classes", 2),
            data_saved_per_slice=cfg.get("data_saved_per_slice", False),
            transform=JRSMRIDataTransforms(
                apply_prewhitening=cfg.get("apply_prewhitening", False),
                prewhitening_scale_factor=cfg.get("prewhitening_scale_factor", 1.0),
                prewhitening_patch_start=cfg.get("prewhitening_patch_start", 10),
                prewhitening_patch_length=cfg.get("prewhitening_patch_length", 30),
                apply_gcc=cfg.get("apply_gcc", False),
                gcc_virtual_coils=cfg.get("gcc_virtual_coils", 10),
                gcc_calib_lines=cfg.get("gcc_calib_lines", 24),
                gcc_align_data=cfg.get("gcc_align_data", True),
                coil_combination_method=cfg.get("coil_combination_method", "SENSE"),
                dimensionality=cfg.get("dimensionality", 2),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                half_scan_percentage=cfg.get("half_scan_percentage", 0.0),
                remask=cfg.get("remask", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", True),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", False),
                max_norm=cfg.get("max_norm", True),
                fft_centered=cfg.get("fft_centered", False),
                fft_normalization=cfg.get("fft_normalization", "ortho"),
                spatial_dims=cfg.get("spatial_dims", [-2, -1]),
                coil_dim=cfg.get("coil_dim", 0),
                consecutive_slices=cfg.get("consecutive_slices", 1),
                use_seed=cfg.get("use_seed", True),
            ),
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size"),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
