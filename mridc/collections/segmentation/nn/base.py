# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import mridc.collections.segmentation.losses as segmentation_losses
from mridc.collections.common.data import subsample
from mridc.collections.common.nn.base import BaseMRIModel, DistributedMetricSum
from mridc.collections.common.parts import utils
from mridc.collections.segmentation.data import mri_segmentation_loader
from mridc.collections.segmentation.parts import transforms

__all__ = ["BaseMRISegmentationModel"]


class BaseMRISegmentationModel(BaseMRIModel, ABC):  # type: ignore
    """
    Base class of all MRI Segmentation models.

    Parameters
    ----------
    cfg: DictConfig
        The configuration file.
    trainer: Trainer
        The PyTorch Lightning trainer.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.acc = 1  # fixed acceleration factor to ensure acc is not None

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.input_channels = cfg_dict.get("segmentation_module_input_channels", 2)
        if self.input_channels == 0:
            raise ValueError("Segmentation module input channels cannot be 0.")
        if self.input_channels > 2:
            raise ValueError(f"Segmentation module input channels must be either 1 or 2. Found: {self.input_channels}")

        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        self.segmentation_loss_fn = {"cross_entropy": None, "dice": None}
        self.total_segmentation_loss_weight = cfg_dict.get("total_segmentation_loss_weight", 1.0)
        segmentation_loss_fn = cfg_dict.get("segmentation_loss_fn")
        if "cross_entropy" in segmentation_loss_fn:
            cross_entropy_loss_weight = cfg_dict.get("cross_entropy_loss_weight", None)
            if not utils.is_none(cross_entropy_loss_weight):
                cross_entropy_loss_weight = torch.tensor(cross_entropy_loss_weight)
            else:
                cross_entropy_loss_weight = None
            self.segmentation_loss_fn["cross_entropy"] = segmentation_losses.cross_entropy.CrossEntropyLoss(
                # type: ignore
                num_samples=cfg_dict.get("cross_entropy_loss_num_samples", 50),
                ignore_index=cfg_dict.get("cross_entropy_loss_ignore_index", -100),
                reduction=cfg_dict.get("cross_entropy_loss_reduction", "none"),
                label_smoothing=cfg_dict.get("cross_entropy_loss_label_smoothing", 0.0),
                weight=cross_entropy_loss_weight,
            )
            self.cross_entropy_loss_weighting_factor = cfg_dict.get("cross_entropy_loss_weighting_factor", 1.0)
        if "dice" in segmentation_loss_fn:
            self.segmentation_loss_fn["dice"] = segmentation_losses.dice.Dice(  # type: ignore
                include_background=cfg_dict.get("dice_loss_include_background", False),
                to_onehot_y=cfg_dict.get("dice_loss_to_onehot_y", False),
                sigmoid=cfg_dict.get("dice_loss_sigmoid", True),
                softmax=cfg_dict.get("dice_loss_softmax", False),
                other_act=cfg_dict.get("dice_loss_other_act", None),
                squared_pred=cfg_dict.get("dice_loss_squared_pred", False),
                jaccard=cfg_dict.get("dice_loss_jaccard", False),
                flatten=cfg_dict.get("dice_loss_flatten", False),
                reduction=cfg_dict.get("dice_loss_reduction", "mean"),
                smooth_nr=cfg_dict.get("dice_loss_smooth_nr", 1e-5),
                smooth_dr=cfg_dict.get("dice_loss_smooth_dr", 1e-5),
                batch=cfg_dict.get("dice_loss_batch", False),
            )
            self.dice_loss_weighting_factor = cfg_dict.get("dice_loss_weighting_factor", 1.0)

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        cross_entropy_metric_weight = cfg_dict.get("cross_entropy_metric_weight", None)
        if not utils.is_none(cross_entropy_metric_weight):
            cross_entropy_metric_weight = torch.tensor(cross_entropy_metric_weight)
        else:
            cross_entropy_metric_weight = None
        self.cross_entropy_metric = segmentation_losses.cross_entropy.CrossEntropyLoss(  # type: ignore
            num_samples=cfg_dict.get("cross_entropy_metric_num_samples", 50),
            ignore_index=cfg_dict.get("cross_entropy_metric_ignore_index", -100),
            reduction=cfg_dict.get("cross_entropy_metric_reduction", "none"),
            label_smoothing=cfg_dict.get("cross_entropy_metric_label_smoothing", 0.0),
            weight=cross_entropy_metric_weight,
        )
        self.dice_coefficient_metric = segmentation_losses.dice.Dice(  # type: ignore
            include_background=cfg_dict.get("dice_metric_include_background", False),
            to_onehot_y=cfg_dict.get("dice_metric_to_onehot_y", False),
            sigmoid=cfg_dict.get("dice_metric_sigmoid", False),
            softmax=cfg_dict.get("dice_metric_softmax", True),
            other_act=cfg_dict.get("dice_metric_other_act", None),
            squared_pred=cfg_dict.get("dice_metric_squared_pred", False),
            jaccard=cfg_dict.get("dice_metric_jaccard", False),
            flatten=cfg_dict.get("dice_metric_flatten", False),
            reduction=cfg_dict.get("dice_metric_reduction", "mean"),
            smooth_nr=cfg_dict.get("dice_metric_smooth_nr", 1e-5),
            smooth_dr=cfg_dict.get("dice_metric_smooth_dr", 1e-5),
            batch=cfg_dict.get("dice_metric_batch", True),
        )

        self.segmentation_classes_thresholds = cfg_dict.get("segmentation_classes_thresholds", None)

        self.CROSS_ENTROPY = DistributedMetricSum()
        self.cross_entropy_vals: Dict = defaultdict(dict)

        self.DICE = DistributedMetricSum()
        self.dice_vals: Dict = defaultdict(dict)

        self.TotExamples = DistributedMetricSum()

    def process_segmentation_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> Dict:
        """
        Processes the segmentation loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, nr_classes, n_x, n_y].
        prediction : torch.Tensor
            Prediction of shape [batch_size, nr_classes, n_x, n_y].

        Returns
        -------
        Dict
            Dictionary containing the (multiple) loss values. For example, if the cross entropy loss and the dice loss
            are used, the dictionary will contain the keys ``cross_entropy_loss``, ``dice_loss``, and
            (combined) ``segmentation_loss``.
        """
        loss_dict = {"cross_entropy_loss": 0.0, "dice_loss": 0.0}
        if self.segmentation_loss_fn["cross_entropy"] is not None:
            loss_dict["cross_entropy_loss"] = (
                self.segmentation_loss_fn["cross_entropy"].cpu()(
                    target.argmax(1).detach().cpu(), prediction.detach().cpu()
                )
                * self.cross_entropy_loss_weighting_factor
            )
        if self.segmentation_loss_fn["dice"] is not None:
            _, loss_dict["dice_loss"] = self.segmentation_loss_fn["dice"](target, prediction)  # noqa: F841
            loss_dict["dice_loss"] = loss_dict["dice_loss"] * self.dice_loss_weighting_factor
        loss_dict["segmentation_loss"] = loss_dict["cross_entropy_loss"] + loss_dict["dice_loss"]
        return loss_dict

    @staticmethod
    def process_inputs(
        y: Union[list, torch.Tensor], mask: Union[list, torch.Tensor], init_pred: Union[list, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the inputs
        are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If only one
        acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        y : Union[list, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : Union[list, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        init_pred : Union[list, torch.Tensor]
            Initial prediction of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].


        Returns
        -------
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        init_pred : torch.Tensor
            Initial prediction of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        r : int
            Random index used to select the acceleration.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
            init_pred = init_pred[r]
        else:
            r = 0
        return y, mask, init_pred, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # noqa: C901
        """
        Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
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
            'fname' : str
                File name.
            'slice_num' : int
                Slice number.
            'acc' : float
                Acceleration factor of the sampling mask.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            kspace,  # noqa: F841
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,  # noqa: F841
            slice_idx,  # noqa: F841
            acc,
        ) = batch

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)

        pred_segmentation = self.forward(y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction)

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )

        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        train_loss = self.total_segmentation_loss_weight * segmentation_loss

        self.acc = r if r != 0 else acc  # type: ignore
        tensorboard_logs = {
            f"train_loss_{self.acc}x": train_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:  # noqa: C901
        """
        Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
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
            'fname' : str
                File name.
            'slice_num' : int
                Slice number.
            'acc' : float
                Acceleration factor of the sampling mask.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            kspace,  # noqa: F841
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acc,  # noqa: F841
        ) = batch

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)  # noqa: F841

        pred_segmentation = self.forward(y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction)

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )

        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        if isinstance(pred_segmentation, list):
            pred_segmentation = pred_segmentation[-1]

        val_loss = self.total_segmentation_loss_weight * segmentation_loss

        # normalize for visualization
        if not utils.is_none(self.segmentation_classes_thresholds):
            for class_idx, class_threshold in enumerate(self.segmentation_classes_thresholds):
                if not utils.is_none(class_threshold):
                    target_segmentation[:, class_idx] = (  # type: ignore
                        target_segmentation[:, class_idx] > class_threshold  # type: ignore
                    )
                    pred_segmentation[:, class_idx] = pred_segmentation[:, class_idx] > class_threshold

        if self.log_images:
            slice_idx = int(slice_idx)
            key = f"{fname[0]}_images_idx_{slice_idx}"  # type: ignore

            target_reconstruction = (
                torch.abs(target_reconstruction / torch.max(torch.abs(target_reconstruction))).detach().cpu()
            )
            self.log_image(f"{key}/reconstruction/target", target_reconstruction)

            for class_idx in range(pred_segmentation.shape[1]):  # type: ignore
                target_image_segmentation_class = target_segmentation[:, class_idx]  # type: ignore
                output_image_segmentation_class = pred_segmentation[:, class_idx]

                self.log_image(
                    f"{key}/segmentation_classes/target_class_{class_idx}",
                    target_image_segmentation_class,  # type: ignore
                )
                self.log_image(
                    f"{key}/segmentation_classes/prediction_class_{class_idx}", output_image_segmentation_class
                )
                self.log_image(
                    f"{key}/segmentation_classes/error_1_class_{class_idx}",
                    torch.abs(target_image_segmentation_class - output_image_segmentation_class),
                )

        self.cross_entropy_vals[fname][slice_idx] = self.cross_entropy_metric.to(self.device)(  # noqa: F841
            target_segmentation.argmax(1), pred_segmentation  # type: ignore
        )
        dice_score, _ = self.dice_coefficient_metric(target_segmentation, pred_segmentation)
        self.dice_vals[fname][slice_idx] = dice_score

        return {"val_loss": val_loss}

    def test_step(  # noqa: D102
        self, batch: Dict[float, torch.Tensor], batch_idx: int
    ) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
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
            'fname' : str
                File name.
            'slice_num' : int
                Slice number.
            'acc' : float
                Acceleration factor of the sampling mask.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            kspace,  # noqa: F841
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            fname,
            slice_idx,
            acc,  # noqa: F841
        ) = batch

        y, mask, init_reconstruction_pred, r = self.process_inputs(y, mask, init_reconstruction_pred)  # noqa: F841

        pred_segmentation = self.forward(y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction)

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )

        if isinstance(pred_segmentation, list):
            pred_segmentation = pred_segmentation[-1]

        # normalize for visualization
        if not utils.is_none(self.segmentation_classes_thresholds):
            for class_idx, class_threshold in enumerate(self.segmentation_classes_thresholds):
                if not utils.is_none(class_threshold):
                    if target_segmentation.dim() != 1:  # type: ignore
                        target_segmentation[:, class_idx] = (  # type: ignore
                            target_segmentation[:, class_idx] > class_threshold  # type: ignore
                        )
                    pred_segmentation[:, class_idx] = pred_segmentation[:, class_idx] > class_threshold

        if self.log_images:
            slice_idx = int(slice_idx)
            key = f"{fname[0]}_images_idx_{slice_idx}"  # type: ignore

            target_reconstruction = (
                torch.abs(target_reconstruction / torch.max(torch.abs(target_reconstruction))).detach().cpu()
            )
            self.log_image(f"{key}/reconstruction/target", target_reconstruction)

            for class_idx in range(pred_segmentation.shape[1]):  # type: ignore
                output_image_segmentation_class = pred_segmentation[:, class_idx]
                self.log_image(
                    f"{key}/segmentation_classes/prediction_class_{class_idx}", output_image_segmentation_class
                )

                if target_segmentation.dim() != 1:  # type: ignore
                    target_image_segmentation_class = target_segmentation[:, class_idx]  # type: ignore
                    self.log_image(
                        f"{key}/segmentation_classes/target_class_{class_idx}",
                        target_image_segmentation_class,  # type: ignore
                    )

                    self.log_image(
                        f"{key}/segmentation_classes/error_1_class_{class_idx}",
                        torch.abs(target_image_segmentation_class - output_image_segmentation_class),
                    )

        if target_segmentation.dim() != 1:  # type: ignore
            self.cross_entropy_vals[fname][slice_idx] = self.cross_entropy_metric.to(self.device)(  # noqa: F841
                target_segmentation.argmax(1), pred_segmentation  # type: ignore
            )
            dice_score, _ = self.dice_coefficient_metric(target_segmentation, pred_segmentation)
            self.dice_vals[fname][slice_idx] = dice_score

        return str(fname[0]), slice_idx, pred_segmentation.detach().cpu().numpy()  # type: ignore

    def train_epoch_end(self, outputs):
        """
        Called at the end of train epoch to aggregate the loss values.

        Parameters
        ----------
        outputs : List
            List of outputs of the train batches.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("train_loss", torch.stack([x["train_loss"] for x in outputs]).mean(), sync_dist=True)
        self.log(
            f"train_loss_{self.acc}x",
            torch.stack([x[f"train_loss_{self.acc}x"] for x in outputs]).mean(),
            sync_dist=True,
        )
        self.log("lr", torch.stack([x["lr"] for x in outputs]).mean(), sync_dist=True)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs : List
            List of outputs of the train batches.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean(), sync_dist=True)

        # Log metrics.
        cross_entropy_vals = defaultdict(dict)
        dice_vals = defaultdict(dict)

        for k, v in self.cross_entropy_vals.items():
            cross_entropy_vals[k].update(v)
        for k, v in self.dice_vals.items():
            dice_vals[k].update(v)

        metrics_segmentation = {"Cross_Entropy": 0, "DICE": 0}

        local_examples = 0
        for fname in dice_vals:
            local_examples += 1

            metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics_segmentation.items():
            self.log(f"{metric}_Segmentation", value / tot_examples, sync_dist=True)

    def test_epoch_end(self, outputs):  # noqa: D102
        """
        Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Parameters
        ----------
        outputs : List
            List of outputs of the train batches.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        cross_entropy_vals = defaultdict(dict)
        dice_vals = defaultdict(dict)

        for k, v in self.cross_entropy_vals.items():
            cross_entropy_vals[k].update(v)
        for k, v in self.dice_vals.items():
            dice_vals[k].update(v)

        metrics_segmentation = {"Cross_Entropy": 0, "DICE": 0}

        local_examples = 0
        for fname in dice_vals:
            local_examples += 1

            metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics_segmentation.items():
            self.log(f"{metric}_Segmentation", value / tot_examples, sync_dist=True)

        segmentations = defaultdict(list)
        for fname, slice_num, segmentations_pred in outputs:
            segmentations[fname].append((slice_num, segmentations_pred))

        for fname in segmentations:
            segmentations[fname] = np.stack([out for _, out in sorted(segmentations[fname])])
            if self.consecutive_slices > 1:
                # If we have consecutive slices, we need to make sure that we will save all slices.
                segmentations_slices = []
                for i in range(segmentations[fname].shape[0]):
                    if i == 0:
                        segmentations_slices.append(segmentations[fname][i][0])
                    elif i == segmentations[fname].shape[0] - 1:
                        for j in range(self.consecutive_slices):
                            segmentations_slices.append(segmentations[fname][i][j])
                    else:
                        segmentations_slices.append(segmentations[fname][i][self.consecutive_slices // 2])
                segmentations[fname] = np.stack(segmentations_slices)

        out_dir = Path(os.path.join(self.logger.log_dir, "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        for fname, segmentations_pred in segmentations.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("segmentation", data=segmentations_pred)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration file.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Dataloader.
        """
        mask_root = cfg.get("mask_path", None)
        mask_args = cfg.get("mask_args", None)
        shift_mask = mask_args.get("shift_mask", False)
        mask_type = mask_args.get("type", None)

        mask_func = None  # type: ignore
        mask_center_scale = 0.02

        if utils.is_none(mask_root) and not utils.is_none(mask_type):
            accelerations = mask_args.get("accelerations", [1])
            center_fractions = mask_args.get("center_fractions", [1])
            mask_center_scale = mask_args.get("scale", 0.02)

            mask_func = (
                [
                    subsample.create_masker(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) >= 2
                else [subsample.create_masker(mask_type, center_fractions, accelerations)]
            )

        complex_data = cfg.get("complex_data", True)

        dataset = mri_segmentation_loader.SegmentationMRIDataset(
            root=cfg.get("data_path"),
            coil_sensitivity_maps_root=cfg.get("coil_sensitivity_maps_path", None),
            mask_root=mask_root,
            dataset_format=cfg.get("dataset_format", None),
            sample_rate=cfg.get("sample_rate", 1.0),
            volume_sample_rate=cfg.get("volume_sample_rate", None),
            use_dataset_cache=cfg.get("use_dataset_cache", False),
            dataset_cache_file=cfg.get("dataset_cache_file", None),
            num_cols=cfg.get("num_cols", None),
            consecutive_slices=cfg.get("consecutive_slices", 1),
            data_saved_per_slice=cfg.get("data_saved_per_slice", False),
            transform=transforms.SegmentationMRIDataTransforms(
                complex_data=complex_data,
                apply_prewhitening=cfg.get("apply_prewhitening", False),
                find_patch_size=cfg.get("find_patch_size", False),
                prewhitening_scale_factor=cfg.get("prewhitening_scale_factor", 1.0),
                prewhitening_patch_start=cfg.get("prewhitening_patch_start", 10),
                prewhitening_patch_length=cfg.get("prewhitening_patch_length", 30),
                apply_gcc=cfg.get("apply_gcc", False),
                gcc_virtual_coils=cfg.get("gcc_virtual_coils", 10),
                gcc_calib_lines=cfg.get("gcc_calib_lines", 10),
                gcc_align_data=cfg.get("gcc_align_data", False),
                coil_combination_method=cfg.get("coil_combination_method", "SENSE"),
                dimensionality=cfg.get("dimensionality", 2),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", False),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", True),
                normalization_type=cfg.get("normalization_type", "max"),
                fft_centered=cfg.get("fft_centered", False),
                fft_normalization=cfg.get("fft_normalization", "backward"),
                spatial_dims=cfg.get("spatial_dims", None),
                coil_dim=cfg.get("coil_dim", 1),
                consecutive_slices=cfg.get("consecutive_slices", 1),
                use_seed=cfg.get("use_seed", True),
            ),
            segmentations_root=cfg.get("segmentations_path"),
            initial_predictions_root=cfg.get("initial_predictions_path"),
            segmentation_classes=cfg.get("segmentation_classes", 2),
            segmentation_classes_to_remove=cfg.get("segmentation_classes_to_remove", None),
            segmentation_classes_to_combine=cfg.get("segmentation_classes_to_combine", None),
            segmentation_classes_to_separate=cfg.get("segmentation_classes_to_separate", None),
            segmentation_classes_thresholds=cfg.get("segmentation_classes_thresholds", None),
            complex_data=complex_data,
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size", 1),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
