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
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader

import mridc.collections.reconstruction.losses as reconstruction_losses
import mridc.collections.segmentation.losses as segmentation_losses
from mridc.collections.common.data import subsample
from mridc.collections.common.nn.base import BaseMRIModel, BaseSensitivityModel, DistributedMetricSum
from mridc.collections.common.parts import fft, utils
from mridc.collections.multitask.rs.data import mrirs_loader
from mridc.collections.multitask.rs.parts.transforms import RSMRIDataTransforms
from mridc.collections.reconstruction.metrics import reconstruction_metrics

__all__ = ["BaseMRIReconstructionSegmentationModel"]


class BaseMRIReconstructionSegmentationModel(BaseMRIModel, ABC):  # type: ignore
    """
    Base class of all (multitask) MRI reconstruction and MRI segmentation models.

    Parameters
    ----------
    cfg: DictConfig
        The configuration file.
    trainer: Trainer
        The PyTorch Lightning trainer.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):  # noqa: W0221
        super().__init__(cfg=cfg, trainer=trainer)

        self.acc = 1  # fixed acceleration factor to ensure acc is not None

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered", False)
        self.fft_normalization = cfg_dict.get("fft_normalization", "backward")
        self.spatial_dims = cfg_dict.get("spatial_dims", None)
        self.coil_dim = cfg_dict.get("coil_dim", 1)

        self.coil_combination_method = cfg_dict.get("coil_combination_method", "SENSE")

        # Initialize the sensitivity network if cfg_dict.get("use_sens_net") is True
        self.use_sens_net = cfg_dict.get("use_sens_net", False)
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans", 8),
                cfg_dict.get("sens_pools", 4),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("sens_mask_type", "2D"),
                normalize=cfg_dict.get("sens_normalize", True),
                mask_center=cfg_dict.get("sens_mask_center", True),
            )

        self.no_dc = cfg_dict.get("no_dc", False)

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

        self.ssdu = cfg_dict.get("ssdu", False)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.total_reconstruction_loss_weight = cfg_dict.get("total_reconstruction_loss_weight", 1.0)
            reconstruction_loss_fn = cfg_dict.get("reconstruction_loss_fn")
            if reconstruction_loss_fn == "ssim":
                if self.ssdu:
                    raise ValueError("SSIM loss is not supported for SSDU.")
                self.train_loss_fn = reconstruction_losses.ssim.SSIMLoss()
                self.val_loss_fn = reconstruction_losses.ssim.SSIMLoss()
            elif reconstruction_loss_fn == "l1":
                self.train_loss_fn = L1Loss()
                self.val_loss_fn = L1Loss()
            elif reconstruction_loss_fn == "mse":
                self.train_loss_fn = MSELoss()
                self.val_loss_fn = MSELoss()
            else:
                self.train_loss_fn = L1Loss()
                self.val_loss_fn = L1Loss()

            if self.ssdu:
                self.kspace_reconstruction_loss = True
            else:
                self.kspace_reconstruction_loss = cfg_dict.get("kspace_reconstruction_loss", False)

            self.reconstruction_loss_regularization_factor = cfg_dict.get(
                "reconstruction_loss_regularization_factor", 1.0
            )

            self.MSE = DistributedMetricSum()
            self.NMSE = DistributedMetricSum()
            self.SSIM = DistributedMetricSum()
            self.PSNR = DistributedMetricSum()
            self.TotExamples = DistributedMetricSum()

            self.mse_vals_reconstruction: Dict = defaultdict(dict)
            self.nmse_vals_reconstruction: Dict = defaultdict(dict)
            self.ssim_vals_reconstruction: Dict = defaultdict(dict)
            self.psnr_vals_reconstruction: Dict = defaultdict(dict)

    def process_reconstruction_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Dict,
        r: int,
        loss_func: torch.nn.Module,
        kspace_reconstruction_loss: bool = False,
    ) -> torch.Tensor:
        """
        Processes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.
        kspace_reconstruction_loss : bool
            If True, the loss will be computed on the k-space data. Otherwise, the loss will be computed on the
            image space data. Default is ``False``. Note that this is different from
            ``self.kspace_reconstruction_loss``, so it can be used with multiple losses.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if self.unnormalize_loss_inputs:
            if self.n2r and not attrs["n2r_supervised"]:
                target = utils.unnormalize(
                    target,
                    {
                        "min": attrs["prediction_min"] if "prediction_min" in attrs else attrs[f"prediction_min_{r}"],
                        "max": attrs["prediction_max"] if "prediction_max" in attrs else attrs[f"prediction_max_{r}"],
                        "mean": (
                            attrs["prediction_mean"] if "prediction_mean" in attrs else attrs[f"prediction_mean_{r}"]
                        ),
                        "std": attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"],
                    },
                    self.normalization_type,
                )
                prediction = utils.unnormalize(
                    prediction,
                    {
                        "min": (
                            attrs["noise_prediction_min"]
                            if "noise_prediction_min" in attrs
                            else attrs[f"noise_prediction_min_{r}"]
                        ),
                        "max": (
                            attrs["noise_prediction_max"]
                            if "noise_prediction_max" in attrs
                            else attrs[f"noise_prediction_max_{r}"]
                        ),
                        attrs["noise_prediction_mean"] if "noise_prediction_mean" in attrs else "mean": attrs[
                            f"noise_prediction_mean_{r}"
                        ],
                        attrs["noise_prediction_std"] if "noise_prediction_std" in attrs else "std": attrs[
                            f"noise_prediction_std_{r}"
                        ],
                    },
                    self.normalization_type,
                )
            else:
                target = utils.unnormalize(
                    target,
                    {
                        "min": attrs["target_min"],
                        "max": attrs["target_max"],
                        "mean": attrs["target_mean"],
                        "std": attrs["target_std"],
                    },
                    self.normalization_type,
                )
                prediction = utils.unnormalize(
                    prediction,
                    {
                        "min": attrs["prediction_min"] if "prediction_min" in attrs else attrs[f"prediction_min_{r}"],
                        "max": attrs["prediction_max"] if "prediction_max" in attrs else attrs[f"prediction_max_{r}"],
                        "mean": (
                            attrs["prediction_mean"] if "prediction_mean" in attrs else attrs[f"prediction_mean_{r}"]
                        ),
                        "std": attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"],
                    },
                    self.normalization_type,
                )

            sensitivity_maps = utils.unnormalize(
                sensitivity_maps,
                {
                    "min": attrs["sensitivity_maps_min"],
                    "max": attrs["sensitivity_maps_max"],
                    "mean": attrs["sensitivity_maps_mean"],
                    "std": attrs["sensitivity_maps_std"],
                },
                self.normalization_type,
            )

        if not self.kspace_reconstruction_loss and not kspace_reconstruction_loss and not self.unnormalize_loss_inputs:
            target = torch.abs(target / torch.max(torch.abs(target)))
        else:
            if target.shape[-1] != 2:
                target = torch.view_as_real(target)
            if self.ssdu or kspace_reconstruction_loss:
                target = utils.expand_op(target, sensitivity_maps, self.coil_dim)
            target = fft.fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)

        if "ssim" in str(loss_func).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for SSIM loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                y = torch.abs(y / torch.max(torch.abs(y)))
                return loss_func(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for any (expect the SSIM) loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                if (
                    not self.kspace_reconstruction_loss
                    and not kspace_reconstruction_loss
                    and not self.unnormalize_loss_inputs
                ):
                    y = torch.abs(y / torch.max(torch.abs(y)))
                else:
                    if y.shape[-1] != 2:
                        y = torch.view_as_real(y)
                    if self.ssdu or kspace_reconstruction_loss:
                        y = utils.expand_op(y, sensitivity_maps, self.coil_dim)
                    y = fft.fft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                    if self.ssdu or kspace_reconstruction_loss:
                        y = y * mask
                return loss_func(x, y)

        return compute_reconstruction_loss(target, prediction) * self.reconstruction_loss_regularization_factor

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
            _, loss_dict["dice_loss"] = self.segmentation_loss_fn["dice"](target, prediction)  # noqa: E1102
            loss_dict["dice_loss"] = loss_dict["dice_loss"] * self.dice_loss_weighting_factor
        loss_dict["segmentation_loss"] = loss_dict["cross_entropy_loss"] + loss_dict["dice_loss"]
        return loss_dict

    @staticmethod
    def process_inputs(
        kspace: Union[list, torch.Tensor],
        y: Union[list, torch.Tensor],
        mask: Union[list, torch.Tensor],
        init_pred: Union[list, torch.Tensor],
        target: Union[list, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the inputs
        are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If only one
        acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        kspace : Union[list, torch.Tensor]
            Full k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : Union[list, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : Union[list, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].
        init_pred : Union[list, torch.Tensor]
            Initial prediction of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : Union[list, torch.Tensor]
            Target data of length n_accelerations or shape [batch_size, n_x, n_y, 2].

        Returns
        -------
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        init_pred : torch.Tensor
            Initial prediction of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
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
        if isinstance(kspace, list):
            kspace = kspace[r]
            target = target[r]
        return kspace, y, mask, init_pred, target, r

    def training_step(  # noqa: W0613
        self, batch: Dict[float, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
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
            'attrs' : dict
                Attributes dictionary.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            kspace,
            y,
            sensitivity_maps,
            mask,
            init_reconstruction_pred,
            target_reconstruction,
            target_segmentation,
            _,
            _,
            acc,
            attrs,
        ) = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_reconstruction_pred, n2r_init_reconstruction_pred = init_reconstruction_pred  # type: ignore

        kspace, y, mask, init_reconstruction_pred, target_reconstruction, r = self.process_inputs(
            kspace, y, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            if isinstance(n2r_y, list):  # type: ignore
                n2r_y = n2r_y[r]  # type: ignore
                n2r_mask = n2r_mask[r]  # type: ignore
                n2r_init_reconstruction_pred = n2r_init_reconstruction_pred[r]  # type: ignore
            if self.ssdu:
                mask, loss_mask = mask  # type: ignore
            else:
                loss_mask = torch.ones_like(mask)
            if n2r_mask.dim() < mask.dim():  # type: ignore
                n2r_mask = None
        elif self.ssdu and not self.n2r:
            mask, loss_mask = mask  # type: ignore
            n2r_mask = None
        else:
            loss_mask = torch.ones_like(mask)
            n2r_mask = None

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        pred_reconstruction_n2r = None
        if self.n2r and n2r_mask is not None:
            pred_reconstruction_n2r, _ = self.forward(
                n2r_y, sensitivity_maps, n2r_mask, n2r_init_reconstruction_pred, target_reconstruction
            )
            pred_reconstruction_n2r = utils.real_to_complex_tensor_or_list(pred_reconstruction_n2r)

        pred_reconstruction = utils.real_to_complex_tensor_or_list(pred_reconstruction)
        target_reconstruction = utils.real_to_complex_tensor_or_list(target_reconstruction)  # type: ignore

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )
        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        if self.use_reconstruction_module:
            if pred_reconstruction_n2r is not None:
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    reconstruction_loss = sum(
                        self.process_reconstruction_loss(
                            target_reconstruction,
                            pred_reconstruction,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.train_loss_fn,
                            # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                            kspace_reconstruction_loss=self.ssdu,
                        )
                    )
                elif not attrs["n2r_supervised"]:  # type: ignore
                    reconstruction_loss = self.n2r_loss_regularization_factor * sum(
                        self.process_reconstruction_loss(
                            pred_reconstruction,
                            pred_reconstruction_n2r,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.train_loss_fn,
                            kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                        )
                    )
            else:
                reconstruction_loss = self.process_reconstruction_loss(
                    target_reconstruction,
                    pred_reconstruction,
                    sensitivity_maps,
                    loss_mask,
                    attrs,  # type: ignore
                    r,
                    loss_func=self.train_loss_fn,
                    kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                )
            train_loss = (
                self.total_segmentation_loss_weight * segmentation_loss
                + self.total_reconstruction_loss_weight * reconstruction_loss
            )
        else:
            train_loss = self.total_segmentation_loss_weight * segmentation_loss

        self.acc = r if r != 0 else acc  # type: ignore
        tensorboard_logs = {
            f"train_loss_{self.acc}x": train_loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:  # noqa: W0221
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
            'attrs' : dict
                Attributes dictionary.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
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
            _,
            attrs,
        ) = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_reconstruction_pred, n2r_init_reconstruction_pred = init_reconstruction_pred  # type: ignore

        kspace, y, mask, init_reconstruction_pred, target_reconstruction, r = self.process_inputs(
            kspace, y, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            if isinstance(n2r_y, list):  # type: ignore
                n2r_y = n2r_y[r]  # type: ignore
                n2r_mask = n2r_mask[r]  # type: ignore
                n2r_init_reconstruction_pred = n2r_init_reconstruction_pred[r]  # type: ignore
            if self.ssdu:
                mask, loss_mask = mask  # type: ignore
            else:
                loss_mask = torch.ones_like(mask)
            if n2r_mask.dim() < mask.dim():  # type: ignore
                n2r_mask = None
        elif self.ssdu and not self.n2r:
            mask, loss_mask = mask  # type: ignore
            n2r_mask = None
        else:
            loss_mask = torch.ones_like(mask)
            n2r_mask = None

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        pred_reconstruction_n2r = None
        if self.n2r and n2r_mask is not None:
            pred_reconstruction_n2r, _ = self.forward(
                n2r_y, sensitivity_maps, n2r_mask, n2r_init_reconstruction_pred, target_reconstruction
            )
            pred_reconstruction_n2r = utils.real_to_complex_tensor_or_list(pred_reconstruction_n2r)

        if isinstance(pred_reconstruction, list):
            pred_reconstruction = pred_reconstruction[-1]
        pred_reconstruction = utils.real_to_complex_tensor_or_list(pred_reconstruction)
        target_reconstruction = utils.real_to_complex_tensor_or_list(target_reconstruction)  # type: ignore

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )
            target_reconstruction = target_reconstruction.reshape(  # type: ignore
                batch_size * slices, *target_reconstruction.shape[2:]  # type: ignore
            )
            if self.n2r and n2r_mask is not None:
                pred_reconstruction_n2r = pred_reconstruction_n2r.reshape(  # type: ignore
                    batch_size * slices, *pred_reconstruction_n2r.shape[2:]  # type: ignore
                )

        if self.log_images:
            slice_idx = int(slice_idx)
            key = f"{fname[0]}_images_idx_{slice_idx}"  # type: ignore
            self.log_image(f"{key}/reconstruction/target", target_reconstruction)

        segmentation_loss = self.process_segmentation_loss(target_segmentation, pred_segmentation)["segmentation_loss"]

        if isinstance(pred_segmentation, list):
            pred_segmentation = pred_segmentation[-1]

        if self.use_reconstruction_module:
            if pred_reconstruction_n2r is not None:
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    reconstruction_loss = sum(
                        self.process_reconstruction_loss(
                            target_reconstruction,
                            pred_reconstruction,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.val_loss_fn,
                            # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                            kspace_reconstruction_loss=self.ssdu,
                        )
                    )
                elif not attrs["n2r_supervised"]:  # type: ignore
                    reconstruction_loss = self.n2r_loss_regularization_factor * sum(
                        self.process_reconstruction_loss(
                            pred_reconstruction,
                            pred_reconstruction_n2r,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.val_loss_fn,
                            kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                        )
                    )
            else:
                reconstruction_loss = self.process_reconstruction_loss(
                    target_reconstruction,
                    pred_reconstruction,
                    sensitivity_maps,
                    loss_mask,
                    attrs,  # type: ignore
                    r,
                    loss_func=self.val_loss_fn,
                    kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                )
            val_loss = (
                self.total_segmentation_loss_weight * segmentation_loss
                + self.total_reconstruction_loss_weight * reconstruction_loss
            )

            pred_reconstruction = utils.real_to_complex_tensor_or_list(pred_reconstruction)

            if self.consecutive_slices > 1:
                pred_reconstruction = pred_reconstruction.reshape(
                    pred_reconstruction.shape[0] * pred_reconstruction.shape[1], *pred_reconstruction.shape[2:]
                )

            target_reconstruction = (
                torch.abs(target_reconstruction / torch.max(torch.abs(target_reconstruction))).detach().cpu()
            )
            output_reconstruction = (
                torch.abs(pred_reconstruction / torch.max(torch.abs(pred_reconstruction))).detach().cpu()
            )

            if self.log_images:
                self.log_image(f"{key}/reconstruction/prediction", output_reconstruction)
                self.log_image(f"{key}/reconstruction/error", torch.abs(target_reconstruction - output_reconstruction))

            target_reconstruction = target_reconstruction.numpy()  # type: ignore
            output_reconstruction = output_reconstruction.numpy()
            self.mse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.mse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.nmse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.nmse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.ssim_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.ssim(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
            self.psnr_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.psnr(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
        else:
            val_loss = self.total_segmentation_loss_weight * segmentation_loss

        # normalize for visualization
        if not utils.is_none(self.segmentation_classes_thresholds):
            for class_idx, class_threshold in enumerate(self.segmentation_classes_thresholds):
                if not utils.is_none(class_threshold):
                    target_segmentation[:, class_idx] = (  # type: ignore
                        target_segmentation[:, class_idx] > class_threshold  # type: ignore
                    )  # type: ignore
                    pred_segmentation[:, class_idx] = pred_segmentation[:, class_idx] > class_threshold

        if self.log_images:
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

        self.cross_entropy_vals[fname][slice_idx] = self.cross_entropy_metric.to(self.device)(  # noqa: E1102
            target_segmentation.argmax(1), pred_segmentation  # type: ignore
        )
        dice_score, _ = self.dice_coefficient_metric(target_segmentation, pred_segmentation)
        self.dice_vals[fname][slice_idx] = dice_score

        return {"val_loss": val_loss}

    def test_step(  # noqa: W0221
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
            'attrs' : dict
                Attributes dictionary.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
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
            _,
            attrs,
        ) = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_reconstruction_pred, n2r_init_reconstruction_pred = init_reconstruction_pred  # type: ignore

        kspace, y, mask, init_reconstruction_pred, target_reconstruction, r = self.process_inputs(
            kspace, y, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            if isinstance(n2r_y, list):  # type: ignore
                n2r_y = n2r_y[r]  # type: ignore
                n2r_mask = n2r_mask[r]  # type: ignore
                n2r_init_reconstruction_pred = n2r_init_reconstruction_pred[r]  # type: ignore
            if self.ssdu:
                mask, loss_mask = mask  # type: ignore
            else:
                loss_mask = torch.ones_like(mask)
            if n2r_mask.dim() < mask.dim():  # type: ignore
                n2r_mask = None
        elif self.ssdu and not self.n2r:
            mask, loss_mask = mask  # type: ignore
            n2r_mask = None
        else:
            loss_mask = torch.ones_like(mask)
            n2r_mask = None

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        pred_reconstruction, pred_segmentation = self.forward(
            y, sensitivity_maps, mask, init_reconstruction_pred, target_reconstruction
        )

        if self.consecutive_slices > 1:
            batch_size, slices = target_segmentation.shape[:2]  # type: ignore
            target_segmentation = target_segmentation.reshape(  # type: ignore
                batch_size * slices, *target_segmentation.shape[2:]  # type: ignore
            )
            target_reconstruction = target_reconstruction.reshape(  # type: ignore
                batch_size * slices, *target_reconstruction.shape[2:]  # type: ignore
            )

        if self.log_images:
            slice_idx = int(slice_idx)
            key = f"{fname[0]}_images_idx_{slice_idx}"  # type: ignore
            if target_reconstruction.dim() > 2:  # type: ignore
                self.log_image(f"{key}/reconstruction/target", target_reconstruction)

        if isinstance(pred_segmentation, list):
            pred_segmentation = pred_segmentation[-1]

        if self.use_reconstruction_module:
            # JRS Cascades
            if isinstance(pred_reconstruction, list):
                pred_reconstruction = pred_reconstruction[-1]
            # Cascades
            if isinstance(pred_reconstruction, list):
                pred_reconstruction = pred_reconstruction[-1]
            # Time-steps
            if isinstance(pred_reconstruction, list):
                pred_reconstruction = pred_reconstruction[-1]

            if self.consecutive_slices > 1:
                pred_reconstruction = pred_reconstruction.reshape(
                    pred_reconstruction.shape[0] * pred_reconstruction.shape[1], *pred_reconstruction.shape[2:]
                )

            target_reconstruction = (
                torch.abs(target_reconstruction / torch.max(torch.abs(target_reconstruction))).detach().cpu()
            )
            output_reconstruction = (
                torch.abs(pred_reconstruction / torch.max(torch.abs(pred_reconstruction))).detach().cpu()
            )

            if self.log_images:
                self.log_image(f"{key}/reconstruction/prediction", output_reconstruction)
                self.log_image(f"{key}/reconstruction/error", torch.abs(target_reconstruction - output_reconstruction))

            target_reconstruction = target_reconstruction.numpy()  # type: ignore
            output_reconstruction = output_reconstruction.numpy()
            self.mse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.mse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.nmse_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.nmse(target_reconstruction, output_reconstruction)
            ).view(1)
            self.ssim_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.ssim(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)
            self.psnr_vals_reconstruction[fname][slice_idx] = torch.tensor(
                reconstruction_metrics.psnr(
                    target_reconstruction,
                    output_reconstruction,
                    maxval=output_reconstruction.max() - output_reconstruction.min(),
                )
            ).view(1)

        # normalize for visualization
        if not utils.is_none(self.segmentation_classes_thresholds):
            for class_idx, class_threshold in enumerate(self.segmentation_classes_thresholds):
                if not utils.is_none(class_threshold):
                    if target_segmentation.dim() != 1:  # type: ignore
                        target_segmentation[:, class_idx] = (  # type: ignore
                            target_segmentation[:, class_idx] > class_threshold  # type: ignore
                        )  # type: ignore
                    pred_segmentation[:, class_idx] = pred_segmentation[:, class_idx] > class_threshold

        if self.log_images:
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
            self.cross_entropy_vals[fname][slice_idx] = self.cross_entropy_metric.to(self.device)(  # noqa: E1102
                target_segmentation.argmax(1), pred_segmentation  # type: ignore
            )
            dice_score, _ = self.dice_coefficient_metric(target_segmentation, pred_segmentation)
            self.dice_vals[fname][slice_idx] = dice_score

        predictions = (
            (pred_segmentation.detach().cpu().numpy(), pred_reconstruction.detach().cpu().numpy())
            if self.use_reconstruction_module
            else (pred_segmentation.detach().cpu().numpy(), pred_segmentation.detach().cpu().numpy())
        )

        return (str(fname[0]), slice_idx, predictions)  # type: ignore

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

    def validation_epoch_end(self, outputs):  # noqa: W0221, MC0001
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

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)

            metrics_reconstruction = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}

        local_examples = 0
        for fname in dice_vals:
            local_examples += 1

            metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics_reconstruction["MSE"] = metrics_reconstruction["MSE"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["NMSE"] = metrics_reconstruction["NMSE"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["SSIM"] = metrics_reconstruction["SSIM"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["PSNR"] = metrics_reconstruction["PSNR"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in psnr_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        if self.use_reconstruction_module:
            metrics_reconstruction["MSE"] = self.MSE(metrics_reconstruction["MSE"])
            metrics_reconstruction["NMSE"] = self.NMSE(metrics_reconstruction["NMSE"])
            metrics_reconstruction["SSIM"] = self.SSIM(metrics_reconstruction["SSIM"])
            metrics_reconstruction["PSNR"] = self.PSNR(metrics_reconstruction["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics_segmentation.items():
            self.log(f"{metric}_Segmentation", value / tot_examples, sync_dist=True)
        if self.use_reconstruction_module:
            for metric, value in metrics_reconstruction.items():
                self.log(f"{metric}_Reconstruction", value / tot_examples, sync_dist=True)

    def test_epoch_end(self, outputs):  # noqa: MC0001
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

        if self.use_reconstruction_module:
            mse_vals_reconstruction = defaultdict(dict)
            nmse_vals_reconstruction = defaultdict(dict)
            ssim_vals_reconstruction = defaultdict(dict)
            psnr_vals_reconstruction = defaultdict(dict)

            for k, v in self.mse_vals_reconstruction.items():
                mse_vals_reconstruction[k].update(v)
            for k, v in self.nmse_vals_reconstruction.items():
                nmse_vals_reconstruction[k].update(v)
            for k, v in self.ssim_vals_reconstruction.items():
                ssim_vals_reconstruction[k].update(v)
            for k, v in self.psnr_vals_reconstruction.items():
                psnr_vals_reconstruction[k].update(v)

            metrics_reconstruction = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}

        local_examples = 0
        for fname in dice_vals:
            local_examples += 1

            metrics_segmentation["Cross_Entropy"] = metrics_segmentation["Cross_Entropy"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in cross_entropy_vals[fname].items()])
            )
            metrics_segmentation["DICE"] = metrics_segmentation["DICE"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in dice_vals[fname].items()])
            )

            if self.use_reconstruction_module:
                metrics_reconstruction["MSE"] = metrics_reconstruction["MSE"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in mse_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["NMSE"] = metrics_reconstruction["NMSE"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in nmse_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["SSIM"] = metrics_reconstruction["SSIM"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in ssim_vals_reconstruction[fname].items()])
                )
                metrics_reconstruction["PSNR"] = metrics_reconstruction["PSNR"] + torch.mean(
                    torch.cat([v.view(-1).float() for _, v in psnr_vals_reconstruction[fname].items()])
                )

        # reduce across ddp via sum
        metrics_segmentation["Cross_Entropy"] = self.CROSS_ENTROPY(metrics_segmentation["Cross_Entropy"])
        metrics_segmentation["DICE"] = self.DICE(metrics_segmentation["DICE"])

        if self.use_reconstruction_module:
            metrics_reconstruction["MSE"] = self.MSE(metrics_reconstruction["MSE"])
            metrics_reconstruction["NMSE"] = self.NMSE(metrics_reconstruction["NMSE"])
            metrics_reconstruction["SSIM"] = self.SSIM(metrics_reconstruction["SSIM"])
            metrics_reconstruction["PSNR"] = self.PSNR(metrics_reconstruction["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics_segmentation.items():
            self.log(f"{metric}_Segmentation", value / tot_examples, sync_dist=True)
        if self.use_reconstruction_module:
            for metric, value in metrics_reconstruction.items():
                self.log(f"{metric}_Reconstruction", value / tot_examples, sync_dist=True)

        segmentations = defaultdict(list)
        for fname, slice_num, output in outputs:
            segmentations_pred, _ = output
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

        if self.use_reconstruction_module:  # noqa: R1792
            reconstructions = defaultdict(list)
            for fname, slice_num, output in outputs:
                _, reconstructions_pred = output
                reconstructions[fname].append((slice_num, reconstructions_pred))

            for fname in reconstructions:
                reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])
                if self.consecutive_slices > 1:
                    # If we have consecutive slices, we need to make sure that we will save all slices.
                    reconstructions_slices = []
                    for i in range(reconstructions[fname].shape[0]):
                        if i == 0:
                            reconstructions_slices.append(reconstructions[fname][i][0])
                        elif i == segmentations[fname].shape[0] - 1:
                            for j in range(self.consecutive_slices):
                                reconstructions_slices.append(reconstructions[fname][i][j])
                        else:
                            reconstructions_slices.append(reconstructions[fname][i][self.consecutive_slices // 2])
                    reconstructions[fname] = np.stack(reconstructions_slices)
        else:
            reconstructions = None

        out_dir = Path(os.path.join(self.logger.log_dir, "predictions"))
        out_dir.mkdir(exist_ok=True, parents=True)

        if reconstructions is not None:
            for (fname, segmentations_pred), (_, reconstructions_pred) in zip(
                segmentations.items(), reconstructions.items()
            ):
                with h5py.File(out_dir / fname, "w") as hf:
                    hf.create_dataset("segmentation", data=segmentations_pred)
                    hf.create_dataset("reconstruction", data=reconstructions_pred)
        else:
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

        dataset = mrirs_loader.RSMRIDataset(
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
            transform=RSMRIDataTransforms(
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
                ssdu=cfg.get("ssdu", False),
                ssdu_mask_type=cfg.get("ssdu_mask_type", "Gaussian"),
                ssdu_rho=cfg.get("ssdu_rho", 0.4),
                ssdu_acs_block_size=cfg.get("ssdu_acs_block_size", (4, 4)),
                ssdu_gaussian_std_scaling_factor=cfg.get("ssdu_gaussian_std_scaling_factor", 4.0),
                ssdu_export_and_reuse_masks=cfg.get("ssdu_export_and_reuse_masks", False),
                crop_size=cfg.get("crop_size", None),
                kspace_crop=cfg.get("kspace_crop", False),
                crop_before_masking=cfg.get("crop_before_masking", False),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size", None),
                normalize_inputs=cfg.get("normalize_inputs", True),
                normalization_type=cfg.get("normalization_type", "max"),
                kspace_normalization=cfg.get("kspace_normalization", False),
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
