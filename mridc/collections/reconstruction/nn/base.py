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
from mridc.collections.common.data import subsample
from mridc.collections.common.nn.base import BaseMRIModel, BaseSensitivityModel, DistributedMetricSum
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.data import mri_reconstruction_loader
from mridc.collections.reconstruction.metrics import reconstruction_metrics
from mridc.collections.reconstruction.parts import transforms as reconstruction_transforms

__all__ = ["BaseMRIReconstructionModel"]


class BaseMRIReconstructionModel(BaseMRIModel, ABC):  # type: ignore
    """
    Base class of all MRI reconstruction models.

    Parameters
    ----------
    cfg: DictConfig
        The configuration file.
    trainer: Trainer
        The PyTorch Lightning trainer.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered", False)
        self.fft_normalization = cfg_dict.get("fft_normalization", "backward")
        self.spatial_dims = cfg_dict.get("spatial_dims", None)
        self.coil_dim = cfg_dict.get("coil_dim", 1)

        self.coil_combination_method = cfg_dict.get("coil_combination_method", "SENSE")

        self.ssdu = cfg_dict.get("ssdu", False)
        self.n2r = cfg_dict.get("n2r", False)

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

        if cfg_dict.get("loss_fn") == "ssim":
            if self.ssdu:
                raise ValueError("SSIM loss is not supported for SSDU.")
            self.train_loss_fn = reconstruction_losses.ssim.SSIMLoss()
            self.val_loss_fn = reconstruction_losses.ssim.SSIMLoss()
        elif cfg_dict.get("loss_fn") == "mse":
            self.train_loss_fn = MSELoss()
            self.val_loss_fn = MSELoss()
        elif cfg_dict.get("loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
            self.val_loss_fn = L1Loss()
        else:
            self.train_loss_fn = L1Loss()
            self.val_loss_fn = L1Loss()

        self.kspace_reconstruction_loss = cfg_dict.get("kspace_reconstruction_loss", False)
        self.reconstruction_loss_regularization_factor = cfg_dict.get("reconstruction_loss_regularization_factor", 1.0)
        if self.n2r:
            self.n2r_loss_regularization_factor = cfg_dict.get("n2r_loss_regularization_factor", 1.0)
        self.unnormalize_loss_inputs = cfg_dict.get("unnormalize_loss_inputs", False)
        self.normalization_type = cfg_dict.get("normalization_type", "max")

        self.accumulate_predictions = cfg_dict.get("accumulate_predictions", False)

        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals: Dict = defaultdict(dict)
        self.nmse_vals: Dict = defaultdict(dict)
        self.ssim_vals: Dict = defaultdict(dict)
        self.psnr_vals: Dict = defaultdict(dict)

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
                        "mean": attrs["prediction_mean"]
                        if "prediction_mean" in attrs
                        else attrs[f"prediction_mean_{r}"],
                        "std": attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"],
                    },
                    self.normalization_type,
                )
                prediction = utils.unnormalize(
                    prediction,
                    {
                        "min": attrs["noise_prediction_min"]
                        if "noise_prediction_min" in attrs
                        else attrs[f"noise_prediction_min_{r}"],
                        "max": attrs["noise_prediction_max"]
                        if "noise_prediction_max" in attrs
                        else attrs[f"noise_prediction_max_{r}"],
                        attrs["noise_prediction_mean"]
                        if "noise_prediction_mean" in attrs
                        else "mean": attrs[f"noise_prediction_mean_{r}"],
                        attrs["noise_prediction_std"]
                        if "noise_prediction_std" in attrs
                        else "std": attrs[f"noise_prediction_std_{r}"],
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
                        "mean": attrs["prediction_mean"]
                        if "prediction_mean" in attrs
                        else attrs[f"prediction_mean_{r}"],
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

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'kspace' : List of torch.Tensor
                Fully-sampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'init_pred' : Union[torch.Tensor, None]
                Initial prediction. Shape [batch_size, 1, n_x, n_y, 2] or None.
            'target' : Union[torch.Tensor, None]
                Target data. Shape [batch_size, n_x, n_y] or None.
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
        kspace, y, sensitivity_maps, mask, init_pred, target, _, _, acc, attrs = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_pred, n2r_init_pred = init_pred  # type: ignore

        kspace, y, mask, init_pred, target, r = self.process_inputs(kspace, y, mask, init_pred, target)

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            if isinstance(n2r_y, list):  # type: ignore
                n2r_y = n2r_y[r]  # type: ignore
                n2r_mask = n2r_mask[r]  # type: ignore
                n2r_init_pred = n2r_init_pred[r]  # type: ignore
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

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        preds_n2r = None
        if self.n2r and n2r_mask is not None:
            preds_n2r = self.forward(n2r_y, sensitivity_maps, n2r_mask, n2r_init_pred, target)

        target = utils.real_to_complex_tensor_or_list(target)  # type: ignore

        if self.accumulate_predictions:
            try:
                preds = next(preds)
                if preds_n2r is not None:
                    preds_n2r = next(preds_n2r)
            except StopIteration:
                pass

            preds = utils.real_to_complex_tensor_or_list(preds)

            if preds_n2r is not None:
                preds_n2r = utils.real_to_complex_tensor_or_list(preds_n2r)
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    train_loss = sum(
                        self.process_reconstruction_loss(
                            target,
                            preds,
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
                    train_loss = self.n2r_loss_regularization_factor * sum(
                        self.process_reconstruction_loss(
                            preds,
                            preds_n2r,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.train_loss_fn,
                            kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                        )
                    )
            else:
                preds = utils.real_to_complex_tensor_or_list(preds)
                train_loss = sum(
                    self.process_reconstruction_loss(
                        target,
                        preds,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    )
                )

        else:
            if preds_n2r is not None:
                preds_n2r = utils.real_to_complex_tensor_or_list(preds_n2r)
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    train_loss = self.process_reconstruction_loss(
                        target,
                        preds,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                        kspace_reconstruction_loss=self.ssdu,
                    )
                elif not attrs["n2r_supervised"]:  # type: ignore
                    train_loss = self.n2r_loss_regularization_factor * self.process_reconstruction_loss(
                        preds,
                        preds_n2r,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    )
            else:
                preds = utils.real_to_complex_tensor_or_list(preds)
                train_loss = self.process_reconstruction_loss(
                    target,
                    preds,
                    sensitivity_maps,
                    loss_mask,
                    attrs,  # type: ignore
                    r,
                    loss_func=self.train_loss_fn,
                    kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                )

        acc = r if r != 0 else acc
        tensorboard_logs = {
            f"train_loss_{acc}x": train_loss.item(),  # type: ignore
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
            'kspace' : List of torch.Tensor
                Fully-sampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'init_pred' : Union[torch.Tensor, None]
                Initial prediction. Shape [batch_size, 1, n_x, n_y, 2] or None.
            'target' : Union[torch.Tensor, None]
                Target data. Shape [batch_size, n_x, n_y] or None.
            'fname' : str
                File name.
            'slice_num' : int
                Slice number.
            'attrs' : float
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
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _, attrs = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_pred, n2r_init_pred = init_pred  # type: ignore

        kspace, y, mask, init_pred, target, r = self.process_inputs(kspace, y, mask, init_pred, target)

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            if isinstance(n2r_y, list):  # type: ignore
                n2r_y = n2r_y[r]  # type: ignore
                n2r_mask = n2r_mask[r]  # type: ignore
                n2r_init_pred = n2r_init_pred[r]  # type: ignore
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

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        preds_n2r = None
        if self.n2r and n2r_mask is not None:
            preds_n2r = self.forward(n2r_y, sensitivity_maps, n2r_mask, n2r_init_pred, target)

        target = utils.real_to_complex_tensor_or_list(target)  # type: ignore

        if self.accumulate_predictions:
            try:
                preds = next(preds)
                if preds_n2r is not None:
                    preds_n2r = next(preds_n2r)
            except StopIteration:
                pass

            preds = utils.real_to_complex_tensor_or_list(preds)

            if preds_n2r is not None:
                preds_n2r = utils.real_to_complex_tensor_or_list(preds_n2r)
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    # val_loss = self.n2r_loss_regularization_factor * sum(
                    #     self.process_reconstruction_loss(
                    #         preds,
                    #         preds_n2r,
                    #         sensitivity_maps,
                    #         loss_mask,
                    #         loss_func=self.train_loss_fn,
                    #         kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    #     )
                    # ) + sum(
                    #     self.process_reconstruction_loss(
                    #         target,
                    #         preds,
                    #         sensitivity_maps,
                    #         loss_mask,
                    #         loss_func=self.train_loss_fn,
                    #         # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                    #         kspace_reconstruction_loss=self.ssdu,
                    #     )
                    # )
                    val_loss = sum(
                        self.process_reconstruction_loss(
                            target,
                            preds,
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
                    val_loss = self.n2r_loss_regularization_factor * sum(
                        self.process_reconstruction_loss(
                            preds,
                            preds_n2r,
                            sensitivity_maps,
                            loss_mask,
                            attrs,  # type: ignore
                            r,
                            loss_func=self.train_loss_fn,
                            kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                        )
                    )
            else:
                preds = utils.real_to_complex_tensor_or_list(preds)
                val_loss = sum(
                    self.process_reconstruction_loss(
                        target,
                        preds,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    )
                )

        else:
            if preds_n2r is not None:
                preds_n2r = utils.real_to_complex_tensor_or_list(preds_n2r)
                if self.ssdu or attrs["n2r_supervised"]:  # type: ignore
                    # val_loss = self.n2r_loss_regularization_factor * self.process_reconstruction_loss(
                    #     preds,
                    #     preds_n2r,
                    #     sensitivity_maps,
                    #     loss_mask,
                    #     loss_func=self.train_loss_fn,
                    #     kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    # ) + self.process_reconstruction_loss(
                    #     target,
                    #     preds,
                    #     sensitivity_maps,
                    #     loss_mask,
                    #     loss_func=self.train_loss_fn,
                    #     # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                    #     kspace_reconstruction_loss=self.ssdu,
                    # )
                    val_loss = self.process_reconstruction_loss(
                        target,
                        preds,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        # in case of fully unsupervised n2r, we want to use the ssdu loss as pseudo-supervised loss
                        kspace_reconstruction_loss=self.ssdu,
                    )
                elif not attrs["n2r_supervised"]:  # type: ignore
                    val_loss = self.n2r_loss_regularization_factor * self.process_reconstruction_loss(
                        preds,
                        preds_n2r,
                        sensitivity_maps,
                        loss_mask,
                        attrs,  # type: ignore
                        r,
                        loss_func=self.train_loss_fn,
                        kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                    )
            else:
                preds = utils.real_to_complex_tensor_or_list(preds)
                val_loss = self.process_reconstruction_loss(
                    target,
                    preds,
                    sensitivity_maps,
                    loss_mask,
                    attrs,  # type: ignore
                    r,
                    loss_func=self.train_loss_fn,
                    kspace_reconstruction_loss=self.kspace_reconstruction_loss,
                )

        preds = utils.real_to_complex_tensor_or_list(preds)
        if isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

        target = utils.real_to_complex_tensor_or_list(target)  # type: ignore

        target = target.unsqueeze(1)  # type: ignore
        preds = preds.unsqueeze(1)  # type: ignore
        for batch_idx in range(target.shape[0]):  # type: ignore
            output = torch.abs(preds[batch_idx]).detach().cpu()
            _target = torch.abs(target[batch_idx]).detach().cpu()  # type: ignore
            output = output / output.max()  # type: ignore
            _target = _target / _target.max()  # type: ignore

            key = f"{fname[batch_idx]}_images_idx_{int(slice_num[batch_idx])}"  # type: ignore

            if self.log_images:
                error = torch.abs(_target - output)
                self.log_image(f"{key}/target", _target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

            _target = _target.numpy()  # type: ignore
            output = output.numpy()  # type: ignore
            self.mse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.mse(_target, output)).view(1)
            self.nmse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.nmse(_target, output)).view(1)
            self.ssim_vals[fname][slice_num] = torch.tensor(
                reconstruction_metrics.ssim(_target, output, maxval=output.max() - output.min())
            ).view(1)
            self.psnr_vals[fname][slice_num] = torch.tensor(
                reconstruction_metrics.psnr(_target, output, maxval=output.max() - output.min())
            ).view(1)

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
            'kspace' : List of torch.Tensor
                Fully-sampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'init_pred' : Union[torch.Tensor, None]
                Initial prediction. Shape [batch_size, 1, n_x, n_y, 2] or None.
            'target' : Union[torch.Tensor, None]
                Target data. Shape [batch_size, n_x, n_y] or None.
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
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _, attrs = batch

        if self.n2r and (not attrs["n2r_supervised"].all() or self.ssdu):  # type: ignore
            y, n2r_y = y  # type: ignore
            mask, n2r_mask = mask  # type: ignore
            init_pred, n2r_init_pred = init_pred  # type: ignore

        kspace, y, mask, init_pred, target, r = self.process_inputs(kspace, y, mask, init_pred, target)

        if self.ssdu:
            mask, _ = mask  # type: ignore

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_predictions:
            try:
                preds = next(preds)
            except StopIteration:
                pass

        preds = utils.real_to_complex_tensor_or_list(preds)
        if isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

        target = utils.real_to_complex_tensor_or_list(target)  # type: ignore

        target = target.unsqueeze(1)  # type: ignore
        preds = preds.unsqueeze(1)  # type: ignore
        for batch_idx in range(target.shape[0]):  # type: ignore
            output = torch.abs(preds[batch_idx]).detach().cpu()
            _target = torch.abs(target[batch_idx]).detach().cpu()  # type: ignore
            output = output / output.max()  # type: ignore
            _target = _target / _target.max()  # type: ignore

            name = str(fname[batch_idx])  # type: ignore
            slice_num = int(slice_num[batch_idx])  # type: ignore
            key = f"{name}_images_idx_{slice_num}"  # type: ignore

            if self.log_images:
                error = torch.abs(_target - output)
                self.log_image(f"{key}/target", _target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)

            _target = _target.numpy()  # type: ignore
            output = output.numpy()  # type: ignore
            self.mse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.mse(_target, output)).view(1)
            self.nmse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.nmse(_target, output)).view(1)
            self.ssim_vals[fname][slice_num] = torch.tensor(
                reconstruction_metrics.ssim(_target, output, maxval=output.max() - output.min())
            ).view(1)
            self.psnr_vals[fname][slice_num] = torch.tensor(
                reconstruction_metrics.psnr(_target, output, maxval=output.max() - output.min())
            ).view(1)

        return name, slice_num, preds.detach().cpu().numpy()  # type: ignore

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs : dict
            List of outputs from the validation step.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)

        # apply means across image volumes
        metrics = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}", value / tot_examples, sync_dist=True)

    def test_epoch_end(self, outputs):  # noqa: W0221
        """
        Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Parameters
        ----------
        outputs : dict
            List of outputs of the train batches.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k, v in self.mse_vals.items():
            mse_vals[k].update(v)
        for k, v in self.nmse_vals.items():
            nmse_vals[k].update(v)
        for k, v in self.ssim_vals.items():
            ssim_vals[k].update(v)
        for k, v in self.psnr_vals.items():
            psnr_vals[k].update(v)

        # apply means across image volumes
        metrics = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}", value / tot_examples, sync_dist=True)

        reconstructions = defaultdict(list)
        for fname, slice_num, output in outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])  # type: ignore

        out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)
        for fname, recons in reconstructions.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("reconstruction", data=recons)

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

        dataset = mri_reconstruction_loader.ReconstructionMRIDataset(
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
            n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
            transform=reconstruction_transforms.ReconstructionMRIDataTransforms(
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
                ssdu_outer_kspace_fraction=cfg.get("ssdu_outer_kspace_fraction", 0.0),
                ssdu_export_and_reuse_masks=cfg.get("ssdu_export_and_reuse_masks", False),
                n2r=cfg.get("n2r", False),
                n2r_supervised_rate=cfg.get("n2r_supervised_rate", 0.0),
                n2r_probability=cfg.get("n2r_probability", 0.5),
                n2r_std_devs=cfg.get("n2r_std_devs", (0.0, 0.0)),
                n2r_rhos=cfg.get("n2r_rhos", (0.4, 0.4)),
                n2r_use_mask=cfg.get("n2r_use_mask", True),
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
