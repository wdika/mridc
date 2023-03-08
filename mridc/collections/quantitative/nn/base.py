# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader

import mridc.collections.quantitative.parts.transforms as quantitative_transforms
import mridc.collections.reconstruction.losses as reconstruction_losses
import mridc.collections.reconstruction.nn.base as base_reconstruction_models
from mridc.collections.common.data import subsample
from mridc.collections.common.nn.base import BaseMRIModel, BaseSensitivityModel
from mridc.collections.common.parts import fft, utils
from mridc.collections.quantitative.data import qmri_loader
from mridc.collections.reconstruction.metrics import reconstruction_metrics

__all__ = ["BaseqMRIReconstructionModel", "SignalForwardModel"]


class BaseqMRIReconstructionModel(BaseMRIModel, ABC):  # type: ignore
    """
    Base class of all quantitative MRIReconstruction models.

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

        self.fft_centered = cfg_dict.get("fft_centered", False)
        self.fft_normalization = cfg_dict.get("fft_normalization", "backward")
        self.spatial_dims = cfg_dict.get("spatial_dims", None)
        self.coil_dim = cfg_dict.get("coil_dim", 1)

        self.coil_combination_method = cfg_dict.get("coil_combination_method", "SENSE")

        self.ssdu = cfg_dict.get("ssdu", False)

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

        if self.ssdu:
            self.kspace_reconstruction_loss = True
        else:
            self.kspace_reconstruction_loss = cfg_dict.get("kspace_reconstruction_loss", False)

        self.reconstruction_loss_regularization_factor = cfg_dict.get("reconstruction_loss_regularization_factor", 1.0)

        loss_regularization_factors = cfg_dict.get("loss_regularization_factors")
        self.loss_regularization_factors = {
            "R2star": loss_regularization_factors[0]["R2star"],
            "S0": loss_regularization_factors[1]["S0"],
            "B0": loss_regularization_factors[2]["B0"],
            "phi": loss_regularization_factors[3]["phi"],
        }

        self.MSE = base_reconstruction_models.DistributedMetricSum()
        self.NMSE = base_reconstruction_models.DistributedMetricSum()
        self.SSIM = base_reconstruction_models.DistributedMetricSum()
        self.PSNR = base_reconstruction_models.DistributedMetricSum()
        self.TotExamples = base_reconstruction_models.DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals_reconstruction: Dict = defaultdict(dict)
        self.nmse_vals_reconstruction: Dict = defaultdict(dict)
        self.ssim_vals_reconstruction: Dict = defaultdict(dict)
        self.psnr_vals_reconstruction: Dict = defaultdict(dict)

        self.mse_vals_R2star: Dict = defaultdict(dict)
        self.nmse_vals_R2star: Dict = defaultdict(dict)
        self.ssim_vals_R2star: Dict = defaultdict(dict)
        self.psnr_vals_R2star: Dict = defaultdict(dict)

        self.mse_vals_S0: Dict = defaultdict(dict)
        self.nmse_vals_S0: Dict = defaultdict(dict)
        self.ssim_vals_S0: Dict = defaultdict(dict)
        self.psnr_vals_S0: Dict = defaultdict(dict)

        self.mse_vals_B0: Dict = defaultdict(dict)
        self.nmse_vals_B0: Dict = defaultdict(dict)
        self.ssim_vals_B0: Dict = defaultdict(dict)
        self.psnr_vals_B0: Dict = defaultdict(dict)

        self.mse_vals_phi: Dict = defaultdict(dict)
        self.nmse_vals_phi: Dict = defaultdict(dict)
        self.ssim_vals_phi: Dict = defaultdict(dict)
        self.psnr_vals_phi: Dict = defaultdict(dict)

    def process_quantitative_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        mask_brain: torch.Tensor,
        quantitative_map: str,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Processes the quantitative loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        mask_brain : torch.Tensor
            Mask for brain of shape [batch_size, n_x, n_y, 1].
        quantitative_map : str
            Type of quantitative map to regularize the loss. Must be one of {"R2star", "S0", "B0", "phi"}.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if "ssim" in str(loss_func).lower():

            def compute_quantitative_loss(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.FloatTensor:
                """
                Wrapper for SSIM loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].
                m : torch.Tensor
                    Mask of shape [batch_size, n_x, n_y, 1].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                max_value = torch.max(torch.abs(y)) - torch.min(torch.abs(y)).unsqueeze(dim=0)
                m = torch.abs(m).to(x)

                loss = (
                    loss_func(x * m, y * m, data_range=max_value) * self.loss_regularization_factors[quantitative_map]
                )
                return loss

        else:

            def compute_quantitative_loss(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.FloatTensor:
                """
                Wrapper for any (expect the SSIM) loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].
                m : torch.Tensor
                    Mask of shape [batch_size, n_x, n_y, 1].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                m = torch.abs(m).to(x)

                if "mse" in str(loss_func).lower():
                    x = x.float()
                    y = y.float()
                    m = m.float()
                return loss_func(x * m, y * m) / self.loss_regularization_factors[quantitative_map]

        return compute_quantitative_loss(target, prediction, mask_brain)

    def process_reconstruction_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        loss_func: torch.nn.Module,
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
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if not self.kspace_reconstruction_loss:
            target = torch.abs(target / torch.max(torch.abs(target)))
        else:
            if target.shape[-1] != 2:
                target = torch.view_as_real(target)
            if self.ssdu:
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
                if not self.kspace_reconstruction_loss:
                    y = torch.abs(y / torch.max(torch.abs(y)))
                else:
                    if y.shape[-1] != 2:
                        y = torch.view_as_real(y)
                    if self.ssdu:
                        y = utils.expand_op(y, sensitivity_maps, self.coil_dim)
                    y = fft.fft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                    if self.ssdu:
                        y = y * mask
                return loss_func(x, y)

        return compute_reconstruction_loss(target, prediction) * self.reconstruction_loss_regularization_factor

    @staticmethod
    def process_inputs(  # noqa: W0221
        R2star_map_init: Union[list, torch.Tensor],
        S0_map_init: Union[list, torch.Tensor],
        B0_map_init: Union[list, torch.Tensor],
        phi_map_init: Union[list, torch.Tensor],
        kspace: Union[list, torch.Tensor],
        target: Union[list, torch.Tensor],
        y: Union[list, torch.Tensor],
        mask: Union[list, torch.Tensor],
    ) -> tuple[
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Union[list, Tensor], Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[Tensor, Any],
        Union[Union[list, Tensor], Any],
        Union[int, ndarray[int]],
    ]:
        """
        Processes lists of inputs to torch.Tensor. In the case where multiple accelerations are used, then the inputs
        are lists. This function converts the lists to torch.Tensor by randomly selecting one acceleration. If only one
        acceleration is used, then the inputs are torch.Tensor and are returned as is.

        Parameters
        ----------
        R2star_map_init : Union[list, torch.Tensor]
            R2* map of length n_accelerations or shape [batch_size, n_x, n_y].
        S0_map_init : Union[list, torch.Tensor]
            S0 map of length n_accelerations or shape [batch_size, n_x, n_y].
        B0_map_init : Union[list, torch.Tensor]
            B0 map of length n_accelerations or shape [batch_size, n_x, n_y].
        phi_map_init : Union[list, torch.Tensor]
            Phi map of length n_accelerations or shape [batch_size, n_x, n_y].
        kspace : Union[list, torch.Tensor]
            Full k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : Union[list, torch.Tensor]
            Target data of length n_accelerations or shape [batch_size, n_x, n_y, 2].
        y : Union[list, torch.Tensor]
            Subsampled k-space data of length n_accelerations or shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : Union[list, torch.Tensor]
            Sampling mask of length n_accelerations or shape [batch_size, 1, n_x, n_y, 1].

        Returns
        -------
        R2star_map_init : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map_init : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map_init : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map_init : torch.Tensor
            Phi map of shape [batch_size, n_x, n_y].
        kspace : torch.Tensor
            Full k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        r : int
            Random index used to select the acceleration.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            R2star_map_init = R2star_map_init[r]
            S0_map_init = S0_map_init[r]
            B0_map_init = B0_map_init[r]
            phi_map_init = phi_map_init[r]
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        if isinstance(kspace, list):
            kspace = kspace[r]
            target = target[r]
        return (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            mask,
            r,
        )

    @staticmethod
    def __check_if_isinstance_pred__(x):
        """Checks if x is a list of a list of predictions."""
        if isinstance(x, list):
            x = x[-1]
        if isinstance(x, list):
            x = x[-1]
        return x

    def training_step(  # noqa: W0221
        self, batch: Dict[float, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'R2star_map_init' : List of torch.Tensor
                R2* initial map. Shape [batch_size,  n_x, n_y].
            'R2star_map_target' : torch.Tensor
                R2* target map. Shape [batch_size,  n_x, n_y].
            'S0_map_init' : List of torch.Tensor
                S0 initial map. Shape [batch_size,  n_x, n_y].
            'S0_map_target' : torch.Tensor
                S0 target map. Shape [batch_size,  n_x, n_y].
            'B0_map_init' : List of torch.Tensor
                B0 initial map. Shape [batch_size,  n_x, n_y].
            'B0_map_target' : torch.Tensor
                B0 target map. Shape [batch_size,  n_x, n_y].
            'phi_map_init' : List of torch.Tensor
                Phi initial map. Shape [batch_size,  n_x, n_y].
            'phi_map_target' : torch.Tensor
                Phi target map. Shape [batch_size,  n_x, n_y].
            'TEs' : List of float
                Echo times.
            'kspace' : List of torch.Tensor
                Full k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'mask_brain' : torch.Tensor
                Brain mask. Shape [n_x, n_y].
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
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            mask,
            mask_brain,
            _,
            target,
            _,
            _,
            acc,
        ) = batch

        (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            sampling_mask,
            r,
        ) = self.process_inputs(  # type: ignore
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            mask,
        )

        if self.ssdu:
            mask, loss_mask = mask  # type: ignore
        else:
            loss_mask = torch.ones_like(mask)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, sampling_mask)

        preds = self.forward(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs.tolist()[0],  # type: ignore
            y,
            sensitivity_maps,
            torch.ones_like(mask_brain),
            sampling_mask,
        )

        if self.accumulate_predictions:
            try:
                preds = next(preds)
            except StopIteration:
                pass

            recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
            )

            if self.use_reconstruction_module:
                lossrecon = sum(
                    self.process_reconstruction_loss(
                        target, recon_pred, sensitivity_maps, loss_mask, self.train_loss_fn
                    )
                )
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = sum(
                self.process_quantitative_loss(
                    R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.train_loss_fn
                )
            )
            lossS0 = sum(
                self.process_quantitative_loss(S0_map_target, S0_map_pred, mask_brain, "S0", self.train_loss_fn)
            )
            lossB0 = sum(
                self.process_quantitative_loss(B0_map_target, B0_map_pred, mask_brain, "B0", self.train_loss_fn)
            )
            lossPhi = sum(
                self.process_quantitative_loss(phi_map_target, phi_map_pred, mask_brain, "phi", self.train_loss_fn)
            )
        else:
            recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
            )

            if self.use_reconstruction_module:
                lossrecon = self.process_reconstruction_loss(
                    target, recon_pred, sensitivity_maps, loss_mask, self.train_loss_fn
                ).mean()
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = self.process_quantitative_loss(
                R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.train_loss_fn
            ).sum()
            lossS0 = self.process_quantitative_loss(
                S0_map_target, S0_map_pred, mask_brain, "S0", self.train_loss_fn
            ).sum()
            lossB0 = self.process_quantitative_loss(
                B0_map_target, B0_map_pred, mask_brain, "B0", self.train_loss_fn
            ).sum()
            lossPhi = self.process_quantitative_loss(
                phi_map_target, phi_map_pred, mask_brain, "phi", self.train_loss_fn
            ).sum()

        train_loss = sum([lossR2star, lossS0, lossB0, lossPhi]) / 4
        train_loss = train_loss.mean() / 2

        if self.use_reconstruction_module:
            train_loss = train_loss + lossrecon

        self.acc = r if r != 0 else acc  # type: ignore
        tensorboard_logs = {
            f"train_loss_{self.acc}x": train_loss.item(),  # type: ignore
            f"loss_reconstruction_{self.acc}x": lossrecon.item(),  # type: ignore
            f"loss_R2star_{self.acc}x": lossR2star.item(),  # type: ignore
            f"loss_S0_{self.acc}x": lossS0.item(),  # type: ignore
            f"loss_B0_{self.acc}x": lossB0.item(),  # type: ignore
            f"loss_phi_{self.acc}x": lossPhi.item(),  # type: ignore
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
            'R2star_map_init' : List of torch.Tensor
                R2* initial map. Shape [batch_size,  n_x, n_y].
            'R2star_map_target' : torch.Tensor
                R2* target map. Shape [batch_size,  n_x, n_y].
            'S0_map_init' : List of torch.Tensor
                S0 initial map. Shape [batch_size,  n_x, n_y].
            'S0_map_target' : torch.Tensor
                S0 target map. Shape [batch_size,  n_x, n_y].
            'B0_map_init' : List of torch.Tensor
                B0 initial map. Shape [batch_size,  n_x, n_y].
            'B0_map_target' : torch.Tensor
                B0 target map. Shape [batch_size,  n_x, n_y].
            'phi_map_init' : List of torch.Tensor
                Phi initial map. Shape [batch_size,  n_x, n_y].
            'phi_map_target' : torch.Tensor
                Phi target map. Shape [batch_size,  n_x, n_y].
            'TEs' : List of float
                Echo times.
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'mask_brain' : torch.Tensor
                Brain mask. Shape [n_x, n_y].
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
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of loss and log.
        """
        (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            mask,
            mask_brain,
            _,
            target,
            fname,
            slice_num,
            _,
        ) = batch

        (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            sampling_mask,
            _,
        ) = self.process_inputs(  # type: ignore
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            mask,
        )

        if self.ssdu:
            mask, loss_mask = mask  # type: ignore
        else:
            loss_mask = torch.ones_like(mask)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, sampling_mask)

        preds = self.forward(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs.tolist()[0],  # type: ignore
            y,
            sensitivity_maps,
            torch.ones_like(mask_brain),
            sampling_mask,
        )

        if self.accumulate_predictions:
            try:
                preds = next(preds)
            except StopIteration:
                pass

            recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
            )

            if self.use_reconstruction_module:
                lossrecon = sum(
                    self.process_reconstruction_loss(target, recon_pred, sensitivity_maps, loss_mask, self.val_loss_fn)
                )
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = sum(
                self.process_quantitative_loss(
                    R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.val_loss_fn
                )
            )
            lossS0 = sum(
                self.process_quantitative_loss(S0_map_target, S0_map_pred, mask_brain, "S0", self.val_loss_fn)
            )
            lossB0 = sum(
                self.process_quantitative_loss(B0_map_target, B0_map_pred, mask_brain, "B0", self.val_loss_fn)
            )
            lossPhi = sum(
                self.process_quantitative_loss(phi_map_target, phi_map_pred, mask_brain, "phi", self.val_loss_fn)
            )
        else:
            recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                preds[0],
                preds[1],
                preds[2],
                preds[3],
                preds[4],
            )

            if self.use_reconstruction_module:
                lossrecon = self.process_reconstruction_loss(
                    target, recon_pred, sensitivity_maps, loss_mask, self.val_loss_fn
                ).mean()
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = self.process_quantitative_loss(
                R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.val_loss_fn
            ).sum()
            lossS0 = self.process_quantitative_loss(
                S0_map_target, S0_map_pred, mask_brain, "S0", self.val_loss_fn
            ).sum()
            lossB0 = self.process_quantitative_loss(
                B0_map_target, B0_map_pred, mask_brain, "B0", self.val_loss_fn
            ).sum()
            lossPhi = self.process_quantitative_loss(
                phi_map_target, phi_map_pred, mask_brain, "phi", self.val_loss_fn
            ).sum()

        val_loss = sum([lossR2star, lossS0, lossB0, lossPhi]) / 4
        val_loss = val_loss.mean() / 2

        if self.use_reconstruction_module:
            val_loss = val_loss + lossrecon

            if isinstance(recon_pred, list):
                recon_pred = torch.stack([self.__check_if_isinstance_pred__(x) for x in recon_pred], dim=1)

            recon_pred = torch.stack(
                [
                    self.__check_if_isinstance_pred__(recon_pred[:, echo_time, :, :])
                    for echo_time in range(recon_pred.shape[1])
                ],
                1,
            )

            recon_pred = recon_pred.detach().cpu()
            recon_pred = torch.abs(recon_pred / torch.max(torch.abs(recon_pred)))
            target = target.detach().cpu()  # type: ignore
            target = torch.abs(target / torch.max(torch.abs(target)))

        R2star_map_pred = self.__check_if_isinstance_pred__(R2star_map_pred)
        S0_map_pred = self.__check_if_isinstance_pred__(S0_map_pred)
        B0_map_pred = self.__check_if_isinstance_pred__(B0_map_pred)
        phi_map_pred = self.__check_if_isinstance_pred__(phi_map_pred)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        R2star_map_output = R2star_map_pred.detach().cpu()
        R2star_map_output = torch.abs(R2star_map_output / torch.max(torch.abs(R2star_map_output)))
        S0_map_output = S0_map_pred.detach().cpu()
        S0_map_output = torch.abs(S0_map_output / torch.max(torch.abs(S0_map_output)))
        B0_map_output = B0_map_pred.detach().cpu()
        B0_map_output = torch.abs(B0_map_output / torch.max(torch.abs(B0_map_output)))
        phi_map_output = phi_map_pred.detach().cpu()
        phi_map_output = torch.abs(phi_map_output / torch.max(torch.abs(phi_map_output)))
        R2star_map_target = R2star_map_target.detach().cpu()  # type: ignore
        R2star_map_target = torch.abs(R2star_map_target / torch.max(torch.abs(R2star_map_target)))
        S0_map_target = S0_map_target.detach().cpu()  # type: ignore
        S0_map_target = torch.abs(S0_map_target / torch.max(torch.abs(S0_map_target)))
        B0_map_target = B0_map_target.detach().cpu()  # type: ignore
        B0_map_target = torch.abs(B0_map_target / torch.max(torch.abs(B0_map_target)))
        phi_map_target = phi_map_target.detach().cpu()  # type: ignore
        phi_map_target = torch.abs(phi_map_target / torch.max(torch.abs(phi_map_target)))

        if self.log_images:
            if self.use_reconstruction_module:
                for echo_time in range(target.shape[1]):  # type: ignore
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/target",
                        target[:, echo_time, :, :],  # type: ignore
                    )  # type: ignore
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/reconstruction", recon_pred[:, echo_time, :, :]
                    )
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/error",
                        torch.abs(target[:, echo_time, :, :] - recon_pred[:, echo_time, :, :]),  # type: ignore
                    )

            self.log_image(f"{key}/R2star/target", R2star_map_target)
            self.log_image(f"{key}/R2star/reconstruction", R2star_map_output)
            self.log_image(f"{key}/R2star/error", torch.abs(R2star_map_target - R2star_map_output))

            self.log_image(f"{key}/S0/target", S0_map_target)
            self.log_image(f"{key}/S0/reconstruction", S0_map_output)
            self.log_image(f"{key}/S0/error", S0_map_target - S0_map_output)

            self.log_image(f"{key}/B0/target", B0_map_target)
            self.log_image(f"{key}/B0/reconstruction", B0_map_output)
            self.log_image(f"{key}/B0/error", torch.abs(B0_map_target - B0_map_output))

            self.log_image(f"{key}/phi/target", phi_map_target)
            self.log_image(f"{key}/phi/reconstruction", phi_map_output)
            self.log_image(f"{key}/phi/error", phi_map_target - phi_map_output)

        if self.use_reconstruction_module:
            recon_pred = recon_pred.numpy()  # type: ignore
            target = target.numpy()  # type: ignore

            mses = []
            nmses = []
            ssims = []
            psnrs = []
            for echo_time in range(target.shape[1]):  # type: ignore
                mses.append(
                    torch.tensor(
                        reconstruction_metrics.mse(
                            target[:, echo_time, ...], recon_pred[:, echo_time, ...]  # type: ignore
                        )
                    ).view(1)
                )
                nmses.append(
                    torch.tensor(
                        reconstruction_metrics.nmse(
                            target[:, echo_time, ...], recon_pred[:, echo_time, ...]  # type: ignore
                        )
                    ).view(1)
                )
                ssims.append(
                    torch.tensor(
                        reconstruction_metrics.ssim(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )
                psnrs.append(
                    torch.tensor(
                        reconstruction_metrics.psnr(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )

                self.mse_vals_reconstruction[fname][slice_num] = torch.tensor(mses).mean()
                self.nmse_vals_reconstruction[fname][slice_num] = torch.tensor(nmses).mean()
                self.ssim_vals_reconstruction[fname][slice_num] = torch.tensor(ssims).mean()
                self.psnr_vals_reconstruction[fname][slice_num] = torch.tensor(psnrs).mean()
        else:
            self.mse_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.nmse_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.ssim_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.psnr_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)

        R2star_map_output = R2star_map_output.numpy()  # type: ignore
        S0_map_output = S0_map_output.numpy()  # type: ignore
        B0_map_output = B0_map_output.numpy()  # type: ignore
        phi_map_output = phi_map_output.numpy()  # type: ignore

        R2star_map_target = R2star_map_target.numpy()  # type: ignore
        S0_map_target = S0_map_target.numpy()  # type: ignore
        B0_map_target = B0_map_target.numpy()  # type: ignore
        phi_map_target = phi_map_target.numpy()  # type: ignore

        self.mse_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(R2star_map_target, R2star_map_output)
        ).view(1)
        self.nmse_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(R2star_map_target, R2star_map_output)
        ).view(1)
        self.ssim_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(
                R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min()
            )
        ).view(1)
        self.psnr_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(
                R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min()
            )
        ).view(1)

        self.mse_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(S0_map_target, S0_map_output)
        ).view(1)
        self.nmse_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(S0_map_target, S0_map_output)
        ).view(1)
        self.ssim_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)
        self.psnr_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)

        self.mse_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(B0_map_target, B0_map_output)
        ).view(1)
        self.nmse_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(B0_map_target, B0_map_output)
        ).view(1)
        self.ssim_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)
        self.psnr_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)

        self.mse_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(phi_map_target, phi_map_output)
        ).view(1)
        self.nmse_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(phi_map_target, phi_map_output)
        ).view(1)
        self.ssim_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(
                phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min()
            )
        ).view(1)
        self.psnr_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(
                phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min()
            )
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
            'R2star_map_init' : List of torch.Tensor
                R2* initial map. Shape [batch_size,  n_x, n_y].
            'R2star_map_target' : torch.Tensor
                R2* target map. Shape [batch_size,  n_x, n_y].
            'S0_map_init' : List of torch.Tensor
                S0 initial map. Shape [batch_size,  n_x, n_y].
            'S0_map_target' : torch.Tensor
                S0 target map. Shape [batch_size,  n_x, n_y].
            'B0_map_init' : List of torch.Tensor
                B0 initial map. Shape [batch_size,  n_x, n_y].
            'B0_map_target' : torch.Tensor
                B0 target map. Shape [batch_size,  n_x, n_y].
            'phi_map_init' : List of torch.Tensor
                Phi initial map. Shape [batch_size,  n_x, n_y].
            'phi_map_target' : torch.Tensor
                Phi target map. Shape [batch_size,  n_x, n_y].
            'TEs' : List of float
                Echo times.
            'y' : List of torch.Tensor
                Subsampled k-space data. Shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
            'sensitivity_maps' : torch.Tensor
                Coils sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2].
            'mask' : List of torch.Tensor
                Brain & sampling mask. Shape [batch_size, 1, n_x, n_y, 1].
            'mask_brain' : torch.Tensor
                Brain mask. Shape [n_x, n_y].
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
        batch_idx : int
            Batch index.

        Returns
        -------
        name : str
            File name.
        slice_num : int
            Slice number.
        pred : torch.Tensor
            Predicted data. Shape [batch_size, n_x, n_y, 2].
        """
        (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            TEs,
            kspace,
            y,
            sensitivity_maps,
            mask,
            mask_brain,
            _,
            target,
            fname,
            slice_num,
            _,
        ) = batch

        (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            sampling_mask,
            _,
        ) = self.process_inputs(  # type: ignore
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            kspace,
            target,
            y,
            mask,
        )

        if self.ssdu:
            mask, _ = mask  # type: ignore

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, sampling_mask)

        preds = self.forward(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            TEs.tolist()[0],  # type: ignore
            y,
            sensitivity_maps,
            torch.ones_like(mask_brain),
            sampling_mask,
        )

        if self.accumulate_predictions:
            try:
                preds = next(preds)
            except StopIteration:
                pass

        recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
            preds[0],
            preds[1],
            preds[2],
            preds[3],
            preds[4],
        )

        if self.use_reconstruction_module:
            if isinstance(recon_pred, list):
                recon_pred = torch.stack([self.__check_if_isinstance_pred__(x) for x in recon_pred], dim=1)

            recon_pred = torch.stack(
                [
                    self.__check_if_isinstance_pred__(recon_pred[:, echo_time, :, :])
                    for echo_time in range(recon_pred.shape[1])
                ],
                1,
            )

            recon_pred = recon_pred.detach().cpu()
            recon_pred = torch.abs(recon_pred / torch.max(torch.abs(recon_pred)))
            target = target.detach().cpu()  # type: ignore
            target = torch.abs(target / torch.max(torch.abs(target)))

        R2star_map_pred = self.__check_if_isinstance_pred__(R2star_map_pred)
        S0_map_pred = self.__check_if_isinstance_pred__(S0_map_pred)
        B0_map_pred = self.__check_if_isinstance_pred__(B0_map_pred)
        phi_map_pred = self.__check_if_isinstance_pred__(phi_map_pred)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        R2star_map_output = R2star_map_pred.detach().cpu()
        R2star_map_output = torch.abs(R2star_map_output / torch.max(torch.abs(R2star_map_output)))
        S0_map_output = S0_map_pred.detach().cpu()
        S0_map_output = torch.abs(S0_map_output / torch.max(torch.abs(S0_map_output)))
        B0_map_output = B0_map_pred.detach().cpu()
        B0_map_output = torch.abs(B0_map_output / torch.max(torch.abs(B0_map_output)))
        phi_map_output = phi_map_pred.detach().cpu()
        phi_map_output = torch.abs(phi_map_output / torch.max(torch.abs(phi_map_output)))
        R2star_map_target = R2star_map_target.detach().cpu()  # type: ignore
        R2star_map_target = torch.abs(R2star_map_target / torch.max(torch.abs(R2star_map_target)))
        S0_map_target = S0_map_target.detach().cpu()  # type: ignore
        S0_map_target = torch.abs(S0_map_target / torch.max(torch.abs(S0_map_target)))
        B0_map_target = B0_map_target.detach().cpu()  # type: ignore
        B0_map_target = torch.abs(B0_map_target / torch.max(torch.abs(B0_map_target)))
        phi_map_target = phi_map_target.detach().cpu()  # type: ignore
        phi_map_target = torch.abs(phi_map_target / torch.max(torch.abs(phi_map_target)))

        if self.log_images:
            if self.use_reconstruction_module:
                for echo_time in range(target.shape[1]):  # type: ignore
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/target",
                        target[:, echo_time, :, :],  # type: ignore
                    )  # type: ignore
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/reconstruction", recon_pred[:, echo_time, :, :]
                    )
                    self.log_image(
                        f"{key}/reconstruction_echo_{echo_time}/error",
                        torch.abs(target[:, echo_time, :, :] - recon_pred[:, echo_time, :, :]),  # type: ignore
                    )

            self.log_image(f"{key}/R2star/target", R2star_map_target)
            self.log_image(f"{key}/R2star/reconstruction", R2star_map_output)
            self.log_image(f"{key}/R2star/error", torch.abs(R2star_map_target - R2star_map_output))

            self.log_image(f"{key}/S0/target", S0_map_target)
            self.log_image(f"{key}/S0/reconstruction", S0_map_output)
            self.log_image(f"{key}/S0/error", torch.abs(S0_map_target - S0_map_output))

            self.log_image(f"{key}/B0/target", B0_map_target)
            self.log_image(f"{key}/B0/reconstruction", B0_map_output)
            self.log_image(f"{key}/B0/error", torch.abs(B0_map_target - B0_map_output))

            self.log_image(f"{key}/phi/target", phi_map_target)
            self.log_image(f"{key}/phi/reconstruction", phi_map_output)
            self.log_image(f"{key}/phi/error", torch.abs(phi_map_target - phi_map_output))

        if self.use_reconstruction_module:
            recon_pred = recon_pred.numpy()  # type: ignore
            target = target.numpy()  # type: ignore

            mses = []
            nmses = []
            ssims = []
            psnrs = []
            for echo_time in range(target.shape[1]):  # type: ignore
                mses.append(
                    torch.tensor(
                        reconstruction_metrics.mse(
                            target[:, echo_time, ...], recon_pred[:, echo_time, ...]  # type: ignore
                        )
                    ).view(1)
                )
                nmses.append(
                    torch.tensor(
                        reconstruction_metrics.nmse(
                            target[:, echo_time, ...], recon_pred[:, echo_time, ...]  # type: ignore
                        )
                    ).view(1)
                )  # type: ignore
                ssims.append(
                    torch.tensor(
                        reconstruction_metrics.ssim(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )
                psnrs.append(
                    torch.tensor(
                        reconstruction_metrics.psnr(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )

                self.mse_vals_reconstruction[fname][slice_num] = torch.tensor(mses).mean()
                self.nmse_vals_reconstruction[fname][slice_num] = torch.tensor(nmses).mean()
                self.ssim_vals_reconstruction[fname][slice_num] = torch.tensor(ssims).mean()
                self.psnr_vals_reconstruction[fname][slice_num] = torch.tensor(psnrs).mean()
        else:
            self.mse_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.nmse_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.ssim_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)
            self.psnr_vals_reconstruction[fname][slice_num] = torch.tensor(0).view(1)

        R2star_map_output = R2star_map_output.numpy()  # type: ignore
        S0_map_output = S0_map_output.numpy()  # type: ignore
        B0_map_output = B0_map_output.numpy()  # type: ignore
        phi_map_output = phi_map_output.numpy()  # type: ignore

        R2star_map_target = R2star_map_target.numpy()  # type: ignore
        S0_map_target = S0_map_target.numpy()  # type: ignore
        B0_map_target = B0_map_target.numpy()  # type: ignore
        phi_map_target = phi_map_target.numpy()  # type: ignore

        self.mse_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(R2star_map_target, R2star_map_output)
        ).view(1)
        self.nmse_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(R2star_map_target, R2star_map_output)
        ).view(1)
        self.ssim_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(
                R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min()
            )
        ).view(1)
        self.psnr_vals_R2star[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(
                R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min()
            )
        ).view(1)

        self.mse_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(S0_map_target, S0_map_output)
        ).view(1)
        self.nmse_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(S0_map_target, S0_map_output)
        ).view(1)
        self.ssim_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)
        self.psnr_vals_S0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)

        self.mse_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(B0_map_target, B0_map_output)
        ).view(1)
        self.nmse_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(B0_map_target, B0_map_output)
        ).view(1)
        self.ssim_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)
        self.psnr_vals_B0[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)

        self.mse_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.mse(phi_map_target, phi_map_output)
        ).view(1)
        self.nmse_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.nmse(phi_map_target, phi_map_output)
        ).view(1)
        self.ssim_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(
                phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min()
            )
        ).view(1)
        self.psnr_vals_phi[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(
                phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min()
            )
        ).view(1)

        return (
            name,
            slice_num,
            torch.stack([R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred], dim=0).detach().cpu().numpy(),
        )

    def train_epoch_end(self, outputs):
        """
        Called at the end of train epoch to aggregate the loss values.

        Parameters
        ----------
        outputs : list
            List of outputs from the training step.
        """
        self.log("train_loss", torch.stack([x["train_loss"] for x in outputs]).mean())
        self.log(f"train_loss_{self.acc}x", torch.stack([x[f"train_loss_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_R2star_{self.acc}x", torch.stack([x[f"loss_R2star_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_S0_{self.acc}x", torch.stack([x[f"loss_S0_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_B0_{self.acc}x", torch.stack([x[f"loss_B0_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_phi_{self.acc}x", torch.stack([x[f"loss_phi_{self.acc}x"] for x in outputs]).mean())

    def validation_epoch_end(self, outputs):  # noqa: MC0001
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
        mse_vals_reconstruction = defaultdict(dict)
        nmse_vals_reconstruction = defaultdict(dict)
        ssim_vals_reconstruction = defaultdict(dict)
        psnr_vals_reconstruction = defaultdict(dict)

        mse_vals_R2star = defaultdict(dict)
        nmse_vals_R2star = defaultdict(dict)
        ssim_vals_R2star = defaultdict(dict)
        psnr_vals_R2star = defaultdict(dict)

        mse_vals_S0 = defaultdict(dict)
        nmse_vals_S0 = defaultdict(dict)
        ssim_vals_S0 = defaultdict(dict)
        psnr_vals_S0 = defaultdict(dict)

        mse_vals_B0 = defaultdict(dict)
        nmse_vals_B0 = defaultdict(dict)
        ssim_vals_B0 = defaultdict(dict)
        psnr_vals_B0 = defaultdict(dict)

        mse_vals_phi = defaultdict(dict)
        nmse_vals_phi = defaultdict(dict)
        ssim_vals_phi = defaultdict(dict)
        psnr_vals_phi = defaultdict(dict)

        for k, v in self.mse_vals_reconstruction.items():
            mse_vals_reconstruction[k].update(v)
        for k, v in self.nmse_vals_reconstruction.items():
            nmse_vals_reconstruction[k].update(v)
        for k, v in self.ssim_vals_reconstruction.items():
            ssim_vals_reconstruction[k].update(v)
        for k, v in self.psnr_vals_reconstruction.items():
            psnr_vals_reconstruction[k].update(v)

        for k, v in self.mse_vals_R2star.items():
            mse_vals_R2star[k].update(v)
        for k, v in self.nmse_vals_R2star.items():
            nmse_vals_R2star[k].update(v)
        for k, v in self.ssim_vals_R2star.items():
            ssim_vals_R2star[k].update(v)
        for k, v in self.psnr_vals_R2star.items():
            psnr_vals_R2star[k].update(v)

        for k, v in self.mse_vals_S0.items():
            mse_vals_S0[k].update(v)
        for k, v in self.nmse_vals_S0.items():
            nmse_vals_S0[k].update(v)
        for k, v in self.ssim_vals_S0.items():
            ssim_vals_S0[k].update(v)
        for k, v in self.psnr_vals_S0.items():
            psnr_vals_S0[k].update(v)

        for k, v in self.mse_vals_B0.items():
            mse_vals_B0[k].update(v)
        for k, v in self.nmse_vals_B0.items():
            nmse_vals_B0[k].update(v)
        for k, v in self.ssim_vals_B0.items():
            ssim_vals_B0[k].update(v)
        for k, v in self.psnr_vals_B0.items():
            psnr_vals_B0[k].update(v)

        for k, v in self.mse_vals_phi.items():
            mse_vals_phi[k].update(v)
        for k, v in self.nmse_vals_phi.items():
            nmse_vals_phi[k].update(v)
        for k, v in self.ssim_vals_phi.items():
            ssim_vals_phi[k].update(v)
        for k, v in self.psnr_vals_phi.items():
            psnr_vals_phi[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "NMSE": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "SSIM": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "PSNR": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
        }
        local_examples = 0
        for fname in mse_vals_R2star:
            local_examples += 1
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

            metrics["MSE"]["R2star"] = metrics["MSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_R2star[fname].items()])
            )
            metrics["NMSE"]["R2star"] = metrics["NMSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_R2star[fname].items()])
            )
            metrics["SSIM"]["R2star"] = metrics["SSIM"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_R2star[fname].items()])
            )
            metrics["PSNR"]["R2star"] = metrics["PSNR"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_R2star[fname].items()])
            )

            metrics["MSE"]["S0"] = metrics["MSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_S0[fname].items()])
            )
            metrics["NMSE"]["S0"] = metrics["NMSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_S0[fname].items()])
            )
            metrics["SSIM"]["S0"] = metrics["SSIM"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_S0[fname].items()])
            )
            metrics["PSNR"]["S0"] = metrics["PSNR"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_S0[fname].items()])
            )

            metrics["MSE"]["B0"] = metrics["MSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_B0[fname].items()])
            )
            metrics["NMSE"]["B0"] = metrics["NMSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_B0[fname].items()])
            )
            metrics["SSIM"]["B0"] = metrics["SSIM"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_B0[fname].items()])
            )
            metrics["PSNR"]["B0"] = metrics["PSNR"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_B0[fname].items()])
            )

            metrics["MSE"]["phi"] = metrics["MSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_phi[fname].items()])
            )
            metrics["NMSE"]["phi"] = metrics["NMSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_phi[fname].items()])
            )
            metrics["SSIM"]["phi"] = metrics["SSIM"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_phi[fname].items()])
            )
            metrics["PSNR"]["phi"] = metrics["PSNR"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_phi[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
        metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
        metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
        metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])

        metrics["MSE"]["R2star"] = self.MSE(metrics["MSE"]["R2star"])
        metrics["NMSE"]["R2star"] = self.NMSE(metrics["NMSE"]["R2star"])
        metrics["SSIM"]["R2star"] = self.SSIM(metrics["SSIM"]["R2star"])
        metrics["PSNR"]["R2star"] = self.PSNR(metrics["PSNR"]["R2star"])

        metrics["MSE"]["S0"] = self.MSE(metrics["MSE"]["S0"])
        metrics["NMSE"]["S0"] = self.NMSE(metrics["NMSE"]["S0"])
        metrics["SSIM"]["S0"] = self.SSIM(metrics["SSIM"]["S0"])
        metrics["PSNR"]["S0"] = self.PSNR(metrics["PSNR"]["S0"])

        metrics["MSE"]["B0"] = self.MSE(metrics["MSE"]["B0"])
        metrics["NMSE"]["B0"] = self.NMSE(metrics["NMSE"]["B0"])
        metrics["SSIM"]["B0"] = self.SSIM(metrics["SSIM"]["B0"])
        metrics["PSNR"]["B0"] = self.PSNR(metrics["PSNR"]["B0"])

        metrics["MSE"]["phi"] = self.MSE(metrics["MSE"]["phi"])
        metrics["NMSE"]["phi"] = self.NMSE(metrics["NMSE"]["phi"])
        metrics["SSIM"]["phi"] = self.SSIM(metrics["SSIM"]["phi"])
        metrics["PSNR"]["phi"] = self.PSNR(metrics["PSNR"]["phi"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_R2star", value["R2star"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_S0", value["S0"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_B0", value["B0"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_phi", value["phi"] / tot_examples, sync_dist=True)

    def test_epoch_end(self, outputs):  # noqa: MC0001
        """
        Called at the end of test epoch to aggregate outputs, log metrics and save predictions.

        Parameters
        ----------
        outputs : dict
            List of outputs from the validation step.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals_reconstruction = defaultdict(dict)
        nmse_vals_reconstruction = defaultdict(dict)
        ssim_vals_reconstruction = defaultdict(dict)
        psnr_vals_reconstruction = defaultdict(dict)

        mse_vals_R2star = defaultdict(dict)
        nmse_vals_R2star = defaultdict(dict)
        ssim_vals_R2star = defaultdict(dict)
        psnr_vals_R2star = defaultdict(dict)

        mse_vals_S0 = defaultdict(dict)
        nmse_vals_S0 = defaultdict(dict)
        ssim_vals_S0 = defaultdict(dict)
        psnr_vals_S0 = defaultdict(dict)

        mse_vals_B0 = defaultdict(dict)
        nmse_vals_B0 = defaultdict(dict)
        ssim_vals_B0 = defaultdict(dict)
        psnr_vals_B0 = defaultdict(dict)

        mse_vals_phi = defaultdict(dict)
        nmse_vals_phi = defaultdict(dict)
        ssim_vals_phi = defaultdict(dict)
        psnr_vals_phi = defaultdict(dict)

        for k, v in self.mse_vals_reconstruction.items():
            mse_vals_reconstruction[k].update(v)
        for k, v in self.nmse_vals_reconstruction.items():
            nmse_vals_reconstruction[k].update(v)
        for k, v in self.ssim_vals_reconstruction.items():
            ssim_vals_reconstruction[k].update(v)
        for k, v in self.psnr_vals_reconstruction.items():
            psnr_vals_reconstruction[k].update(v)

        for k, v in self.mse_vals_R2star.items():
            mse_vals_R2star[k].update(v)
        for k, v in self.nmse_vals_R2star.items():
            nmse_vals_R2star[k].update(v)
        for k, v in self.ssim_vals_R2star.items():
            ssim_vals_R2star[k].update(v)
        for k, v in self.psnr_vals_R2star.items():
            psnr_vals_R2star[k].update(v)

        for k, v in self.mse_vals_S0.items():
            mse_vals_S0[k].update(v)
        for k, v in self.nmse_vals_S0.items():
            nmse_vals_S0[k].update(v)
        for k, v in self.ssim_vals_S0.items():
            ssim_vals_S0[k].update(v)
        for k, v in self.psnr_vals_S0.items():
            psnr_vals_S0[k].update(v)

        for k, v in self.mse_vals_B0.items():
            mse_vals_B0[k].update(v)
        for k, v in self.nmse_vals_B0.items():
            nmse_vals_B0[k].update(v)
        for k, v in self.ssim_vals_B0.items():
            ssim_vals_B0[k].update(v)
        for k, v in self.psnr_vals_B0.items():
            psnr_vals_B0[k].update(v)

        for k, v in self.mse_vals_phi.items():
            mse_vals_phi[k].update(v)
        for k, v in self.nmse_vals_phi.items():
            nmse_vals_phi[k].update(v)
        for k, v in self.ssim_vals_phi.items():
            ssim_vals_phi[k].update(v)
        for k, v in self.psnr_vals_phi.items():
            psnr_vals_phi[k].update(v)

        # apply means across image volumes
        metrics = {
            "MSE": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "NMSE": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "SSIM": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
            "PSNR": {
                "reconstruction": 0,
                "R2star": 0,
                "S0": 0,
                "B0": 0,
                "phi": 0,
            },
        }
        local_examples = 0
        for fname in mse_vals_R2star:
            local_examples += 1
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

            metrics["MSE"]["R2star"] = metrics["MSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_R2star[fname].items()])
            )
            metrics["NMSE"]["R2star"] = metrics["NMSE"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_R2star[fname].items()])
            )
            metrics["SSIM"]["R2star"] = metrics["SSIM"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_R2star[fname].items()])
            )
            metrics["PSNR"]["R2star"] = metrics["PSNR"]["R2star"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_R2star[fname].items()])
            )

            metrics["MSE"]["S0"] = metrics["MSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_S0[fname].items()])
            )
            metrics["NMSE"]["S0"] = metrics["NMSE"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_S0[fname].items()])
            )
            metrics["SSIM"]["S0"] = metrics["SSIM"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_S0[fname].items()])
            )
            metrics["PSNR"]["S0"] = metrics["PSNR"]["S0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_S0[fname].items()])
            )

            metrics["MSE"]["B0"] = metrics["MSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_B0[fname].items()])
            )
            metrics["NMSE"]["B0"] = metrics["NMSE"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_B0[fname].items()])
            )
            metrics["SSIM"]["B0"] = metrics["SSIM"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_B0[fname].items()])
            )
            metrics["PSNR"]["B0"] = metrics["PSNR"]["B0"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_B0[fname].items()])
            )

            metrics["MSE"]["phi"] = metrics["MSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in mse_vals_phi[fname].items()])
            )
            metrics["NMSE"]["phi"] = metrics["NMSE"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in nmse_vals_phi[fname].items()])
            )
            metrics["SSIM"]["phi"] = metrics["SSIM"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in ssim_vals_phi[fname].items()])
            )
            metrics["PSNR"]["phi"] = metrics["PSNR"]["phi"] + torch.mean(
                torch.cat([v.view(-1).float() for _, v in psnr_vals_phi[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"]["reconstruction"] = self.MSE(metrics["MSE"]["reconstruction"])
        metrics["NMSE"]["reconstruction"] = self.NMSE(metrics["NMSE"]["reconstruction"])
        metrics["SSIM"]["reconstruction"] = self.SSIM(metrics["SSIM"]["reconstruction"])
        metrics["PSNR"]["reconstruction"] = self.PSNR(metrics["PSNR"]["reconstruction"])

        metrics["MSE"]["R2star"] = self.MSE(metrics["MSE"]["R2star"])
        metrics["NMSE"]["R2star"] = self.NMSE(metrics["NMSE"]["R2star"])
        metrics["SSIM"]["R2star"] = self.SSIM(metrics["SSIM"]["R2star"])
        metrics["PSNR"]["R2star"] = self.PSNR(metrics["PSNR"]["R2star"])

        metrics["MSE"]["S0"] = self.MSE(metrics["MSE"]["S0"])
        metrics["NMSE"]["S0"] = self.NMSE(metrics["NMSE"]["S0"])
        metrics["SSIM"]["S0"] = self.SSIM(metrics["SSIM"]["S0"])
        metrics["PSNR"]["S0"] = self.PSNR(metrics["PSNR"]["S0"])

        metrics["MSE"]["B0"] = self.MSE(metrics["MSE"]["B0"])
        metrics["NMSE"]["B0"] = self.NMSE(metrics["NMSE"]["B0"])
        metrics["SSIM"]["B0"] = self.SSIM(metrics["SSIM"]["B0"])
        metrics["PSNR"]["B0"] = self.PSNR(metrics["PSNR"]["B0"])

        metrics["MSE"]["phi"] = self.MSE(metrics["MSE"]["phi"])
        metrics["NMSE"]["phi"] = self.NMSE(metrics["NMSE"]["phi"])
        metrics["SSIM"]["phi"] = self.SSIM(metrics["SSIM"]["phi"])
        metrics["PSNR"]["phi"] = self.PSNR(metrics["PSNR"]["phi"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_R2star", value["R2star"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_S0", value["S0"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_B0", value["B0"] / tot_examples, sync_dist=True)
            self.log(f"{metric}_phi", value["phi"] / tot_examples, sync_dist=True)

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

        dataset = qmri_loader.qMRIDataset(
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
            transform=quantitative_transforms.qMRIDataTransforms(  # type: ignore
                TEs=cfg.get("TEs"),
                precompute_quantitative_maps=cfg.get("precompute_quantitative_maps"),
                qmaps_scaling_factor=cfg.get("qmaps_scaling_factor"),
                shift_B0_input=cfg.get("shift_B0_input"),
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
            sequence=cfg.get("sequence", None),
            fixed_precomputed_acceleration=cfg.get("fixed_precomputed_acceleration", None),
            kspace_scaling_factor=cfg.get("kspace_scaling_factor", 1000),
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


class SignalForwardModel:
    """
    Defines a signal forward model based on sequence.

    Parameters
    ----------
    sequence : str
        Sequence name.
    """

    def __init__(self, sequence: Union[str, None] = None):
        super().__init__()
        self.sequence = sequence.lower() if isinstance(sequence, str) else None
        self.scaling = 1e-3

    def __call__(  # noqa: W0221
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """
        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map : torch.Tensor
            phi map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        if TEs is None:
            TEs = torch.Tensor([3.0, 11.5, 20.0, 28.5])
        if self.sequence == "megre":
            return self.MEGRESignalModel(R2star_map, S0_map, B0_map, phi_map, TEs)
        if self.sequence == "megre_no_phase":
            return self.MEGRENoPhaseSignalModel(R2star_map, S0_map, TEs)
        raise ValueError(
            "Only MEGRE and MEGRE no phase are supported are signal forward model at the moment. "
            f"Found {self.sequence}"
        )

    def MEGRESignalModel(  # noqa: W0221
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """
        MEGRE forward model.

        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        B0_map : torch.Tensor
            B0 map of shape [batch_size, n_x, n_y].
        phi_map : torch.Tensor
            phi map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        S0_map_real = S0_map
        S0_map_imag = phi_map

        def first_term(i):
            return torch.exp(-TEs[i] * self.scaling * R2star_map)

        def second_term(i):
            return torch.cos(B0_map * self.scaling * -TEs[i])

        def third_term(i):
            return torch.sin(B0_map * self.scaling * -TEs[i])

        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map_real * first_term(i) * second_term(i) - S0_map_imag * first_term(i) * third_term(i),
                        S0_map_real * first_term(i) * third_term(i) + S0_map_imag * first_term(i) * second_term(i),
                    ),
                    -1,
                )
                for i in range(len(TEs))  # type: ignore
            ],
            1,
        )
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])

    def MEGRENoPhaseSignalModel(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        TEs: Optional[List] = None,
    ):
        """
        MEGRE no phase forward model.

        Parameters
        ----------
        R2star_map : torch.Tensor
            R2* map of shape [batch_size, n_x, n_y].
        S0_map : torch.Tensor
            S0 map of shape [batch_size, n_x, n_y].
        TEs : list of float, optional
            List of echo times.
        """
        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),  # type: ignore
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),  # type: ignore
                    ),
                    -1,
                )
                for i in range(len(TEs))  # type: ignore
            ],
            1,
        )
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])
