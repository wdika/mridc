# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

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

from torch.nn import L1Loss, MSELoss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.utils import is_none
from mridc.collections.quantitative.data.qmri_data import qMRISliceDataset
from mridc.collections.quantitative.parts.transforms import qMRIDataTransforms
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.metrics.evaluate import mse, nmse, psnr, ssim
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, DistributedMetricSum
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["BaseqMRIReconstructionModel"]


class BaseqMRIReconstructionModel(BaseMRIReconstructionModel, ABC):
    """Base class of all quantitative MRIReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.acc = 1

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        if cfg_dict.get("loss_fn") == "ssim":
            self.train_loss_fn = SSIMLoss()
            self.eval_loss_fn = SSIMLoss()
        elif cfg_dict.get("loss_fn") == "mse":
            self.train_loss_fn = MSELoss(reduction="none")
            self.eval_loss_fn = MSELoss(reduction="none")
        elif cfg_dict.get("loss_fn") == "l1":
            self.train_loss_fn = L1Loss(reduction="none")
            self.eval_loss_fn = L1Loss(reduction="none")

        loss_regularization_factors = cfg_dict.get("loss_regularization_factors")
        self.loss_regularization_factors = {
            "R2star": loss_regularization_factors[0]["R2star"],
            "S0": loss_regularization_factors[1]["S0"],
            "B0": loss_regularization_factors[2]["B0"],
            "phi": loss_regularization_factors[3]["phi"],
        }

        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

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

    def process_quantitative_loss(self, target, pred, mask_brain, map, _loss_fn):
        """
        Processes the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        mask_brain: Mask for brain.
            torch.Tensor, shape [batch_size, n_x, n_y, 1]
        map: Type of map to regularize the loss.
            str in {"R2star", "S0", "B0", "phi"}
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        if "ssim" in str(_loss_fn).lower():

            def loss_fn(x, y, m):
                """Calculate the ssim loss."""
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                max_value = torch.max(torch.abs(y)) - torch.min(torch.abs(y)).unsqueeze(dim=0)
                m = torch.abs(m).to(x)

                loss = _loss_fn(x * m, y * m, data_range=max_value) * self.loss_regularization_factors[map]
                return loss

        else:

            def loss_fn(x, y, m):
                """Calculate other loss."""
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                m = torch.abs(m).to(x)

                if "mse" in str(_loss_fn).lower():
                    x = x.float()
                    y = y.float()
                    m = m.float()
                return _loss_fn(x * m, y * m) / self.loss_regularization_factors[map]

        return loss_fn(target, pred, mask_brain)

    def process_reconstruction_loss(self, target, pred, _loss_fn):
        """
        Processes the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()
        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        target = torch.abs(target / torch.max(torch.abs(target)))
        pred = torch.abs(pred / torch.max(torch.abs(pred)))

        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))

                return _loss_fn(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(x, y)

        return loss_fn(target, pred)

    @staticmethod
    def process_inputs(
        R2star_map_init,
        S0_map_init,
        B0_map_init,
        phi_map_init,
        y,
        mask,
    ):
        """
        Processes the inputs to the method.

        Parameters
        ----------
        R2star_map_init: R2* map.
            list of torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map_init: S0 map.
            list of torch.Tensor, shape [batch_size, n_x, n_y]
        B0_map_init: B0 map.
            list of torch.Tensor, shape [batch_size, n_x, n_y]
        phi_map_init: Phi map.
            list of torch.Tensor, shape [batch_size, n_x, n_y]
        y: Subsampled k-space data.
            list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            list of torch.Tensor, shape [batch_size, 1, n_x, n_y, 1]

        Returns
        -------
        R2star_map_init: Initial R2* map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map_init: Initial S0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        B0_map_init: Initial B0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        phi_map_init: Initial phi map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        mask_brain: Brain mask to regularize the inputs.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 1]
        mask: Sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 1]
        r: Random index.
            int
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
        return (
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            y,
            mask,
            r,
        )

    @staticmethod
    def _check_if_isinstance_pred(x):
        """
        Checks if x is a list of predictions.
        """
        # Cascades
        if isinstance(x, list):
            x = x[-1]
        # Time-steps
        if isinstance(x, list):
            x = x[-1]
        return x

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'R2star_map_init': R2* initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'R2star_map_target': R2* target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_init': S0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_target': S0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_init': B0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_target': B0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_init': Phi initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_target': Phi target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'TEs': Echo times.
                list of float
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Brain & sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'mask_brain': Brain mask.
                torch.Tensor, shape [n_x, n_y]
            'init_pred': Initial prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target': Target data.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
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
            eta,
            target,
            fname,
            slice_num,
            acc,
        ) = batch

        (R2star_map_init, S0_map_init, B0_map_init, phi_map_init, y, sampling_mask, r,) = self.process_inputs(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            y,
            mask,
        )

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

        if self.accumulate_estimates:
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
                lossrecon = sum(self.process_reconstruction_loss(target, recon_pred, self.train_loss_fn))
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
                lossrecon = self.process_reconstruction_loss(target, recon_pred, self.train_loss_fn).mean()
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

        self.acc = r if r != 0 else acc
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

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """
        Performs a validation step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'R2star_map_init': R2* initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'R2star_map_target': R2* target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_init': S0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_target': S0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_init': B0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_target': B0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_init': Phi initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_target': Phi target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'TEs': Echo times.
                list of float
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Brain & sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'mask_brain': Brain mask.
                torch.Tensor, shape [n_x, n_y]
            'init_pred': Initial prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target': Target data.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
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
            eta,
            target,
            fname,
            slice_num,
            acc,
        ) = batch

        (R2star_map_init, S0_map_init, B0_map_init, phi_map_init, y, sampling_mask, r,) = self.process_inputs(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            y,
            mask,
        )

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

        if self.accumulate_estimates:
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
                lossrecon = sum(self.process_reconstruction_loss(target, recon_pred, self.eval_loss_fn))
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = sum(
                self.process_quantitative_loss(
                    R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.eval_loss_fn
                )
            )
            lossS0 = sum(
                self.process_quantitative_loss(S0_map_target, S0_map_pred, mask_brain, "S0", self.eval_loss_fn)
            )
            lossB0 = sum(
                self.process_quantitative_loss(B0_map_target, B0_map_pred, mask_brain, "B0", self.eval_loss_fn)
            )
            lossPhi = sum(
                self.process_quantitative_loss(phi_map_target, phi_map_pred, mask_brain, "phi", self.eval_loss_fn)
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
                lossrecon = self.process_reconstruction_loss(target, recon_pred, self.eval_loss_fn).mean()
            else:
                lossrecon = torch.tensor([0.0])

            lossR2star = self.process_quantitative_loss(
                R2star_map_target, R2star_map_pred, mask_brain, "R2star", self.eval_loss_fn
            ).sum()
            lossS0 = self.process_quantitative_loss(
                S0_map_target, S0_map_pred, mask_brain, "S0", self.eval_loss_fn
            ).sum()
            lossB0 = self.process_quantitative_loss(
                B0_map_target, B0_map_pred, mask_brain, "B0", self.eval_loss_fn
            ).sum()
            lossPhi = self.process_quantitative_loss(
                phi_map_target, phi_map_pred, mask_brain, "phi", self.eval_loss_fn
            ).sum()

        val_loss = sum([lossR2star, lossS0, lossB0, lossPhi]) / 4
        val_loss = val_loss.mean() / 2

        if self.use_reconstruction_module:
            val_loss = val_loss + lossrecon

            if isinstance(recon_pred, list):
                recon_pred = torch.stack([self._check_if_isinstance_pred(x) for x in recon_pred], dim=1)

            recon_pred = torch.stack(
                [
                    self._check_if_isinstance_pred(recon_pred[:, echo_time, :, :])
                    for echo_time in range(recon_pred.shape[1])
                ],
                1,
            )

            recon_pred = recon_pred.detach().cpu()
            recon_pred = torch.abs(recon_pred / torch.max(torch.abs(recon_pred)))
            target = target.detach().cpu()  # type: ignore
            target = torch.abs(target / torch.max(torch.abs(target)))

        R2star_map_pred = self._check_if_isinstance_pred(R2star_map_pred)
        S0_map_pred = self._check_if_isinstance_pred(S0_map_pred)
        B0_map_pred = self._check_if_isinstance_pred(B0_map_pred)
        phi_map_pred = self._check_if_isinstance_pred(phi_map_pred)

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

        if self.use_reconstruction_module:
            for echo_time in range(target.shape[1]):  # type: ignore
                self.log_image(
                    f"{key}/reconstruction_echo_{echo_time}/target", target[:, echo_time, :, :]  # type: ignore
                )  # type: ignore
                self.log_image(f"{key}/reconstruction_echo_{echo_time}/reconstruction", recon_pred[:, echo_time, :, :])
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
                    torch.tensor(mse(target[:, echo_time, ...], recon_pred[:, echo_time, ...])).view(1)  # type: ignore
                )  # type: ignore
                nmses.append(
                    torch.tensor(nmse(target[:, echo_time, ...], recon_pred[:, echo_time, ...])).view(1)  # type: ignore
                )  # type: ignore
                ssims.append(
                    torch.tensor(
                        ssim(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )
                psnrs.append(
                    torch.tensor(
                        psnr(
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

        self.mse_vals_R2star[fname][slice_num] = torch.tensor(mse(R2star_map_target, R2star_map_output)).view(1)
        self.nmse_vals_R2star[fname][slice_num] = torch.tensor(nmse(R2star_map_target, R2star_map_output)).view(1)
        self.ssim_vals_R2star[fname][slice_num] = torch.tensor(
            ssim(R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min())
        ).view(1)
        self.psnr_vals_R2star[fname][slice_num] = torch.tensor(
            psnr(R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min())
        ).view(1)

        self.mse_vals_S0[fname][slice_num] = torch.tensor(mse(S0_map_target, S0_map_output)).view(1)
        self.nmse_vals_S0[fname][slice_num] = torch.tensor(nmse(S0_map_target, S0_map_output)).view(1)
        self.ssim_vals_S0[fname][slice_num] = torch.tensor(
            ssim(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)
        self.psnr_vals_S0[fname][slice_num] = torch.tensor(
            psnr(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)

        self.mse_vals_B0[fname][slice_num] = torch.tensor(mse(B0_map_target, B0_map_output)).view(1)
        self.nmse_vals_B0[fname][slice_num] = torch.tensor(nmse(B0_map_target, B0_map_output)).view(1)
        self.ssim_vals_B0[fname][slice_num] = torch.tensor(
            ssim(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)
        self.psnr_vals_B0[fname][slice_num] = torch.tensor(
            psnr(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)

        self.mse_vals_phi[fname][slice_num] = torch.tensor(mse(phi_map_target, phi_map_output)).view(1)
        self.nmse_vals_phi[fname][slice_num] = torch.tensor(nmse(phi_map_target, phi_map_output)).view(1)
        self.ssim_vals_phi[fname][slice_num] = torch.tensor(
            ssim(phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min())
        ).view(1)
        self.psnr_vals_phi[fname][slice_num] = torch.tensor(
            psnr(phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min())
        ).view(1)

        return {"val_loss": val_loss}

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch: Batch of data. List for multiple acceleration factors. Dict[str, torch.Tensor], with keys,
            'R2star_map_init': R2* initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'R2star_map_target': R2* target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_init': S0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'S0_map_target': S0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_init': B0 initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'B0_map_target': B0 target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_init': Phi initial map.
                list of torch.Tensor, shape [batch_size,  n_x, n_y]
            'phi_map_target': Phi target map.
                torch.Tensor, shape [batch_size,  n_x, n_y]
            'TEs': Echo times.
                list of float
            'kspace': k-space data.
                torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'y': Subsampled k-space data.
                list of torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
            'sensitivity_maps': Coils sensitivity maps.
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': Sampling mask.
                list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'mask_brain': Brain mask.
                torch.Tensor, shape [n_x, n_y]
            'init_pred': Initial prediction.
                torch.Tensor, shape [batch_size, 1, n_x, n_y, 2] or None
            'target': Target data.
                torch.Tensor, shape [batch_size, n_x, n_y] or None
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
        name: Name of the volume.
            str
        slice_num: Slice number.
            int
        pred: Predicted data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
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
            eta,
            target,
            fname,
            slice_num,
            acc,
        ) = batch

        (R2star_map_init, S0_map_init, B0_map_init, phi_map_init, y, sampling_mask, r,) = self.process_inputs(
            R2star_map_init,
            S0_map_init,
            B0_map_init,
            phi_map_init,
            y,
            mask,
        )

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

        if self.accumulate_estimates:
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
                recon_pred = torch.stack([self._check_if_isinstance_pred(x) for x in recon_pred], dim=1)

            recon_pred = torch.stack(
                [
                    self._check_if_isinstance_pred(recon_pred[:, echo_time, :, :])
                    for echo_time in range(recon_pred.shape[1])
                ],
                1,
            )

            recon_pred = recon_pred.detach().cpu()
            recon_pred = torch.abs(recon_pred / torch.max(torch.abs(recon_pred)))
            target = target.detach().cpu()  # type: ignore
            target = torch.abs(target / torch.max(torch.abs(target)))

        R2star_map_pred = self._check_if_isinstance_pred(R2star_map_pred)
        S0_map_pred = self._check_if_isinstance_pred(S0_map_pred)
        B0_map_pred = self._check_if_isinstance_pred(B0_map_pred)
        phi_map_pred = self._check_if_isinstance_pred(phi_map_pred)

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

        if self.use_reconstruction_module:
            for echo_time in range(target.shape[1]):  # type: ignore
                self.log_image(
                    f"{key}/reconstruction_echo_{echo_time}/target", target[:, echo_time, :, :]  # type: ignore
                )  # type: ignore
                self.log_image(f"{key}/reconstruction_echo_{echo_time}/reconstruction", recon_pred[:, echo_time, :, :])
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
                    torch.tensor(mse(target[:, echo_time, ...], recon_pred[:, echo_time, ...])).view(1)  # type: ignore
                )  # type: ignore
                nmses.append(
                    torch.tensor(nmse(target[:, echo_time, ...], recon_pred[:, echo_time, ...])).view(1)  # type: ignore
                )  # type: ignore
                ssims.append(
                    torch.tensor(
                        ssim(
                            target[:, echo_time, ...],  # type: ignore
                            recon_pred[:, echo_time, ...],
                            maxval=recon_pred[:, echo_time, ...].max() - recon_pred[:, echo_time, ...].min(),
                        )
                    ).view(1)
                )
                psnrs.append(
                    torch.tensor(
                        psnr(
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

        self.mse_vals_R2star[fname][slice_num] = torch.tensor(mse(R2star_map_target, R2star_map_output)).view(1)
        self.nmse_vals_R2star[fname][slice_num] = torch.tensor(nmse(R2star_map_target, R2star_map_output)).view(1)
        self.ssim_vals_R2star[fname][slice_num] = torch.tensor(
            ssim(R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min())
        ).view(1)
        self.psnr_vals_R2star[fname][slice_num] = torch.tensor(
            psnr(R2star_map_target, R2star_map_output, maxval=R2star_map_output.max() - R2star_map_output.min())
        ).view(1)

        self.mse_vals_S0[fname][slice_num] = torch.tensor(mse(S0_map_target, S0_map_output)).view(1)
        self.nmse_vals_S0[fname][slice_num] = torch.tensor(nmse(S0_map_target, S0_map_output)).view(1)
        self.ssim_vals_S0[fname][slice_num] = torch.tensor(
            ssim(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)
        self.psnr_vals_S0[fname][slice_num] = torch.tensor(
            psnr(S0_map_target, S0_map_output, maxval=S0_map_output.max() - S0_map_output.min())
        ).view(1)

        self.mse_vals_B0[fname][slice_num] = torch.tensor(mse(B0_map_target, B0_map_output)).view(1)
        self.nmse_vals_B0[fname][slice_num] = torch.tensor(nmse(B0_map_target, B0_map_output)).view(1)
        self.ssim_vals_B0[fname][slice_num] = torch.tensor(
            ssim(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)
        self.psnr_vals_B0[fname][slice_num] = torch.tensor(
            psnr(B0_map_target, B0_map_output, maxval=B0_map_output.max() - B0_map_output.min())
        ).view(1)

        self.mse_vals_phi[fname][slice_num] = torch.tensor(mse(phi_map_target, phi_map_output)).view(1)
        self.nmse_vals_phi[fname][slice_num] = torch.tensor(nmse(phi_map_target, phi_map_output)).view(1)
        self.ssim_vals_phi[fname][slice_num] = torch.tensor(
            ssim(phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min())
        ).view(1)
        self.psnr_vals_phi[fname][slice_num] = torch.tensor(
            psnr(phi_map_target, phi_map_output, maxval=phi_map_output.max() - phi_map_output.min())
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
        outputs: List of outputs from the train step.
            list of dicts
        """
        self.log("train_loss", torch.stack([x["train_loss"] for x in outputs]).mean())
        self.log(f"train_loss_{self.acc}x", torch.stack([x[f"train_loss_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_R2star_{self.acc}x", torch.stack([x[f"loss_R2star_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_S0_{self.acc}x", torch.stack([x[f"loss_S0_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_B0_{self.acc}x", torch.stack([x[f"loss_B0_{self.acc}x"] for x in outputs]).mean())
        self.log(f"loss_phi_{self.acc}x", torch.stack([x[f"loss_phi_{self.acc}x"] for x in outputs]).mean())

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

        for k in self.mse_vals_reconstruction.keys():
            mse_vals_reconstruction[k].update(self.mse_vals_reconstruction[k])
        for k in self.nmse_vals_reconstruction.keys():
            nmse_vals_reconstruction[k].update(self.nmse_vals_reconstruction[k])
        for k in self.ssim_vals_reconstruction.keys():
            ssim_vals_reconstruction[k].update(self.ssim_vals_reconstruction[k])
        for k in self.psnr_vals_R2star.keys():
            psnr_vals_reconstruction[k].update(self.psnr_vals_reconstruction[k])

        for k in self.mse_vals_R2star.keys():
            mse_vals_R2star[k].update(self.mse_vals_R2star[k])
        for k in self.nmse_vals_R2star.keys():
            nmse_vals_R2star[k].update(self.nmse_vals_R2star[k])
        for k in self.ssim_vals_R2star.keys():
            ssim_vals_R2star[k].update(self.ssim_vals_R2star[k])
        for k in self.psnr_vals_R2star.keys():
            psnr_vals_R2star[k].update(self.psnr_vals_R2star[k])

        for k in self.mse_vals_S0.keys():
            mse_vals_S0[k].update(self.mse_vals_S0[k])
        for k in self.nmse_vals_S0.keys():
            nmse_vals_S0[k].update(self.nmse_vals_S0[k])
        for k in self.ssim_vals_S0.keys():
            ssim_vals_S0[k].update(self.ssim_vals_S0[k])
        for k in self.psnr_vals_S0.keys():
            psnr_vals_S0[k].update(self.psnr_vals_S0[k])

        for k in self.mse_vals_B0.keys():
            mse_vals_B0[k].update(self.mse_vals_B0[k])
        for k in self.nmse_vals_B0.keys():
            nmse_vals_B0[k].update(self.nmse_vals_B0[k])
        for k in self.ssim_vals_B0.keys():
            ssim_vals_B0[k].update(self.ssim_vals_B0[k])
        for k in self.psnr_vals_B0.keys():
            psnr_vals_B0[k].update(self.psnr_vals_B0[k])

        for k in self.mse_vals_phi.keys():
            mse_vals_phi[k].update(self.mse_vals_phi[k])
        for k in self.nmse_vals_phi.keys():
            nmse_vals_phi[k].update(self.nmse_vals_phi[k])
        for k in self.ssim_vals_phi.keys():
            ssim_vals_phi[k].update(self.ssim_vals_phi[k])
        for k in self.psnr_vals_phi.keys():
            psnr_vals_phi[k].update(self.psnr_vals_phi[k])

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
            self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples)
            self.log(f"{metric}_R2star", value["R2star"] / tot_examples)
            self.log(f"{metric}_S0", value["S0"] / tot_examples)
            self.log(f"{metric}_B0", value["B0"] / tot_examples)
            self.log(f"{metric}_phi", value["phi"] / tot_examples)

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

        for k in self.mse_vals_reconstruction.keys():
            mse_vals_reconstruction[k].update(self.mse_vals_reconstruction[k])
        for k in self.nmse_vals_reconstruction.keys():
            nmse_vals_reconstruction[k].update(self.nmse_vals_reconstruction[k])
        for k in self.ssim_vals_reconstruction.keys():
            ssim_vals_reconstruction[k].update(self.ssim_vals_reconstruction[k])
        for k in self.psnr_vals_R2star.keys():
            psnr_vals_reconstruction[k].update(self.psnr_vals_reconstruction[k])

        for k in self.mse_vals_R2star.keys():
            mse_vals_R2star[k].update(self.mse_vals_R2star[k])
        for k in self.nmse_vals_R2star.keys():
            nmse_vals_R2star[k].update(self.nmse_vals_R2star[k])
        for k in self.ssim_vals_R2star.keys():
            ssim_vals_R2star[k].update(self.ssim_vals_R2star[k])
        for k in self.psnr_vals_R2star.keys():
            psnr_vals_R2star[k].update(self.psnr_vals_R2star[k])

        for k in self.mse_vals_S0.keys():
            mse_vals_S0[k].update(self.mse_vals_S0[k])
        for k in self.nmse_vals_S0.keys():
            nmse_vals_S0[k].update(self.nmse_vals_S0[k])
        for k in self.ssim_vals_S0.keys():
            ssim_vals_S0[k].update(self.ssim_vals_S0[k])
        for k in self.psnr_vals_S0.keys():
            psnr_vals_S0[k].update(self.psnr_vals_S0[k])

        for k in self.mse_vals_B0.keys():
            mse_vals_B0[k].update(self.mse_vals_B0[k])
        for k in self.nmse_vals_B0.keys():
            nmse_vals_B0[k].update(self.nmse_vals_B0[k])
        for k in self.ssim_vals_B0.keys():
            ssim_vals_B0[k].update(self.ssim_vals_B0[k])
        for k in self.psnr_vals_B0.keys():
            psnr_vals_B0[k].update(self.psnr_vals_B0[k])

        for k in self.mse_vals_phi.keys():
            mse_vals_phi[k].update(self.mse_vals_phi[k])
        for k in self.nmse_vals_phi.keys():
            nmse_vals_phi[k].update(self.nmse_vals_phi[k])
        for k in self.ssim_vals_phi.keys():
            ssim_vals_phi[k].update(self.ssim_vals_phi[k])
        for k in self.psnr_vals_phi.keys():
            psnr_vals_phi[k].update(self.psnr_vals_phi[k])

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
            self.log(f"{metric}_Reconstruction", value["reconstruction"] / tot_examples)
            self.log(f"{metric}_R2star", value["R2star"] / tot_examples)
            self.log(f"{metric}_S0", value["S0"] / tot_examples)
            self.log(f"{metric}_B0", value["B0"] / tot_examples)
            self.log(f"{metric}_phi", value["phi"] / tot_examples)

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

        dataset = qMRISliceDataset(
            root=cfg.get("data_path"),
            sense_root=cfg.get("sense_path"),
            mask_root=cfg.get("mask_path"),
            sequence=cfg.get("sequence"),
            transform=qMRIDataTransforms(
                TEs=cfg.get("TEs"),
                precompute_quantitative_maps=cfg.get("precompute_quantitative_maps"),
                coil_combination_method=cfg.get("coil_combination_method"),
                dimensionality=cfg.get("dimensionality"),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask"),
                normalize_inputs=cfg.get("normalize_inputs"),
                crop_size=cfg.get("crop_size"),
                crop_before_masking=cfg.get("crop_before_masking"),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size"),
                fft_centered=cfg.get("fft_centered"),
                fft_normalization=cfg.get("fft_normalization"),
                max_norm=cfg.get("max_norm"),
                spatial_dims=cfg.get("spatial_dims"),
                coil_dim=cfg.get("coil_dim"),
                shift_B0_input=cfg.get("shift_B0_input"),
                use_seed=cfg.get("use_seed"),
            ),
            sample_rate=cfg.get("sample_rate"),
            consecutive_slices=cfg.get("consecutive_slices"),
            data_saved_per_slice=cfg.get("data_saved_per_slice"),
            init_coil_dim=cfg.get("init_coil_dim"),
            fixed_precomputed_acceleration=cfg.get("fixed_precomputed_acceleration"),
            kspace_scaling_factor=cfg.get("kspace_scaling_factor"),
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
