# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/wustl-cig/DeCoLearn/blob/main/decolearn/method/DeCoLearn.py

from abc import ABC
from collections import defaultdict
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.motioncompensation.losses.losses import GradientLoss, NormalizedCrossCorrelationLoss
from mridc.collections.motioncompensation.modules.voxelmorph import VoxelMorph
from mridc.collections.reconstruction.metrics.evaluate import mse, nmse, psnr, ssim
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel

__all__ = ["DeCoLearn"]


class DeCoLearn(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the DeCoLearn, as presented in Gan, W. et al.

    References
    ----------

    ..

        Gan, W., Sun, Y., Eldeniz, C., Liu, J., An, H., & Kamilov, U. S. (2022). Deformation-Compensated Learning for
        Image Reconstruction without Ground Truth. IEEE Transactions on Medical Imaging, 0062(c), 1–14.
        https://doi.org/10.1109/TMI.2022.3163018

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Cascades of RIM blocks
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        conv_nonlinearity = cfg_dict.get("moco_stn_conv_nonlinearity")
        self.moco_module = VoxelMorph(
            chans=cfg_dict.get("moco_stn_chans"),
            num_pools=cfg_dict.get("moco_stn_num_pools"),
            in_chans=cfg_dict.get("moco_stn_in_chans"),
            out_chans=cfg_dict.get("moco_stn_out_chans"),
            drop_prob=cfg_dict.get("moco_stn_drop_prob"),
            padding_size=cfg_dict.get("moco_stn_padding_size"),
            normalize=cfg_dict.get("moco_stn_normalize"),
            bidirectional=cfg_dict.get("moco_stn_bidirectional"),
            int_downsize=cfg_dict.get("moco_stn_int_downsize"),
            int_steps=cfg_dict.get("moco_stn_int_steps"),
            shape=cfg_dict.get("moco_stn_input_shape"),
            mode=cfg_dict.get("moco_stn_mode"),
        )
        self.moco_clip_value = cfg_dict.get("moco_clip_value")
        self.moco_stn_optimize_registration = cfg_dict.get("moco_stn_optimize_registration")
        self.moco_lambda_registration = cfg_dict.get("moco_lambda_registration")
        self.moco_lambda_registration_mse = cfg_dict.get("moco_lambda_registration_mse")
        self.moco_loss_consenus_coefficient = cfg_dict.get("moco_loss_consenus_coefficient")

        self.moco_reconstruction_loss_fn = nn.SmoothL1Loss()
        self.moco_similarity_loss_fn = NormalizedCrossCorrelationLoss(
            win_size=cfg_dict.get("moco_similarity_loss_win_size")
        )
        self.moco_gradient_loss_fn = GradientLoss(
            penalty=cfg_dict.get("moco_gradient_loss_penalty"),
            reg=cfg_dict.get("moco_gradient_loss_reg"),
            dimensionality=cfg_dict.get("moco_gradient_loss_dimensionality"),
        )
        self.moco_mse_loss_fn = nn.MSELoss()

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch: Batch of data.
            Dict[str, torch.Tensor], with keys,

            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
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
        kspace, y, sensitivity_maps, mask, init_pred, _, phase_shift, fname, slice_num, acc = batch
        y, mask, _, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        x_r = self.forward(
            y[:, 0],  # type: ignore
            sensitivity_maps,
            mask[:, 0],  # type: ignore
            init_pred=None,
            target=torch.view_as_complex(y[:, 0]).sum(1),  # type: ignore
            phase_shift=phase_shift[:, 0],  # type: ignore
        )
        x_m = self.forward(
            y[:, 1],  # type: ignore
            sensitivity_maps,
            mask[:, 1],  # type: ignore
            init_pred=None,
            target=torch.view_as_complex(y[:, 1]).sum(1),  # type: ignore
            phase_shift=phase_shift[:, 1],  # type: ignore
        )

        if self.accumulate_estimates:
            try:
                x_r = next(x_r)
                x_m = next(x_m)
            except StopIteration:
                pass

        # Cascades
        if isinstance(x_r, list):
            x_r = x_r[-1]
            x_m = x_m[-1]

        # Time-steps
        if isinstance(x_r, list):
            x_r = x_r[-1]
            x_m = x_m[-1]

        # registration module
        if self.moco_stn_optimize_registration:
            x_m = x_m.detach()
            x_r = x_r.detach()

            x_m = x_m / torch.max(torch.abs(x_m))
            x_r = x_r / torch.max(torch.abs(x_r))

            moco_x_m2x_r, _, flow_x_m2x_r = self.moco_module.forward(x_m, x_r)
            moco_x_m2x_r = torch.sqrt(torch.sum(torch.abs(moco_x_m2x_r) ** 2, dim=1, keepdim=True))

            moco_x_m2x_r = moco_x_m2x_r / torch.max(torch.abs(moco_x_m2x_r))
            flow_x_m2x_r = flow_x_m2x_r / torch.max(torch.abs(flow_x_m2x_r))

            similarity_loss_x_m2x_r = self.moco_similarity_loss_fn(
                torch.abs(moco_x_m2x_r),
                torch.abs(x_m).unsqueeze(1),
            )
            gradient_loss_x_m2x_r = self.moco_gradient_loss_fn(torch.abs(flow_x_m2x_r))
            mse_loss_x_m2x_r = self.moco_mse_loss_fn(
                torch.abs(moco_x_m2x_r),
                torch.abs(x_m).unsqueeze(1),
            )

            moco_x_r2x_m, _, flow_x_r2x_m = self.moco_module.forward(x_r, x_m)
            moco_x_r2x_m = torch.sqrt(torch.sum(torch.abs(moco_x_r2x_m) ** 2, dim=1, keepdim=True))

            moco_x_r2x_m = moco_x_r2x_m / torch.max(torch.abs(moco_x_r2x_m))
            flow_x_r2x_m = flow_x_r2x_m / torch.max(torch.abs(flow_x_r2x_m))

            similarity_loss_x_r2x_m = self.moco_similarity_loss_fn(
                torch.abs(moco_x_r2x_m), torch.abs(x_r).unsqueeze(1)
            )
            gradient_loss_x_r2x_m = self.moco_gradient_loss_fn(torch.abs(flow_x_r2x_m))
            mse_loss_x_r2x_m = self.moco_mse_loss_fn(torch.abs(moco_x_r2x_m), torch.abs(x_r).unsqueeze(1))

            train_loss = similarity_loss_x_m2x_r + similarity_loss_x_r2x_m
            if self.moco_lambda_registration > 0:
                train_loss += self.moco_lambda_registration * (gradient_loss_x_m2x_r + gradient_loss_x_r2x_m)
            if self.moco_lambda_registration_mse > 0:
                train_loss += self.moco_lambda_registration_mse * (mse_loss_x_m2x_r + mse_loss_x_r2x_m)

            if self.moco_clip_value > 0:
                torch.nn.utils.clip_grad_value_(self.moco_module.parameters(), self.moco_clip_value)

        tensorboard_logs = {
            f"train_loss_similarity_deformed2reference": similarity_loss_x_m2x_r,  # type: ignore
            f"train_loss_similarity_reference2deformed": similarity_loss_x_r2x_m,  # type: ignore
            f"train_loss_gradient_deformed2reference": gradient_loss_x_m2x_r,  # type: ignore
            f"train_loss_gradient_reference2deformed": gradient_loss_x_r2x_m,  # type: ignore
            f"train_loss_mse_deformed2reference": mse_loss_x_m2x_r,  # type: ignore
            f"train_loss_mse_reference2deformed": mse_loss_x_r2x_m,  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """
        Performs a validation step.

        Parameters
        ----------
        batch: Batch of data. Dict[str, torch.Tensor], with keys,
            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
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
        kspace, y, sensitivity_maps, mask, init_pred, _, phase_shift, fname, slice_num, acc = batch
        y, mask, _, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)

        x_r = self.forward(
            y[:, 0],  # type: ignore
            sensitivity_maps,
            mask[:, 0],  # type: ignore
            init_pred=None,
            target=torch.view_as_complex(y[:, 0]).sum(1),  # type: ignore
            phase_shift=phase_shift[:, 0],  # type: ignore
        )
        x_m = self.forward(
            y[:, 1],  # type: ignore
            sensitivity_maps,
            mask[:, 1],  # type: ignore
            init_pred=None,
            target=torch.view_as_complex(y[:, 1]).sum(1),  # type: ignore
            phase_shift=phase_shift[:, 1],  # type: ignore
        )

        if self.accumulate_estimates:
            try:
                x_r = next(x_r)
                x_m = next(x_m)
            except StopIteration:
                pass

        # Cascades
        if isinstance(x_r, list):
            x_r = x_r[-1]
            x_m = x_m[-1]

        # Time-steps
        if isinstance(x_r, list):
            x_r = x_r[-1]
            x_m = x_m[-1]

        x_r = x_r / torch.max(torch.abs(x_r))
        x_m = x_m / torch.max(torch.abs(x_m))

        # registration module
        if self.moco_stn_optimize_registration:
            _, _, flow_x_m2x_r = self.moco_module.forward(x_m, x_r)
            x_m = torch.view_as_real(x_m)
            x_m_real = self.moco_module.transformer(x_m[..., 0].unsqueeze(1), flow_x_m2x_r)  # type: ignore
            x_m_imag = self.moco_module.transformer(x_m[..., 1].unsqueeze(1), flow_x_m2x_r)  # type: ignore
            moco_x_m2x_r = torch.cat([x_m_real, x_m_imag], 1).permute(0, 2, 3, 1)
            moco_x_m2x_r = torch.view_as_real(moco_x_m2x_r[..., 0] + 1j * moco_x_m2x_r[..., 1])  # type: ignore
            moco_y_x_m2x_r = mask[:, 0, 0] * fft2(  # type: ignore
                moco_x_m2x_r, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            x_m = torch.view_as_complex(x_m)

            _, _, flow_x_r2x_m = self.moco_module.forward(x_r, x_m)
            x_r = torch.view_as_real(x_r)
            x_r_real = self.moco_module.transformer(x_r[..., 0].unsqueeze(1), flow_x_r2x_m)  # type: ignore
            x_r_imag = self.moco_module.transformer(x_r[..., 1].unsqueeze(1), flow_x_r2x_m)  # type: ignore
            moco_x_r2x_m = torch.cat([x_r_real, x_r_imag], 1).permute(0, 2, 3, 1)
            moco_x_r2x_m = torch.view_as_real(moco_x_r2x_m[..., 0] + 1j * moco_x_r2x_m[..., 1])  # type: ignore
            moco_y_x_r2x_m = mask[:, 1, 0] * fft2(  # type: ignore
                moco_x_r2x_m, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            x_r = torch.view_as_complex(x_r)
        else:
            moco_y_x_m2x_r = mask[:, 0, 0] * fft2(x_m, self.fft_centered, self.fft_normalization, self.spatial_dims)  # type: ignore
            moco_y_x_r2x_m = mask[:, 1, 0] * fft2(x_r, self.fft_centered, self.fft_normalization, self.spatial_dims)  # type: ignore

        moco_y_x_m2x_r = ifft2(moco_y_x_m2x_r, self.fft_centered, self.fft_normalization, self.spatial_dims)
        moco_y_x_m2x_r = moco_y_x_m2x_r / torch.max(torch.abs(moco_y_x_m2x_r))
        moco_y_x_m2x_r = fft2(moco_y_x_m2x_r, self.fft_centered, self.fft_normalization, self.spatial_dims)
        moco_y_x_r2x_m = ifft2(moco_y_x_r2x_m, self.fft_centered, self.fft_normalization, self.spatial_dims)
        moco_y_x_r2x_m = moco_y_x_r2x_m / torch.max(torch.abs(moco_y_x_r2x_m))
        moco_y_x_r2x_m = fft2(moco_y_x_r2x_m, self.fft_centered, self.fft_normalization, self.spatial_dims)
        y = ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
        y = y / torch.max(torch.abs(y))
        y = fft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)

        recon_loss_x_m2x_r = self.moco_reconstruction_loss_fn(
            torch.abs(torch.view_as_complex(moco_y_x_m2x_r)),
            torch.abs(torch.view_as_complex(y[:, 1]).sum(1)),  # type: ignore
        )
        recon_loss_x_r2x_m = self.moco_reconstruction_loss_fn(
            torch.abs(torch.view_as_complex(moco_y_x_r2x_m)),
            torch.abs(torch.view_as_complex(y[:, 1]).sum(1)),  # type: ignore
        )
        val_loss = recon_loss_x_m2x_r + recon_loss_x_r2x_m

        if self.moco_loss_consenus_coefficient > 0:
            recon_loss_consensus_r = self.moco_reconstruction_loss_fn(
                torch.abs(
                    torch.view_as_complex(
                        mask[:, 0, 0]  # type: ignore
                        * fft2(torch.view_as_real(x_r), self.fft_centered, self.fft_normalization, self.spatial_dims)
                    )
                ),
                torch.abs(torch.view_as_complex(y[:, 0]).sum(1)),  # type: ignore
            )
            recon_loss_consensus_m = self.moco_reconstruction_loss_fn(
                torch.abs(
                    torch.view_as_complex(
                        mask[:, 1, 0]  # type: ignore
                        * fft2(torch.view_as_real(x_m), self.fft_centered, self.fft_normalization, self.spatial_dims)
                    )
                ),
                torch.abs(torch.view_as_complex(y[:, 1]).sum(1)),  # type: ignore
            )
            val_loss += self.moco_loss_consenus_coefficient * (recon_loss_consensus_r + recon_loss_consensus_m)

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(x_m).detach().cpu()
        target = torch.abs(x_r).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/reference_reconstruction", target)
        self.log_image(f"{key}/deformed_reconstruction", output)
        self.log_image(f"{key}/_error", error)

        target = target.numpy()  # type: ignore
        output = output.numpy()  # type: ignore
        self.mse_vals[fname][slice_num] = torch.tensor(mse(target, output)).view(1)
        self.nmse_vals[fname][slice_num] = torch.tensor(nmse(target, output)).view(1)
        self.ssim_vals[fname][slice_num] = torch.tensor(ssim(target, output, maxval=output.max() - output.min())).view(
            1
        )
        self.psnr_vals[fname][slice_num] = torch.tensor(psnr(target, output, maxval=output.max() - output.min())).view(
            1
        )

        if self.moco_clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.moco_module.parameters(), self.moco_clip_value)

        tensorboard_logs = {
            f"recon_loss_deformed2reference": recon_loss_x_m2x_r,  # type: ignore
            f"recon_loss_reference2deformed": recon_loss_x_r2x_m,  # type: ignore
        }
        if self.moco_loss_consenus_coefficient > 0:
            tensorboard_logs[f"recon_loss_consensus_reference"] = recon_loss_consensus_r  # type: ignore
            tensorboard_logs[f"recon_loss_consensus_deformed"] = recon_loss_consensus_m  # type: ignore

        return {"val_loss": val_loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        """
        Called at the end of training epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the training batches.
            list of dicts
        """
        self.log("train_loss", torch.stack([x["loss"] for x in outputs]).mean())
        self.log(
            f"train_loss_similarity_deformed2referencex",
            torch.stack([x["log"]["train_loss_similarity_deformed2reference"] for x in outputs]).mean(),
        )
        self.log(
            f"train_loss_similarity_reference2deformedx",
            torch.stack([x["log"]["train_loss_similarity_reference2deformed"] for x in outputs]).mean(),
        )
        self.log(
            f"train_loss_gradient_deformed2referencex",
            torch.stack([x["log"]["train_loss_gradient_deformed2reference"] for x in outputs]).mean(),
        )
        self.log(
            f"train_loss_gradient_reference2deformedx",
            torch.stack([x["log"]["train_loss_gradient_reference2deformed"] for x in outputs]).mean(),
        )
        self.log(
            f"train_loss_mse_deformed2referencex",
            torch.stack([x["log"]["train_loss_mse_deformed2reference"] for x in outputs]).mean(),
        )
        self.log(
            f"train_loss_mse_reference2deformedx",
            torch.stack([x["log"]["train_loss_mse_reference2deformed"] for x in outputs]).mean(),
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
        self.log(
            f"recon_loss_deformed2referencex",
            torch.stack([x["log"]["recon_loss_deformed2reference"] for x in outputs]).mean(),
        )
        self.log(
            f"recon_loss_reference2deformedx",
            torch.stack([x["log"]["recon_loss_reference2deformed"] for x in outputs]).mean(),
        )
        if self.moco_loss_consenus_coefficient > 0:
            self.log(
                f"recon_loss_consensus_referencex",
                torch.stack([x["log"]["recon_loss_consensus_reference"] for x in outputs]).mean(),
            )
            self.log(
                f"recon_loss_consensus_deformed",
                torch.stack([x["log"]["recon_loss_consensus_deformed"] for x in outputs]).mean(),
            )

        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k in self.mse_vals.keys():
            mse_vals[k].update(self.mse_vals[k])
        for k in self.nmse_vals.keys():
            nmse_vals[k].update(self.nmse_vals[k])
        for k in self.ssim_vals.keys():
            ssim_vals[k].update(self.ssim_vals[k])
        for k in self.psnr_vals.keys():
            psnr_vals[k].update(self.psnr_vals[k])

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
            self.log(f"{metric}", value / tot_examples)
