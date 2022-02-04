# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Dict, Generator, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["JointICNet"]


class JointICNet(BaseMRIReconstructionModel, ABC):
    """
    Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) implementation as
    presented in [1]_.

    References
    ----------
    .. [1] Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
    (Joint-ICNet) for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    IEEE, 2021, pp. 5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        jointicnet_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = jointicnet_cfg_dict.get("num_iter")

        self.kspace_model = NormUnet(
            jointicnet_cfg_dict.get("kspace_unet_num_filters"),
            jointicnet_cfg_dict.get("kspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=jointicnet_cfg_dict.get("kspace_unet_dropout_probability"),
            padding_size=jointicnet_cfg_dict.get("kspace_unet_padding_size"),
            normalize=jointicnet_cfg_dict.get("kspace_unet_normalize"),
        )

        self.image_model = NormUnet(
            jointicnet_cfg_dict.get("imspace_unet_num_filters"),
            jointicnet_cfg_dict.get("imspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=jointicnet_cfg_dict.get("imspace_unet_dropout_probability"),
            padding_size=jointicnet_cfg_dict.get("imspace_unet_padding_size"),
            normalize=jointicnet_cfg_dict.get("imspace_unet_normalize"),
        )

        self.sens_net = NormUnet(
            jointicnet_cfg_dict.get("sens_unet_num_filters"),
            jointicnet_cfg_dict.get("sens_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=jointicnet_cfg_dict.get("sens_unet_dropout_probability"),
            padding_size=jointicnet_cfg_dict.get("sens_unet_padding_size"),
            normalize=jointicnet_cfg_dict.get("sens_unet_normalize"),
        )

        self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

        self.reg_param_I = nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_C = nn.Parameter(torch.ones(self.num_iter))

        self.lr_image = nn.Parameter(torch.ones(self.num_iter))
        self.lr_sens = nn.Parameter(torch.ones(self.num_iter))

        self.fft_type = jointicnet_cfg_dict.get("fft_type")
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

        self.train_loss_fn = SSIMLoss() if jointicnet_cfg_dict.get("loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if jointicnet_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = jointicnet_cfg_dict.get("output_type")

    def _image_model(self, image):
        image = image.permute(0, 3, 1, 2)
        return self.image_model(image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace):
        kspace = kspace.permute(0, 3, 1, 2)
        return self.kspace_model(kspace).permute(0, 2, 3, 1).contiguous()

    def _sens_model(self, sensitivity_map):
        return (
            self._compute_model_per_coil(self.sens_net, sensitivity_map.permute(0, 1, 4, 2, 3))
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def _compute_model_per_coil(self, model, data):
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self._coil_dim)
        return output

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor = None,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final estimation of the network.
        """
        image = complex_mul(
            ifft2c(torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), y), fft_type=self.fft_type),
            complex_conj(sensitivity_maps),
        ).sum(1)

        for idx in range(self.num_iter):
            step_sensitivity_maps = (
                2
                * self.lr_sens[idx]
                * (
                    complex_mul(
                        ifft2c(
                            torch.where(
                                mask == 0,
                                torch.tensor([0.0], dtype=image.dtype).to(image.device),
                                fft2c(
                                    complex_mul(image.unsqueeze(self._coil_dim), sensitivity_maps),
                                    fft_type=self.fft_type,
                                ),
                            ),
                            fft_type=self.fft_type,
                        ),
                        complex_conj(image).unsqueeze(self._coil_dim),
                    )
                    + self.reg_param_C[idx] * (sensitivity_maps - self._sens_model(ifft2c(y, fft_type=self.fft_type)))
                )
            )

            sensitivity_maps = sensitivity_maps - step_sensitivity_maps
            sensitivity_maps_norm = torch.sqrt(((sensitivity_maps**2).sum(-1)).sum(self._coil_dim))
            sensitivity_maps_norm = sensitivity_maps_norm.unsqueeze(-1).unsqueeze(self._coil_dim)
            sensitivity_maps = torch.where(
                sensitivity_maps == 0,
                torch.tensor([0.0], dtype=sensitivity_maps.dtype).to(sensitivity_maps.device),
                sensitivity_maps / sensitivity_maps_norm,
            )

            input_kspace = fft2c(image, fft_type=self.fft_type)

            forward = (
                torch.where(
                    mask == 0,
                    torch.tensor([0.0], dtype=image.dtype).to(image.device),
                    fft2c(complex_mul(image.unsqueeze(self._coil_dim), sensitivity_maps), fft_type=self.fft_type),
                )
                - y
            )
            backward = complex_mul(
                ifft2c(
                    torch.where(mask == 0, torch.tensor([0.0], dtype=forward.dtype).to(forward.device), forward),
                    fft_type=self.fft_type,
                ),
                complex_conj(sensitivity_maps),
            ).sum(self._coil_dim)
            _input_kspace = self._kspace_model(input_kspace)
            _backward = complex_mul(
                ifft2c(
                    torch.where(
                        mask == 0,
                        torch.tensor([0.0], dtype=_input_kspace.dtype).to(_input_kspace.device),
                        _input_kspace,
                    ),
                    fft_type=self.fft_type,
                    fft_dim=[d - 1 for d in self._spatial_dims],
                ),
                complex_conj(sensitivity_maps),
            ).sum(self._coil_dim)

            step_image = (
                2
                * self.lr_image[idx]
                * (
                    backward
                    + self.reg_param_I[idx] * (image - self._image_model(image))
                    + self.reg_param_F[idx] * (image - _backward)
                )
            )

            image = image - step_image

        out_image = self.conv_out(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out_image = (out_image**2).sum(-1).sqrt()
        _, out_image = center_crop_to_smallest(target, out_image)
        return out_image

    @staticmethod
    def process_loss(target, eta, set_loss_fn):
        """Calculate the loss."""
        target = torch.abs(target / torch.max(torch.abs(target)))
        eta = torch.abs(eta / torch.max(torch.abs(eta)))

        if "ssim" in str(set_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                return set_loss_fn(
                    x.unsqueeze(dim=1),
                    y.unsqueeze(dim=1),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return set_loss_fn(x, y)

        return loss_fn(target, eta)

    @staticmethod
    def process_inputs(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for the XPDNet.
        Args:
            batch: A dictionary of the form {
                'y': subsampled kspace,
                'sensitivity_maps': sensitivity_maps,
                'mask': mask,
                'target': target,
                'scaling_factor': scaling_factor
                }
            batch_idx: The index of the batch.
        Returns:
            A dictionary of the form {
                'loss': loss value,
                'log': log,
            }
        """
        y, sensitivity_maps, mask, _, target, _, _, acc, _, _ = batch
        y, mask, r = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, target)
        train_loss = self.process_loss(target, etas, set_loss_fn=self.train_loss_fn)

        acc = r if r != 0 else acc

        tensorboard_logs = {
            f"train_loss_{str(acc)}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step for the XPDNet."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, target)
        val_loss = self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn)

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(etas).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return {
            "val_loss": val_loss,
        }

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for the XPDNet."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        prediction = self.forward(y, sensitivity_maps, mask, target)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore

        output = torch.abs(prediction).detach().cpu()
        output = output / output.max()  # type: ignore

        target = torch.abs(target).detach().cpu()
        target = target / target.max()  # type: ignore

        error = torch.abs(target - output)

        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, prediction.detach().cpu().numpy()
