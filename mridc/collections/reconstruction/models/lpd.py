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
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.conv.conv2d import Conv2d
from mridc.collections.reconstruction.models.didn.didn import DIDN
from mridc.collections.reconstruction.models.mwcnn.mwcnn import MWCNN
from mridc.collections.reconstruction.models.primaldual.pd import DualNet, PrimalNet
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["LPDNet"]


class LPDNet(BaseMRIReconstructionModel, ABC):
    """
    Learned Primal Dual network implementation inspired by [1]_.

    References
    ----------
    .. [1] Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical Imaging,
    vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        lpd_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = lpd_cfg_dict.get("num_iter")
        self.num_primal = lpd_cfg_dict.get("num_primal")
        self.num_dual = lpd_cfg_dict.get("num_dual")

        primal_model_architecture = lpd_cfg_dict.get("primal_model_architecture")

        if primal_model_architecture == "MWCNN":
            primal_model = nn.Sequential(
                *[
                    MWCNN(
                        input_channels=2 * (self.num_primal + 1),
                        first_conv_hidden_channels=lpd_cfg_dict.get("primal_mwcnn_hidden_channels"),
                        num_scales=lpd_cfg_dict.get("primal_mwcnn_num_scales"),
                        bias=lpd_cfg_dict.get("primal_mwcnn_bias"),
                        batchnorm=lpd_cfg_dict.get("primal_mwcnn_batchnorm"),
                    ),
                    nn.Conv2d(2 * (self.num_primal + 1), 2 * self.num_primal, kernel_size=1),
                ]
            )
        elif primal_model_architecture in ["UNET", "NORMUNET"]:
            primal_model = NormUnet(
                lpd_cfg_dict.get("primal_unet_num_filters"),
                lpd_cfg_dict.get("primal_unet_num_pool_layers"),
                in_chans=2 * (self.num_primal + 1),
                out_chans=2 * self.num_primal,
                drop_prob=lpd_cfg_dict.get("primal_unet_dropout_probability"),
                padding_size=lpd_cfg_dict.get("primal_unet_padding_size"),
                normalize=lpd_cfg_dict.get("primal_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                f"LPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'UNet'."
                f"Got kspace_model_architecture == {primal_model_architecture}."
            )

        dual_model_architecture = lpd_cfg_dict.get("dual_model_architecture")

        if dual_model_architecture == "CONV":
            dual_model = Conv2d(
                in_channels=2 * (self.num_dual + 2),
                out_channels=2 * self.num_dual,
                hidden_channels=lpd_cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=lpd_cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=lpd_cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif dual_model_architecture == "DIDN":
            dual_model = DIDN(
                in_channels=2 * (self.num_dual + 2),
                out_channels=2 * self.num_dual,
                hidden_channels=lpd_cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=lpd_cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=lpd_cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif dual_model_architecture in ["UNET", "NORMUNET"]:
            dual_model = NormUnet(
                lpd_cfg_dict.get("dual_unet_num_filters"),
                lpd_cfg_dict.get("dual_unet_num_pool_layers"),
                in_chans=2 * (self.num_dual + 2),
                out_chans=2 * self.num_dual,
                drop_prob=lpd_cfg_dict.get("dual_unet_dropout_probability"),
                padding_size=lpd_cfg_dict.get("dual_unet_padding_size"),
                normalize=lpd_cfg_dict.get("dual_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                f"LPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got kspace_model_architecture == {dual_model_architecture}."
            )

        self.primal_net = nn.ModuleList(
            [PrimalNet(self.num_primal, primal_architecture=primal_model) for _ in range(self.num_iter)]
        )
        self.dual_net = nn.ModuleList(
            [DualNet(self.num_dual, dual_architecture=dual_model) for _ in range(self.num_iter)]
        )

        self.fft_type = lpd_cfg_dict.get("fft_type")
        self._coil_dim = 1

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = lpd_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                lpd_cfg_dict.get("sens_chans"),
                lpd_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=lpd_cfg_dict.get("sens_mask_type"),
                normalize=lpd_cfg_dict.get("sens_normalize"),
            )

        self.train_loss_fn = SSIMLoss() if lpd_cfg_dict.get("loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if lpd_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = lpd_cfg_dict.get("output_type")

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
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps

        input_image = complex_mul(
            ifft2c(torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), y), fft_type=self.fft_type),
            complex_conj(sensitivity_maps),
        ).sum(1)
        dual_buffer = torch.cat([y] * self.num_dual, -1).to(y.device)
        primal_buffer = torch.cat([input_image] * self.num_primal, -1).to(y.device)

        for idx in range(self.num_iter):
            # Dual
            f_2 = primal_buffer[..., 2:4].clone()
            f_2 = torch.where(
                mask == 0,
                torch.tensor([0.0], dtype=f_2.dtype).to(f_2.device),
                fft2c(complex_mul(f_2.unsqueeze(1), sensitivity_maps), fft_type=self.fft_type).type(f_2.type()),
            )
            dual_buffer = self.dual_net[idx](dual_buffer, f_2, y)

            # Primal
            h_1 = dual_buffer[..., 0:2].clone()
            h_1 = complex_mul(
                ifft2c(
                    torch.where(mask == 0, torch.tensor([0.0], dtype=h_1.dtype).to(h_1.device), h_1),
                    fft_type=self.fft_type,
                ),
                complex_conj(sensitivity_maps),
            ).sum(1)
            primal_buffer = self.primal_net[idx](primal_buffer, h_1)

        output = primal_buffer[..., 0:2]
        output = (output ** 2).sum(-1).sqrt()
        _, output = center_crop_to_smallest(target, output)
        return output

    @staticmethod
    def process_loss(target, eta, set_loss_fn):
        """Calculate the loss."""
        target = torch.abs(target / torch.max(torch.abs(target)))

        if "ssim" in str(set_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                return set_loss_fn(
                    x.unsqueeze(dim=1),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=1),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return set_loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

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
