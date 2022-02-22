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
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.conv.conv2d import Conv2d
from mridc.collections.reconstruction.models.crossdomain.crossdomain import CrossDomainNetwork
from mridc.collections.reconstruction.models.crossdomain.multicoil import MultiCoil
from mridc.collections.reconstruction.models.didn.didn import DIDN
from mridc.collections.reconstruction.models.mwcnn.mwcnn import MWCNN
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["XPDNet"]


class XPDNet(BaseMRIReconstructionModel, ABC):
    """XPDNet as implemented in [1]_.

    References
    ----------
    .. [1] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge.”
    ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        xpdnet_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_primal = xpdnet_cfg_dict.get("num_primal")
        num_dual = xpdnet_cfg_dict.get("num_dual")
        num_iter = xpdnet_cfg_dict.get("num_iter")

        kspace_model_architecture = xpdnet_cfg_dict.get("kspace_model_architecture")
        dual_conv_hidden_channels = xpdnet_cfg_dict.get("dual_conv_hidden_channels")
        dual_conv_num_dubs = xpdnet_cfg_dict.get("dual_conv_num_dubs")
        dual_conv_batchnorm = xpdnet_cfg_dict.get("dual_conv_batchnorm")
        dual_didn_hidden_channels = xpdnet_cfg_dict.get("dual_didn_hidden_channels")
        dual_didn_num_dubs = xpdnet_cfg_dict.get("dual_didn_num_dubs")
        dual_didn_num_convs_recon = xpdnet_cfg_dict.get("dual_didn_num_convs_recon")

        if xpdnet_cfg_dict.get("use_primal_only"):
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        Conv2d(
                            2 * (num_dual + num_primal + 1),
                            2 * num_dual,
                            dual_conv_hidden_channels,
                            dual_conv_num_dubs,
                            batchnorm=dual_conv_batchnorm,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        DIDN(
                            in_channels=2 * (num_dual + num_primal + 1),
                            out_channels=2 * num_dual,
                            hidden_channels=dual_didn_hidden_channels,
                            num_dubs=dual_didn_num_dubs,
                            num_convs_recon=dual_didn_num_convs_recon,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model_list = nn.ModuleList(
                [
                    MultiCoil(
                        NormUnet(
                            xpdnet_cfg_dict.get("kspace_unet_num_filters"),
                            xpdnet_cfg_dict.get("kspace_unet_num_pool_layers"),
                            in_chans=2 * (num_dual + num_primal + 1),
                            out_chans=2 * num_dual,
                            drop_prob=xpdnet_cfg_dict.get("kspace_unet_dropout_probability"),
                            padding_size=xpdnet_cfg_dict.get("kspace_unet_padding_size"),
                            normalize=xpdnet_cfg_dict.get("kspace_unet_normalize"),
                        ),
                        coil_to_batch=True,
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                "XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = xpdnet_cfg_dict.get("image_model_architecture")
        mwcnn_hidden_channels = xpdnet_cfg_dict.get("mwcnn_hidden_channels")
        mwcnn_num_scales = xpdnet_cfg_dict.get("mwcnn_num_scales")
        mwcnn_bias = xpdnet_cfg_dict.get("mwcnn_bias")
        mwcnn_batchnorm = xpdnet_cfg_dict.get("mwcnn_batchnorm")

        if image_model_architecture == "MWCNN":
            image_model_list = nn.ModuleList(
                [
                    nn.Sequential(
                        MWCNN(
                            input_channels=2 * (num_primal + num_dual),
                            first_conv_hidden_channels=mwcnn_hidden_channels,
                            num_scales=mwcnn_num_scales,
                            bias=mwcnn_bias,
                            batchnorm=mwcnn_batchnorm,
                        ),
                        nn.Conv2d(2 * (num_primal + num_dual), 2 * num_primal, kernel_size=3, padding=1),
                    )
                    for _ in range(num_iter)
                ]
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model_list = nn.ModuleList(
                [
                    NormUnet(
                        xpdnet_cfg_dict.get("imspace_unet_num_filters"),
                        xpdnet_cfg_dict.get("imspace_unet_num_pool_layers"),
                        in_chans=2 * (num_primal + num_dual),
                        out_chans=2 * num_primal,
                        drop_prob=xpdnet_cfg_dict.get("imspace_unet_dropout_probability"),
                        padding_size=xpdnet_cfg_dict.get("imspace_unet_padding_size"),
                        normalize=xpdnet_cfg_dict.get("imspace_unet_normalize"),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(f"Image model architecture {image_model_architecture} not found for XPDNet.")

        self.fft_type = xpdnet_cfg_dict.get("fft_type")

        self.xpdnet = CrossDomainNetwork(
            fft_type=self.fft_type,
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
            normalize_image=xpdnet_cfg_dict.get("normalize_image"),
        )

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = xpdnet_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                xpdnet_cfg_dict.get("sens_chans"),
                xpdnet_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=xpdnet_cfg_dict.get("sens_mask_type"),
                normalize=xpdnet_cfg_dict.get("sens_normalize"),
            )

        self.train_loss_fn = SSIMLoss() if xpdnet_cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if xpdnet_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = xpdnet_cfg_dict.get("output_type")

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
        eta = self.xpdnet(y, sensitivity_maps, mask)
        eta = (eta**2).sqrt().sum(-1)
        _, eta = center_crop_to_smallest(target, eta)
        return eta

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
