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
from mridc.collections.reconstruction.models.crossdomain.multicoil import MultiCoil
from mridc.collections.reconstruction.models.didn.didn import DIDN
from mridc.collections.reconstruction.models.mwcnn.mwcnn import MWCNN
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["KIKINet"]


class KIKINet(BaseMRIReconstructionModel, ABC):
    """
    Based on KIKINet implementation [1]_. Modified to work with multi-coil k-space data.

    References
    ----------
    .. [1] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled
    Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed,
    https://doi.org/10.1002/mrm.27201.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        kikinet_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = kikinet_cfg_dict.get("num_iter")
        self.no_dc = kikinet_cfg_dict.get("no_dc")

        kspace_model_architecture = kikinet_cfg_dict.get("kspace_model_architecture")

        if kspace_model_architecture == "CONV":
            kspace_model = Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=kikinet_cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=kikinet_cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=kikinet_cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=kikinet_cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=kikinet_cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=kikinet_cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model = NormUnet(
                kikinet_cfg_dict.get("kspace_unet_num_filters"),
                kikinet_cfg_dict.get("kspace_unet_num_pool_layers"),
                in_chans=2,
                out_chans=2,
                drop_prob=kikinet_cfg_dict.get("kspace_unet_dropout_probability"),
                padding_size=kikinet_cfg_dict.get("kspace_unet_padding_size"),
                normalize=kikinet_cfg_dict.get("kspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                f"KIKINet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = kikinet_cfg_dict.get("imspace_model_architecture")

        if image_model_architecture == "MWCNN":
            image_model = MWCNN(
                input_channels=2,
                first_conv_hidden_channels=kikinet_cfg_dict.get("image_mwcnn_hidden_channels"),
                num_scales=kikinet_cfg_dict.get("image_mwcnn_num_scales"),
                bias=kikinet_cfg_dict.get("image_mwcnn_bias"),
                batchnorm=kikinet_cfg_dict.get("image_mwcnn_batchnorm"),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model = NormUnet(
                kikinet_cfg_dict.get("imspace_unet_num_filters"),
                kikinet_cfg_dict.get("imspace_unet_num_pool_layers"),
                in_chans=2,
                out_chans=2,
                drop_prob=kikinet_cfg_dict.get("imspace_unet_dropout_probability"),
                padding_size=kikinet_cfg_dict.get("imspace_unet_padding_size"),
                normalize=kikinet_cfg_dict.get("imspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                f"KIKINet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        self.fft_type = kikinet_cfg_dict.get("fft_type")
        self._coil_dim = 1

        self.image_model_list = nn.ModuleList([image_model] * self.num_iter)
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * self.num_iter)

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = kikinet_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                kikinet_cfg_dict.get("sens_chans"),
                kikinet_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=kikinet_cfg_dict.get("sens_mask_type"),
                normalize=kikinet_cfg_dict.get("sens_normalize"),
            )

        self.train_loss_fn = SSIMLoss() if kikinet_cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if kikinet_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = kikinet_cfg_dict.get("output_type")

        self.dc_weight = torch.nn.Parameter(torch.ones(1))

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
        kspace = y.clone()
        zero = torch.zeros(1, 1, 1, 1, 1).to(kspace)

        for idx in range(self.num_iter):
            soft_dc = torch.where(mask.bool(), kspace - y, zero) * self.dc_weight

            kspace = self.kspace_model_list[idx](kspace)
            if kspace.shape[-1] != 2:
                kspace = kspace.permute(0, 1, 3, 4, 2).to(target)
            image = complex_mul(ifft2c(kspace, fft_type=self.fft_type), complex_conj(sensitivity_maps)).sum(1)
            image = self.image_model_list[idx](image.unsqueeze(1)).squeeze(1)

            if not self.no_dc:
                image = fft2c(complex_mul(image.unsqueeze(1), sensitivity_maps), fft_type=self.fft_type).type(
                    image.type()
                )
                image = kspace - soft_dc - image
                image = complex_mul(ifft2c(image, fft_type=self.fft_type), complex_conj(sensitivity_maps)).sum(1)

            if idx < self.num_iter - 1:
                kspace = fft2c(complex_mul(image.unsqueeze(1), sensitivity_maps), fft_type=self.fft_type).type(
                    image.type()
                )

        image = torch.view_as_complex(image)
        _, image = center_crop_to_smallest(target, image)
        return image

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
            f"train_loss_{acc}x": train_loss.item(),  # type: ignore
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
