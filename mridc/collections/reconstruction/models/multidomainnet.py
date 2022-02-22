# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Dict, Generator, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.multidomain.multidomain import MultiDomainUnet2d, StandardizationLayer
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["MultiDomainNet"]


class MultiDomainNet(BaseMRIReconstructionModel, ABC):
    """Feature-level multi-domain module. Inspired by AIRS Medical submission to the Fast MRI 2020 challenge."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        self._coil_dim = 1
        self._complex_dim = -1

        multidomainnet_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        standardization = multidomainnet_cfg_dict["standardization"]
        if standardization:
            self.standardization = StandardizationLayer(self._coil_dim, self._complex_dim)

        self.fft_type = multidomainnet_cfg_dict.get("fft_type")

        self.unet = MultiDomainUnet2d(
            in_channels=4 if standardization else 2,  # if standardization, in_channels is 4 due to standardized input
            out_channels=2,
            num_filters=multidomainnet_cfg_dict["num_filters"],
            num_pool_layers=multidomainnet_cfg_dict["num_pool_layers"],
            dropout_probability=multidomainnet_cfg_dict["dropout_probability"],
            fft_type=self.fft_type,
        )

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = multidomainnet_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                multidomainnet_cfg_dict.get("sens_chans"),
                multidomainnet_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=multidomainnet_cfg_dict.get("sens_mask_type"),
                normalize=multidomainnet_cfg_dict.get("sens_normalize"),
            )

        self.train_loss_fn = SSIMLoss() if multidomainnet_cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if multidomainnet_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = multidomainnet_cfg_dict.get("output_type")

    def _compute_model_per_coil(self, model, data):
        """Computes model per coil."""
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
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        image = ifft2c(y, fft_type=self.fft_type)

        if hasattr(self, "standardization"):
            image = self.standardization(image, sensitivity_maps)

        output_image = self._compute_model_per_coil(self.unet, image.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        output_image = coil_combination(output_image, sensitivity_maps, method=self.output_type, dim=1)
        output_image = torch.view_as_complex(output_image)
        _, output_image = center_crop_to_smallest(target, output_image)
        return output_image

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
