# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.models.varnet.vn_block import VarNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["VarNet"]


class VarNet(BaseMRIReconstructionModel, ABC):
    """
    End-to-end Variational Network (VN) model implementation as presented in [1]_.

    References
    ----------

    .. [1] Sriram, A. et al. (2020) ‘End-to-End Variational Networks for Accelerated MRI Reconstruction’.
    Available at: https://github.com/facebookresearch/fastMRI.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        vn_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = vn_cfg_dict.get("no_dc")
        self.fft_type = vn_cfg_dict.get("fft_type")
        self.num_cascades = vn_cfg_dict.get("num_cascades")

        # Cascades of VN blocks
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet(
                        chans=vn_cfg_dict.get("channels"),
                        num_pools=vn_cfg_dict.get("pooling_layers"),
                        padding_size=vn_cfg_dict.get("padding_size"),
                        normalize=vn_cfg_dict.get("normalize"),
                    ),
                    fft_type=self.fft_type,
                    no_dc=self.no_dc,
                )
                for _ in range(self.num_cascades)
            ]
        )

        self.output_type = vn_cfg_dict.get("output_type")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = vn_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                vn_cfg_dict.get("sens_chans"),
                vn_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=vn_cfg_dict.get("sens_mask_type"),
                normalize=vn_cfg_dict.get("sens_normalize"),
            )

        # initialize weights if not using pretrained vn
        # TODO if not vn_cfg_dict.get("pretrained", False)

        self.train_loss_fn = SSIMLoss() if vn_cfg_dict.get("loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if vn_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.accumulate_estimates = vn_cfg_dict.get("accumulate_estimates")

        # Initialize data consistency term
        # TODO: make this configurable
        self.dc_weight = nn.Parameter(torch.ones(1))

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor = None,
    ) -> Union[list, Any]:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final estimation of the network.
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        estimation = y.clone()

        cascades_etas = []
        for i, cascade in enumerate(self.cascades):
            # Forward pass through the cascades
            estimation = cascade(estimation, y, sensitivity_maps, mask)
            cascades_etas.append(estimation)

        if self.accumulate_estimates:
            return cascades_etas
        else:
            return self.process_intermediate_eta(estimation, sensitivity_maps, target, do_coil_combination=True)

    def process_intermediate_eta(self, eta, sensitivity_maps, target, do_coil_combination=False):
        """Process the intermediate eta to be used in the loss function."""
        # Take the last time step of the eta
        if not self.no_dc or do_coil_combination:
            eta = ifft2c(eta, fft_type=self.fft_type)
            eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta

    def process_loss(self, target, eta, set_loss_fn):
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

        if self.accumulate_estimates:
            cascades_loss = [loss_fn(target, cascade_eta) for cascade_eta in eta]
            return sum(list(sum(cascades_loss))) / len(self.cascades)
        else:
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
        Training step for the VarNet.
        Args:
            batch: A dictionary of the form {
                'y': subsampled kspace,
                'sensitivity_maps': sensitivity_maps,
                'mask': mask,
                'eta': initial estimation,
                'hx': hidden states,
                'target': target,
                'sigma': sigma
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

        if self.accumulate_estimates:
            try:
                etas = next(etas)
            except StopIteration:
                pass

            train_loss = sum(self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn))
        else:
            train_loss = self.process_loss(target, etas, set_loss_fn=self.train_loss_fn)

        acc = r if r != 0 else acc

        tensorboard_logs = {
            f"train_loss_{str(acc)}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step for the VarNet."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, target)

        if self.accumulate_estimates:
            try:
                etas = next(etas)
            except StopIteration:
                pass

            val_loss = sum(self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn))
        else:
            val_loss = self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn)

        if isinstance(etas, list):
            etas = etas[-1][-1]

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
        """Test step for the VarNet."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        prediction = self.forward(y, sensitivity_maps, mask, target)

        if self.accumulate_estimates:
            try:
                prediction = next(prediction)
            except StopIteration:
                pass

        if isinstance(prediction, list):
            prediction = prediction[-1][-1]

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
