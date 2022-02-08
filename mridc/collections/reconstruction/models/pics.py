# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Dict, Tuple, Union

# TODO: Currently environment path variables need to be exported every time to find bart, otherwise it throws an
#  import error. Need to fix this.
import bart

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["PICS"]


class PICS(BaseMRIReconstructionModel, ABC):
    """
    Parallel-Imaging Compressed Sensing (PICS) reconstruction using the BART [1].

    References
    ----------

    .. [1] Uecker, M. et al. (2015) ‘Berkeley Advanced Reconstruction Toolbox’, Proc. Intl. Soc. Mag. Reson. Med., 23.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        pics_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.reg_wt = pics_cfg_dict.get("reg_wt")
        self.num_iters = pics_cfg_dict.get("num_iters")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = pics_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                pics_cfg_dict.get("sens_chans"),
                pics_cfg_dict.get("sens_pools"),
                fft_type=pics_cfg_dict.get("fft_type"),
                mask_type=pics_cfg_dict.get("sens_mask_type"),
                normalize=pics_cfg_dict.get("sens_normalize"),
            )

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

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor = None,
    ) -> Union[list, Any]:
        """
        Forward pass of PICS.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final estimation of PICS.
        """
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps

        if "cuda" in str(self.device):
            pred = bart.bart(1, f"pics -d0 -g -S -R W:7:0:{self.reg_wt} -i {self.num_iters}", y, sensitivity_maps)[0]
        else:
            pred = bart.bart(1, f"pics -d0 -S -R W:7:0:{self.reg_wt} -i {self.num_iters}", y, sensitivity_maps)[0]
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for PICS."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)

        y = torch.view_as_complex(y).permute(0, 2, 3, 1).detach().cpu().numpy()
        sensitivity_maps = (
            torch.fft.fftshift(torch.view_as_complex(sensitivity_maps), dim=(-2, -1))
            .permute(0, 2, 3, 1)
            .detach()
            .cpu()
            .numpy()
        )

        prediction = self.forward(y, sensitivity_maps, mask, target)
        prediction = torch.fft.fftshift(torch.from_numpy(prediction), dim=(-2, -1)).unsqueeze(0)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        output = torch.abs(prediction).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, prediction.detach().cpu().numpy()
