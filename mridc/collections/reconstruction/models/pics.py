# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Dict, Tuple, Union

import bart
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import sense
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["PICS"]


class PICS(BaseMRIReconstructionModel, ABC):
    """
    Parallel-Imaging Compressed Sensing (PICS) reconstruction using the BART by Uecker, M. et al.

    References
    ----------

    ..

        Uecker, M. et al. (2015) ‘Berkeley Advanced Reconstruction Toolbox’, Proc. Intl. Soc. Mag. Reson. Med., 23.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.reg_wt = cfg_dict.get("reg_wt")
        self.num_iters = cfg_dict.get("num_iters")
        self._device = cfg_dict.get("_device")
        self.pics_centered = cfg_dict.get("pics_centered")
        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.num_cascades = cfg_dict.get("num_cascades")

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
            )

    @staticmethod
    def process_inputs(y, mask):
        """
        Process the inputs to the method.

        Parameters
        ----------
        y: Subsampled k-space data.
            list of torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            list of torch.Tensor, shape [1, 1, n_x, n_y, 1]

        Returns
        -------
        y: Subsampled k-space data.
            randomly selected y
        mask: Sampling mask.
            randomly selected mask
        r: Random index.
        """
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

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Predicted data.
        """
        if "cuda" in str(self._device):
            pred = bart.bart(1, f"pics -d0 -g -S -R W:7:0:{self.reg_wt} -i {self.num_iters}", y, sensitivity_maps)[0]
        else:
            pred = bart.bart(1, f"pics -d0 -S -R W:7:0:{self.reg_wt} -i {self.num_iters}", y, sensitivity_maps)[0]
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Test step.

        Parameters
        ----------
        batch: Batch of data.
            Dict of torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
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
        kspace, y, sensitivity_maps, mask, _, target, fname, slice_num, _ = batch
        y, mask, _ = self.process_inputs(y, mask)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)
            if self.coil_combination_method.upper() == "SENSE":
                target = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_maps,
                    dim=self.coil_dim,
                )

        y = torch.view_as_complex(y).permute(0, 2, 3, 1).detach().cpu().numpy()

        if sensitivity_maps is None and not self.sens_net:
            raise ValueError(
                "Sensitivity maps are required for PICS. "
                "Please set use_sens_net to True if you precomputed sensitivity maps are not available."
            )

        sensitivity_maps = torch.view_as_complex(sensitivity_maps)
        if self.pics_centered:
            sensitivity_maps = torch.fft.fftshift(sensitivity_maps, dim=self.spatial_dims)
        sensitivity_maps = sensitivity_maps.permute(0, 2, 3, 1).detach().cpu().numpy()  # type: ignore

        prediction = torch.from_numpy(self.forward(y, sensitivity_maps, mask, target)).unsqueeze(0)
        if self.pics_centered:
            prediction = torch.fft.fftshift(prediction, dim=self.spatial_dims)

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
