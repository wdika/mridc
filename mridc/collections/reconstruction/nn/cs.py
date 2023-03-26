# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import sigpy.mri as spmri
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from sigpy.pytorch import from_pytorch, to_pytorch

import mridc.collections.reconstruction.nn.base as base_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils

__all__ = ["CS"]


class CS(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Compressed-Sensing reconstruction using Wavelet transform, as presented in [1].

    References
    ----------
    .. [1] Lustig, M., Donoho, D. L., Santos, J. M., & Pauly, J. M. (2008). Compressed sensing MRI. IEEE signal
    processing magazine, 25(2), 72-82.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cs_type = cfg_dict.get("cs_type")
        self.reg_wt = cfg_dict.get("reg_wt")
        self.num_iters = cfg_dict.get("num_iters")
        self.center_kspace = cfg_dict.get("center_kspace")
        self.center_sensitivity_maps = cfg_dict.get("center_sensitivity_maps")
        self.center_mask = cfg_dict.get("center_mask")
        self.center_reconstruction = cfg_dict.get("center_reconstruction")
        self.spatial_dims = cfg_dict.get("spatial_dims")

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,  # noqa: W0613
        init_pred: torch.Tensor,  # noqa: W0613
        target: torch.Tensor,  # noqa: W0613
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_pred : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        target : torch.Tensor
            Target data to compute the loss. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        y = torch.view_as_complex(y)
        sensitivity_maps = torch.view_as_complex(sensitivity_maps)
        mask = mask.squeeze(-1)
        if self.center_kspace:
            y = fft.ifftshift(y, dim=self.spatial_dims)
        if self.center_sensitivity_maps:
            sensitivity_maps = fft.fftshift(sensitivity_maps, dim=self.spatial_dims)
        if self.center_mask:
            mask = fft.fftshift(mask, dim=self.spatial_dims)
        y = from_pytorch(y.detach().cpu())
        sensitivity_maps = from_pytorch(sensitivity_maps.detach().cpu())
        mask = from_pytorch(mask.detach().cpu())
        if self.cs_type == "l1_wavelet":
            # TODO: find a fix for this. If a cross of zeros appears on the reconstruction, then we need to manually go
            #  to sigpy.wavelet.py and fftshift the wavelet coefficients before the inverse transform and then
            #  ifftshift the result. This is a bug in sigpy (?).
            prediction = torch.stack(
                [
                    to_pytorch(
                        spmri.app.L1WaveletRecon(
                            y[batch_idx],
                            sensitivity_maps[batch_idx],
                            self.reg_wt,
                            weights=mask[batch_idx],
                            max_iter=self.num_iters,
                            show_pbar=False,
                        ).run()
                    )
                    for batch_idx in range(y.shape[0])
                ],
                dim=0,
            )
        elif self.cs_type == "total_variation":
            prediction = torch.stack(
                [
                    to_pytorch(
                        spmri.app.TotalVariationRecon(
                            y[batch_idx],
                            sensitivity_maps[batch_idx],
                            self.reg_wt,
                            weights=mask[batch_idx],
                            max_iter=self.num_iters,
                            show_pbar=False,
                        ).run()
                    )
                    for batch_idx in range(y.shape[0])
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Unknown cs_type: {self.cs_type}")
        if prediction.shape[-1] == 2:
            prediction = torch.view_as_complex(prediction)
        if self.center_reconstruction:
            prediction = fft.fftshift(prediction, dim=self.spatial_dims)
        return prediction
