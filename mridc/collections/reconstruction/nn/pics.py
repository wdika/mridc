# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
import sys
from abc import ABC
from typing import Any, Dict, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.metrics import reconstruction_metrics

os.environ["TOOLBOX_PATH"] = "/opt/amc/bart-0.8.00/bin/"
os.environ["PYTHONPATH"] = "/opt/amc/bart-0.8.00/python"
sys.path.append(os.environ["TOOLBOX_PATH"])
sys.path.append(os.environ["PYTHONPATH"])
try:
    import bart
except ImportError:
    print("BART is not installed")
__all__ = ["PICS"]


class PICS(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
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

        self.num_iters = cfg_dict.get("num_iters", 10)
        self.reg_wt = cfg_dict.get("reg_wt", 0.01)
        self.centered = cfg_dict.get("centered", False)

    @common_classes.typecheck()
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
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _, attrs = batch

        kspace, y, mask, init_pred, target, r = self.process_inputs(kspace, y, mask, init_pred, target)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)
            if self.coil_combination_method.upper() == "SENSE":
                target = utils.sense(
                    fft.ifft2(
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
                "Please set use_sens_net to True if precomputed sensitivity maps are not available."
            )

        sensitivity_maps = torch.view_as_complex(sensitivity_maps)
        if self.centered:
            sensitivity_maps = torch.fft.fftshift(sensitivity_maps, dim=self.spatial_dims)
        sensitivity_maps = sensitivity_maps.permute(0, 2, 3, 1).detach().cpu().numpy()  # type: ignore

        prediction = torch.from_numpy(self.forward(y, sensitivity_maps, mask, target)).unsqueeze(0)
        if self.centered:
            prediction = torch.fft.fftshift(prediction, dim=self.spatial_dims)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        output = prediction / torch.max(torch.abs(prediction))  # type: ignore
        if target.shape[-1] == 2:  # type: ignore
            target = torch.view_as_complex(target)  # type: ignore
        if target.shape[-1] == 1:  # type: ignore
            target = target.squeeze(-1)  # type: ignore
        target = target / torch.max(torch.abs(target))  # type: ignore
        output = torch.abs(output).detach().cpu()
        target = torch.abs(target).detach().cpu()
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        target = target.numpy()  # type: ignore
        output = output.numpy()  # type: ignore
        self.mse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.mse(target, output)).view(1)
        self.nmse_vals[fname][slice_num] = torch.tensor(reconstruction_metrics.nmse(target, output)).view(1)
        self.ssim_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.ssim(target, output, maxval=output.max() - output.min())
        ).view(1)
        self.psnr_vals[fname][slice_num] = torch.tensor(
            reconstruction_metrics.psnr(target, output, maxval=output.max() - output.min())
        ).view(1)

        return name, slice_num, prediction.detach().cpu().numpy()
