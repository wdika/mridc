# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Union

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class DataConsistencyLayer(torch.nn.Module):
    """
    Data consistency layer for the CRNN.
    This layer is used to ensure that the output of the CRNN is the same as the input.
    """

    def __init__(self):
        """Initializes the data consistency layer."""
        super().__init__()
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, pred_kspace, ref_kspace, mask):
        """Forward pass of the data consistency layer."""
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred_kspace)
        return torch.where(mask.bool(), pred_kspace - ref_kspace, zero) * self.dc_weight


class RecurrentConvolutionalNetBlock(torch.nn.Module):
    """
    Model block for Recurrent Convolution Neural Network inspired by [1]_.

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
    """

    def __init__(
        self, model: torch.nn.Module, num_iterations: int = 10, fft_type: str = "orthogonal", no_dc: bool = False
    ):
        """
        Initialize the model block.

        Parameters
        ----------
        model: Model to apply soft data consistency.
        num_iterations: Number of iterations.
        fft_type: Type of FFT to use.
        no_dc: Whether to remove the DC component.
        """
        super().__init__()

        self.model = model
        self.num_iterations = num_iterations
        self.fft_type = fft_type
        self.no_dc = no_dc

        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Expand the sensitivity maps to the same size as the input.

        Parameters
        ----------
        x: Input data.
        sens_maps: Sensitivity maps.

        Returns
        -------
        SENSE reconstruction expanded to the same size as the input.
        """
        return fft2c(complex_mul(x, sens_maps), fft_type=self.fft_type)

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Reduce the sensitivity maps to the same size as the input.

        Parameters
        ----------
        x: Input data.
        sens_maps: Sensitivity maps.

        Returns
        -------
        SENSE reconstruction reduced to the same size as the input.
        """
        x = ifft2c(x, fft_type=self.fft_type)
        return complex_mul(x, complex_conj(sens_maps)).sum(1)

    def forward(
        self,
        ref_kspace: torch.Tensor,
        sens_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Union[torch.Tensor, Any]]:
        """
        Forward pass of the model.

        Parameters
        ----------
        ref_kspace: Reference k-space data.
        sens_maps: Sensitivity maps.
        mask: Mask to apply to the data.

        Returns
        -------
        Reconstructed image.
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(ref_kspace)
        pred = ref_kspace.clone()

        preds = []
        for _ in range(self.num_iterations):
            soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight

            eta = self.sens_reduce(pred, sens_maps)
            eta = self.model(eta.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + eta
            eta = self.sens_expand(eta.unsqueeze(1), sens_maps)

            if not self.no_dc:
                # TODO: Check if this is correct
                eta = pred - soft_dc - eta
            pred = eta

            preds.append(eta)

        return preds
