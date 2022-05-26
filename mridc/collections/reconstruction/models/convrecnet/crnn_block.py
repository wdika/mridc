# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Optional, Tuple, Union

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
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
        self,
        model: torch.nn.Module,
        num_iterations: int = 10,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
    ):
        """
        Initialize the model block.

        Parameters
        ----------
        model: Model to apply soft data consistency.
        num_iterations: Number of iterations.
        fft_centered: Whether to use centered FFT.
        fft_normalization: Whether to use normalized FFT.
        spatial_dims: Spatial dimensions of the input.
        coil_dim: Dimension of the coil.
        no_dc: Whether to remove the DC component.
        """
        super().__init__()

        self.model = model
        self.num_iterations = num_iterations
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
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
        return fft2(
            complex_mul(x, sens_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

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
        x = ifft2(x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims)
        return complex_mul(x, complex_conj(sens_maps)).sum(self.coil_dim)

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
            eta = self.sens_expand(eta.unsqueeze(self.coil_dim), sens_maps)

            if not self.no_dc:
                # TODO: Check if this is correct
                eta = pred - soft_dc - eta
            pred = eta

            preds.append(eta)

        return preds
