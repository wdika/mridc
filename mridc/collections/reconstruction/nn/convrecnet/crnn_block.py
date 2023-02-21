# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Optional, Tuple, Union

import torch

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils


class DataConsistencyLayer(torch.nn.Module):
    """
    Data consistency layer for the CRNN,  inspired by [1].

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent
        Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1,
        pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
    """

    def __init__(self):
        """Initializes the data consistency layer."""
        super().__init__()
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, pred_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the data consistency layer."""
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred_kspace)
        return torch.where(mask.bool(), pred_kspace - ref_kspace, zero) * self.dc_weight


class RecurrentConvolutionalNetBlock(torch.nn.Module):
    """
    Model block for Recurrent Convolution Neural Network inspired by [1].

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent
        Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1,
        pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.

    Parameters
    ----------
    model : torch.nn.Module
        Model to apply soft data consistency.
    num_iterations : int, optional
        Number of iterations. Default is ``10``.
    fft_centered : bool, optional
        Whether to use centered FFT. Default is ``False``.
    fft_normalization : str, optional
        Whether to use normalized FFT. Default is ``"backward"``.
    spatial_dims : tuple, optional
        Spatial dimensions of the input. Default is ``None``.
    coil_dim : int, optional
        Dimension of the coil. Default is ``1``.
    no_dc : bool, optional
        Whether to remove the DC component. Default is ``False``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_iterations: int = 10,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
    ):
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
        Combines the sensitivity maps with coil-combined data to get multicoil data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        sens_maps : torch.Tensor
            Coil Sensitivity maps.

        Returns
        -------
        torch.Tensor
            Expanded multicoil data.
        """
        return fft.fft2(
            utils.complex_mul(x, sens_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Combines the sensitivity maps with multicoil data to get coil-combined data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        sens_maps : torch.Tensor
            Coil Sensitivity maps.

        Returns
        -------
        torch.Tensor
            SENSE coil-combined reconstruction.
        """
        x = fft.ifft2(
            x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(dim=self.coil_dim)

    def forward(
        self,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Union[torch.Tensor, Any]]:
        """
        Forward pass of the model block.

        Parameters
        ----------
        ref_kspace : torch.Tensor
            Reference k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]

        Returns
        -------
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(ref_kspace)
        pred = ref_kspace.clone()

        preds = []
        for _ in range(self.num_iterations):
            soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight

            prediction = self.sens_reduce(pred, sensitivity_maps)
            prediction = self.model(prediction.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + prediction
            prediction = self.sens_expand(prediction.unsqueeze(self.coil_dim), sensitivity_maps)

            if not self.no_dc:
                prediction = pred - soft_dc - prediction

            pred = prediction

            preds.append(prediction)

        return preds
