# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional, Tuple

import torch

from mridc.collections.common.parts import fft, utils


class VarNetBlock(torch.nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.

    Parameters
    ----------
    model : torch.nn.Module
        Model to apply soft data consistency.
    fft_centered : bool, optional
        Whether to center the FFT. Default is ``False``.
    fft_normalization : str, optional
        Whether to normalize the FFT. Default is ``"backward"``.
    spatial_dims : Tuple[int, int], optional
        Spatial dimensions of the input. Default is ``None``.
    coil_dim : int, optional
        Coil dimension. Default is ``1``.
    no_dc : bool, optional
        Flag to disable the soft data consistency. Default is ``False``.
    """

    def __init__(  # noqa: W0221
        self,
        model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
    ):
        super().__init__()

        self.model = model
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
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(dim=self.coil_dim, keepdim=True)

    def forward(
        self,
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model block.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
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
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred)
        soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight

        prediction = self.sens_reduce(pred, sensitivity_maps)
        prediction = self.model(prediction)
        prediction = self.sens_expand(prediction, sensitivity_maps)

        if not self.no_dc:
            prediction = pred - soft_dc - prediction

        return prediction
