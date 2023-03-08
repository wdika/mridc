# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional, Tuple

import torch

from mridc.collections.common.parts import fft, utils


class DataConsistencyLayer(torch.nn.Module):
    """
    Data consistency layer for the VSNet.

    This layer is used to ensure that the output of the VSNet is the same as the input.
    """

    def __init__(self):
        super().__init__()
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, pred_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the data consistency layer."""
        return ((1 - mask) * pred_kspace + mask * ref_kspace) * self.dc_weight


class WeightedAverageTerm(torch.nn.Module):
    """Weighted average term for the VSNet."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, Sx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the weighted average term."""
        return self.param * x + (1 - self.param) * Sx


class VSNetBlock(torch.nn.Module):
    """
    Model block for the Variable-Splitting Network inspired by [1].

    References
    ----------
    .. [1] Duan, J. et al. (2019) ‘Vs-net: Variable splitting network for accelerated parallel MRI reconstruction’,
        Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture
        Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.

    Parameters
    ----------
    denoiser_block : torch.nn.ModuleList
        Model to apply denoising.
    data_consistency_block : torch.nn.ModuleList
        Model to apply data consistency.
    weighted_average_block : torch.nn.ModuleList
        Model to apply weighted average.
    num_cascades : int, optional
        Number of cascades. Default is ``8``.
    fft_centered : bool, optional
        Whether to center the fft. Default is ``False``.
    fft_normalization : str, optional
        The normalization of the fft. Default is ``"backward"``.
    spatial_dims : tuple, optional
        The spatial dimensions of the data. Default is ``None``.
    coil_dim : int, optional
        The dimension of the coil. Default is ``1``.
    """

    def __init__(  # noqa: W0221
        self,
        denoiser_block: torch.nn.ModuleList,
        data_consistency_block: torch.nn.ModuleList,
        weighted_average_block: torch.nn.ModuleList,
        num_cascades: int = 8,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
    ):
        super().__init__()

        self.denoiser_block = denoiser_block
        self.data_consistency_block = data_consistency_block
        self.weighted_average_block = weighted_average_block
        self.num_cascades = num_cascades
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

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
        kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model block.

        Parameters
        ----------
        kspace : torch.Tensor
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
        for idx in range(self.num_cascades):
            pred = self.sens_reduce(kspace, sensitivity_maps)
            pred = self.denoiser_block[idx](pred.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            pred = self.sens_expand(pred, sensitivity_maps)
            sx = self.data_consistency_block[idx](pred, kspace, mask)
            sx = self.sens_reduce(sx, sensitivity_maps)
            kspace = self.weighted_average_block[idx](kspace + pred, sx)
        return kspace
