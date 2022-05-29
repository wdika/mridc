# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Optional, Tuple, Union

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class DataConsistencyLayer(torch.nn.Module):
    """
    Data consistency layer for the VSNet.
    This layer is used to ensure that the output of the VSNet is the same as the input.
    """

    def __init__(self):
        """Initializes the data consistency layer."""
        super().__init__()
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, pred_kspace, ref_kspace, mask):
        """Forward pass of the data consistency layer."""
        return ((1 - mask) * pred_kspace + mask * ref_kspace) * self.dc_weight


class WeightedAverageTerm(torch.nn.Module):
    """Weighted average term for the VSNet."""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, Sx):
        return self.param * x + (1 - self.param) * Sx


class VSNetBlock(torch.nn.Module):
    """
    Model block for the Variable-Splitting Network inspired by [1]_.

    References
    ----------
    .. [1] Duan, J. et al. (2019) ‘Vs-net: Variable splitting network for accelerated parallel MRI reconstruction’, Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.
    """

    def __init__(
        self,
        denoiser_block: torch.nn.ModuleList,
        data_consistency_block: torch.nn.ModuleList,
        weighted_average_block: torch.nn.ModuleList,
        num_cascades: int = 8,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
    ):
        """

        Parameters
        ----------
        denoiser_block: Model to apply denoising.
        data_consistency_block: Model to apply data consistency.
        weighted_average_block: Model to apply weighted average.
        num_cascades: Number of cascades.
        fft_centered: Whether to center the fft.
        fft_normalization: The normalization of the fft.
        spatial_dims: The spatial dimensions of the data.
        coil_dim: The dimension of the coil.
        """
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
        Expand the sensitivity maps to the same size as the input.

        Parameters
        ----------
        x: Input data.
        sens_maps: Coil Sensitivity maps.

        Returns
        -------
        SENSE reconstruction expanded to the same size as the input sens_maps.
        """
        return fft2(
            complex_mul(x, sens_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Reduce the sensitivity maps.

        Parameters
        ----------
        x: Input data.
        sens_maps: Coil Sensitivity maps.

        Returns
        -------
        SENSE coil-combined reconstruction.
        """
        x = ifft2(x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims)
        return complex_mul(x, complex_conj(sens_maps)).sum(self.coil_dim)

    def forward(
        self,
        kspace: torch.Tensor,
        sens_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Union[torch.Tensor, Any]]:
        """

        Parameters
        ----------
        kspace: Reference k-space data.
        sens_maps: Coil sensitivity maps.
        mask: Mask to apply to the data.

        Returns
        -------
        Reconstructed image.
        """
        for idx in range(self.num_cascades):
            pred = self.sens_reduce(kspace, sens_maps)
            pred = self.denoiser_block[idx](pred.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            pred = self.sens_expand(pred, sens_maps)
            sx = self.data_consistency_block[idx](pred, kspace, mask)
            sx = self.sens_reduce(sx, sens_maps)
            kspace = self.weighted_average_block[idx](kspace + pred, sx)
        return kspace
