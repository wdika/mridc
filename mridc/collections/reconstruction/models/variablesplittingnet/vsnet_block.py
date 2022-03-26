# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Union

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class DataConsistencyLayer(torch.nn.Module):
    """Data consistency layer for the VSNet.

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

    .. [1] Duan, J. et al. (2019) ‘Vs-net: Variable splitting network for accelerated parallel MRI reconstruction’,
    Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes
    in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.
    """

    def __init__(
        self,
        denoiser_block: torch.nn.ModuleList,
        data_consistency_block: torch.nn.ModuleList,
        weighted_average_block: torch.nn.ModuleList,
        num_cascades: int = 8,
        fft_type: str = "orthogonal",
    ):
        """
        Initialize the model block.

        Args:
            denoiser_block: Model to apply denoising.
            data_consistency_block: Model to apply data consistency.
            weighted_average_block: Model to apply weighted average.
            num_cascades: Number of cascades.
            fft_type: Type of FFT to use.
        """
        super().__init__()

        self.denoiser_block = denoiser_block
        self.data_consistency_block = data_consistency_block
        self.weighted_average_block = weighted_average_block
        self.num_cascades = num_cascades
        self.fft_type = fft_type

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Expand the sensitivity maps to the same size as the input.

        Args:
            x: Input data.
            sens_maps: Sensitivity maps.

        Returns:
            SENSE reconstruction expanded to the same size as the input.
        """
        return fft2c(complex_mul(x, sens_maps), fft_type=self.fft_type)

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Reduce the sensitivity maps to the same size as the input.

        Args:
            x: Input data.
            sens_maps: Sensitivity maps.

        Returns:
            SENSE reconstruction reduced to the same size as the input.
        """
        x = ifft2c(x, fft_type=self.fft_type)
        return complex_mul(x, complex_conj(sens_maps)).sum(1)

    def forward(
        self,
        kspace: torch.Tensor,
        sens_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[Union[torch.Tensor, Any]]:
        """
        Forward pass of the model.

        Args:
            kspace: Reference k-space data.
            sens_maps: Sensitivity maps.
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
