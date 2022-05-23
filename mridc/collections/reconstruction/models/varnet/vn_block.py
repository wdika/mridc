# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class VarNetBlock(torch.nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(self, model: torch.nn.Module, fft_type: str = "orthogonal", no_dc: bool = False):
        """
        Initialize the model block.

        Parameters
        ----------
        model: Model to apply soft data consistency.
        fft_type: Type of FFT to use.
        no_dc: Whether to remove the DC component.
        """
        super().__init__()

        self.model = model
        self.fft_type = fft_type
        self.no_dc = no_dc
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

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
        return fft2c(complex_mul(x, sens_maps), fft_type=self.fft_type)

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
        x = ifft2c(x, fft_type=self.fft_type)
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        sens_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
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
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred)
        soft_dc = torch.where(mask.bool(), pred - ref_kspace, zero) * self.dc_weight

        eta = self.sens_reduce(pred, sens_maps)
        eta = self.model(eta)
        eta = self.sens_expand(eta, sens_maps)

        if not self.no_dc:
            eta = pred - soft_dc - eta

        return eta
