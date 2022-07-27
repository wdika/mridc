# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

from typing import List, Optional, Tuple

import torch
from matplotlib import pyplot as plt

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import coil_combination, complex_conj, complex_mul
from mridc.collections.quantitative.models.qrim.utils import SignalForwardModel


class qVarNetBlock(torch.nn.Module):
    """
    Implementation of the quantitative End-to-end Variational Network (qVN), as presented in Zhang, C. et al.

    References
    ----------

    ..

        Zhang, C. et al. (2022) ‘A unified model for reconstruction and R2 mapping of accelerated 7T data using \
        quantitative Recurrent Inference Machine’. In review.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
        linear_forward_model=None,
    ):
        """
        Initialize the model block.

        Parameters
        ----------
        model: Model to apply soft data consistency.
        fft_centered: Whether to center the fft.
        fft_normalization: The normalization of the fft.
        spatial_dims: The spatial dimensions of the data.
        coil_dim: The dimension of the coil dimension.
        no_dc: Whether to remove the DC component.
        """
        super().__init__()

        self.linear_forward_model = (
            SignalForwardModel(sequence="MEGRE") if linear_forward_model is None else linear_forward_model
        )

        self.model = model
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
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=self.coil_dim)

    def forward(
        self,
        prediction: torch.Tensor,
        masked_kspace: torch.Tensor,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        sensitivity_maps: torch.Tensor,
        sampling_mask: torch.Tensor,
        gamma: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        prediction: Initial prediction of the subsampled k-space.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        masked_kspace: Data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        R2star_map_init: Initial R2* map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        S0_map_init: Initial S0 map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        B0_map_init: Initial B0 map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        phi_map_init: Initial phi map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        TEs: List of echo times.
            List of int, shape [batch_size, n_echoes]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sampling_mask: Mask of the sampling.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        gamma: Scaling normalization factor.
            torch.Tensor, shape [batch_size, 1, 1, 1, 1]

        Returns
        -------
        Reconstructed image.
        """
        init_eta = torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], dim=1)

        R2star_map_init = (R2star_map_init * gamma[0]).unsqueeze(0)  # type: ignore
        S0_map_init = (S0_map_init * gamma[1]).unsqueeze(0)  # type: ignore
        B0_map_init = (B0_map_init * gamma[2]).unsqueeze(0)  # type: ignore
        phi_map_init = (phi_map_init * gamma[3]).unsqueeze(0)  # type: ignore

        init_pred = self.linear_forward_model(R2star_map_init, S0_map_init, B0_map_init, phi_map_init, TEs)
        pred_kspace = self.sens_expand(init_pred, sensitivity_maps.unsqueeze(self.coil_dim - 1))
        soft_dc = (pred_kspace - masked_kspace) * sampling_mask * self.dc_weight
        init_pred = self.sens_reduce(soft_dc, sensitivity_maps.unsqueeze(self.coil_dim - 1)).to(masked_kspace)

        eta = torch.view_as_real(init_eta + torch.view_as_complex(self.model(init_pred.to(masked_kspace))))
        eta_tmp = eta[:, 0, ...]
        eta_tmp[eta_tmp < 0] = 0
        eta[:, 0, ...] = eta_tmp

        return eta
