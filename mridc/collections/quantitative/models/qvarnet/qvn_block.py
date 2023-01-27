# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

from typing import List, Optional, Tuple

import torch

import mridc.collections.common.parts.utils as utils
from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.quantitative.models.base import SignalForwardModel


class qVarNetBlock(torch.nn.Module):
    """
    Implementation of the quantitative End-to-end Variational Network (qVN), as presented in [1].

    References
    ----------
    .. [1] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW. A unified
        model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent inference
        machine. NeuroImage. 2022 Dec 1;264:119680.

    Parameters
    ----------
    model : torch.nn.Module
        Model to apply soft data consistency.
    fft_centered : bool, optional
        Whether to center the fft. Default is ``False``.
    fft_normalization : str, optional
        The normalization of the fft. Default is ``backward``.
    spatial_dims : tuple, optional
        The spatial dimensions of the data. Default is ``None``.
    coil_dim : int, optional
        The dimension of the coils. Default is ``1``.
    no_dc : bool, optional
        Whether to not apply the DC component. Default is ``False``.
    linear_forward_model : torch.nn.Module, optional
        Linear forward model. Default is ``None``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        no_dc: bool = False,
        linear_forward_model=None,
    ):
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
        return fft2(
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
        x = ifft2(x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(dim=self.coil_dim)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        sensitivity_maps: torch.Tensor,
        sampling_mask: torch.Tensor,
        prediction: torch.Tensor = None,
        gamma: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        masked_kspace : torch.Tensor
            Subsampled k-space of shape [batch_size, n_coils, n_x, n_y, 2].
        R2star_map_init : torch.Tensor
            Initial R2* map of shape [batch_size, n_echoes, n_coils, n_x, n_y].
        S0_map_init : torch.Tensor
            Initial S0 map of shape [batch_size, n_echoes, n_coils, n_x, n_y].
        B0_map_init : torch.Tensor
            Initial B0 map of shape [batch_size, n_echoes, n_coils, n_x, n_y].
        phi_map_init : torch.Tensor
            Initial phi map of shape [batch_size, n_echoes, n_coils, n_x, n_y].
        TEs : List
            Echo times.
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, n_coils, n_x, n_y].
        prediction : torch.Tensor, optional
            Initial prediction of the quantitative maps. If None, it will be initialized with the initial maps.
            Default is ``None``.
        gamma : torch.Tensor, optional
            Scaling normalization factor of shape [batch_size, 1, 1, 1, 1].

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape [batch_size, n_coils, n_x, n_y, 2].
        """
        if prediction is None:
            prediction = torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], dim=1)

        R2star_map_init = (R2star_map_init * gamma[0]).unsqueeze(0)  # type: ignore
        S0_map_init = (S0_map_init * gamma[1]).unsqueeze(0)  # type: ignore
        B0_map_init = (B0_map_init * gamma[2]).unsqueeze(0)  # type: ignore
        phi_map_init = (phi_map_init * gamma[3]).unsqueeze(0)  # type: ignore

        init_pred = self.linear_forward_model(R2star_map_init, S0_map_init, B0_map_init, phi_map_init, TEs)
        pred_kspace = self.sens_expand(init_pred, sensitivity_maps.unsqueeze(self.coil_dim - 1))
        soft_dc = (pred_kspace - masked_kspace) * sampling_mask * self.dc_weight
        init_pred = self.sens_reduce(soft_dc, sensitivity_maps.unsqueeze(self.coil_dim - 1)).to(masked_kspace)

        prediction = torch.view_as_real(prediction + torch.view_as_complex(self.model(init_pred.to(masked_kspace))))
        prediction_tmp = prediction[:, 0, ...]
        prediction_tmp[prediction_tmp < 0] = 0
        prediction[:, 0, ...] = prediction_tmp

        return prediction
