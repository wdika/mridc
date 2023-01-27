# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional, Tuple

import torch
from torch import nn

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils


class SERANetDC(nn.Module):
    """
    SERANet Data consistency block, as presented in [1].

    References
    ----------
    .. [1] Huang, Q., Chen, X., Metaxas, D., Nadar, M.S. (2019). Brain Segmentation from k-Space with End-to-End
         Recurrent Attention Network. In: , et al. Medical Image Computing and Computer Assisted Intervention –
         MICCAI 2019. Lecture Notes in Computer Science(), vol 11766. Springer, Cham.
         https://doi.org/10.1007/978-3-030-32248-9_31

    Parameters
    ----------
    fft_centered: bool
        Whether to center the fft.
    fft_normalization: str
        Normalization to apply to the fft.
    spatial_dims: Tuple[int, ...]
        Spatial dimensions.
    """

    def __init__(
        self,
        fft_centered: bool,
        fft_normalization: str,
        spatial_dims: Tuple[int, ...],
    ):
        super().__init__()
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        prediction: torch.Tensor,
        prev_prediction: torch.Tensor,
        reference_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the DC block.

        Parameters
        ----------
        prediction_kspace : torch.Tensor
            Prediction k-space. Shape: (batch, channels, height, width, complex)
        prev_prediction_kspace : torch.Tensor
            Previous prediction k-space. Shape: (batch, channels, height, width, complex)
        reference_kspace : torch.Tensor
            Reference k-space. Shape: (batch, channels, height, width, complex)
        mask : torch.Tensor
            Subsampling mask. Shape: (batch, channels, height, width, 1)

        Returns
        -------
        torch.Tensor
            Data consistency k-space. Shape: (batch, channels, height, width, complex)
        """
        prediction = fft.fft2(prediction.float(), self.fft_centered, self.fft_normalization, self.spatial_dims).to(
            prediction
        )
        if prediction.dim() < reference_kspace.dim():
            prediction = prediction.unsqueeze(1)
        zero = torch.zeros(1, 1, 1, 1, 1).to(prediction)
        soft_dc = torch.where(mask.bool(), prediction - reference_kspace, zero) * self.dc_weight
        prediction = prev_prediction - soft_dc - prediction
        return fft.ifft2(prediction, self.fft_centered, self.fft_normalization, self.spatial_dims)


class SERANetReconstructionBlock(torch.nn.Module):
    """
    Reconstruction Model block for End-to-End Recurrent Attention Network, as presented in [1].

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.

    References
    ----------
    .. [1] Pramanik A, Wu X, Jacob M. Joint calibrationless reconstruction and segmentation of parallel MRI. arXiv
        preprint arXiv:2105.09220. 2021 May 19.

    Parameters
    ----------
    num_reconstruction_blocks : int
        Number of reconstruction blocks.
    reconstruction_model : torch.nn.Module
        Reconstruction model.
    fft_centered : bool, optional
        Whether to center the fft. Default is ``False``.
    fft_normalization : str, optional
        The normalization of the fft. Default is ``"backward"``.
    spatial_dims : Tuple[int, int], optional
        The spatial dimensions of the data. Default is ``None``.
    coil_dim : int, optional
        The dimension of the coil dimension. Default is ``1``.
    coil_combination_method : str, optional
        The coil combination method. Default is ``"SENSE"``.
    """

    def __init__(
        self,
        num_reconstruction_blocks: int,
        reconstruction_model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        coil_combination_method: str = "SENSE",
    ):
        super().__init__()
        self.reconstruction_module = torch.nn.ModuleList(
            [reconstruction_model for _ in range(num_reconstruction_blocks)]
        )
        self.model_name = self.reconstruction_module[0].__class__.__name__.lower()
        if self.model_name == "modulelist":
            self.model_name = self.reconstruction_module[0][0].__class__.__name__.lower()

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
        self.coil_combination_method = coil_combination_method

        self.reconstruction_module_dc = torch.nn.ModuleList(
            [
                SERANetDC(
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,  # type: ignore
                )
                for _ in range(num_reconstruction_blocks)
            ]
        )

    def forward(
        self,
        prediction: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the reconstruction block.

        Parameters
        ----------
        prediction : torch.Tensor
            Prediction. Shape: [batch, channels, height, width, 2]
        ref_kspace : torch.Tensor
            Reference k-space. Shape: [batch, channels, height, width, 2]
        sensitivity_maps : torch.Tensor
            Sensitivity maps. Shape: [batch, coils, height, width, 2]
        mask : torch.Tensor
            Subsampling mask. Shape: [batch, 1, height, width, 1]

        Returns
        -------
        torch.Tensor
            Reconstruction. Shape: [batch, height, width, 2]
        """
        pred_reconstruction = []
        prev_reconstruction = ref_kspace.clone()
        for recon_block, dc_block in zip(self.reconstruction_module, self.reconstruction_module_dc):
            reconstruction = self.step(recon_block, prediction, ref_kspace, sensitivity_maps, mask)
            reconstruction = dc_block(reconstruction, prev_reconstruction, ref_kspace, mask)
            prev_reconstruction = reconstruction
            pred_reconstruction.append(reconstruction)
        return pred_reconstruction

    def step(
        self,
        block: torch.nn.Module,
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step of the reconstruction block.

        Parameters
        ----------
        block : torch.nn.Module
            The block to apply.
        pred : torch.Tensor
            Prediction. Shape: [batch, height, width, 2]
        ref_kspace : torch.Tensor
            Reference k-space. Shape: [batch, channels, height, width, 2]
        sensitivity_maps : torch.Tensor
            Sensitivity maps. Shape: [batch, coils, height, width, 2]
        mask : torch.Tensor
            Subsampling mask. Shape: [batch, 1, height, width, 1]

        Returns
        -------
        torch.Tensor
            Reconstruction. Shape: [batch, height, width, 2]
        """
        if self.model_name == "unet":
            reconstruction = block(pred).permute(0, 2, 3, 1)
            reconstruction = torch.view_as_real(
                reconstruction[..., 0].float() + 1j * reconstruction[..., 1].float()
            ).to(reconstruction)
        elif "cascadenet" in self.model_name:
            reconstruction = ref_kspace.clone()
            for cascade in block:
                reconstruction = cascade(reconstruction, ref_kspace, sensitivity_maps, mask)
            reconstruction = torch.view_as_complex(
                utils.coil_combination_method(
                    fft.ifft2(
                        reconstruction,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_maps,
                    method=self.coil_combination_method,
                    dim=self.coil_dim,
                )
            )
        else:
            reconstruction = pred.clone()

        return reconstruction


class SERANetRecurrentBlock(torch.nn.Module):
    """
    RecurrentModel block for End-to-End Recurrent Attention Network, as presented in [1].

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.

    References
    ----------
    .. [1] Pramanik A, Wu X, Jacob M. Joint calibrationless reconstruction and segmentation of parallel MRI. arXiv
        preprint arXiv:2105.09220. 2021 May 19.

    Parameters
    ----------
    num_iterations : int
        Number of iterations for the recurrent block.
    attention_model : torch.nn.Module
        Attention model.
    unet_model : torch.nn.Module
        Unet model.
    fft_centered : bool, optional
        Whether to center the fft. Default is ``False``.
    fft_normalization : str, optional
        The normalization of the fft. Default is ``"backward"``.
    spatial_dims : Tuple[int, int], optional
        The spatial dimensions of the data. Default is ``None``.
    coil_dim : int, optional
        The dimension of the coil dimension. Default is ``1``.
    """

    def __init__(
        self,
        num_iterations: int,
        attention_model: torch.nn.Module,
        unet_model: torch.nn.Module,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
    ):
        super().__init__()
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

        self.num_iterations = num_iterations
        self.recurrent_module_unet = unet_model
        self.recurrent_module_attention = attention_model
        self.recurrent_module_dc = SERANetDC(
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,  # type: ignore
        )

    def forward(
        self,
        pred_reconstruction: torch.Tensor,
        pred_segmentation: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the recurrent block.

        Parameters
        ----------
        pred_reconstruction : torch.Tensor
            Prediction. Shape: [batch, height, width, 2]
        pred_segmentation : torch.Tensor
            Prediction. Shape: [batch, num_classes, height, width]
        ref_kspace : torch.Tensor
            Reference k-space. Shape: [batch, channels, height, width, 2]
        sensitivity_maps : torch.Tensor
            Sensitivity maps. Shape: [batch, coils, height, width, 2]
        mask : torch.Tensor
            Subsampling mask. Shape: [batch, 1, height, width, 1]

        Returns
        -------
        torch.Tensor
            Reconstruction. Shape: [batch, height, width, 2]
        """
        attention_map = pred_segmentation.clone()  # TODO: remove this
        prev_prediction = ref_kspace.clone()
        for _ in range(self.num_iterations):
            attention_map = self.chan_complex_to_last_dim(
                self.recurrent_module_attention(
                    self.complex_to_chan_dim(pred_reconstruction), attention_map * pred_segmentation
                )
            )
            attention_map = self.recurrent_module_dc(attention_map, prev_prediction, ref_kspace, mask)
            prev_prediction = attention_map
            attention_map = self.recurrent_module_unet(self.complex_to_chan_dim(attention_map))
        return attention_map

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
