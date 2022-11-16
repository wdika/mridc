# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional, Tuple

import torch
from torch import nn

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils


class SERANetDC(nn.Module):
    """Data consistency block."""

    def __init__(
        self,
        fft_centered: bool,
        fft_normalization: str,
        spatial_dims: Tuple[int, ...],
    ):
        """Initialize the SERANet data consistency block."""
        super().__init__()
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, prediction, prev_prediction, reference_kspace, mask):
        """Forward pass."""
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
    Reconstruction Model block for End-to-End Recurrent Attention Network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(
        self,
        num_reconstruction_blocks: int,
        reconstruction_model: torch.nn.Module,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        coil_combination_method: str = "SENSE",
    ):
        """
        Initialize the model block.

        Parameters
        ----------
        num_reconstruction_blocks: Number of reconstruction blocks.
        reconstruction_model: Reconstruction model.
        fft_centered: Whether to center the fft.
        fft_normalization: The normalization of the fft.
        spatial_dims: The spatial dimensions of the data.
        coil_dim: The dimension of the coil dimension.
        coil_combination_method: The coil combination method.
        """
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
        pred: torch.Tensor,
        ref_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred: Input data.
        ref_kspace: Reference k-space data.
        sensitivity_maps: Sensitivity maps.
        mask: Mask to apply to the data.

        Returns
        -------
        Reconstructed image.
        """
        pred_reconstruction = []
        prev_reconstruction = ref_kspace.clone()
        for recon_block, dc_block in zip(self.reconstruction_module, self.reconstruction_module_dc):
            reconstruction = self.step(recon_block, pred, ref_kspace, sensitivity_maps, mask)
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
        Parameters
        ----------
        block: The block to apply.
        pred: Input data.
        ref_kspace: Reference k-space data.
        sensitivity_maps: Sensitivity maps.
        mask: Mask to apply to the data.

        Returns
        -------
        Reconstructed image.
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
                utils.coil_combination(
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
    RecurrentModel block for End-to-End Recurrent Attention Network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(
        self,
        num_iterations: int,
        attention_model: torch.nn.Module,
        unet_model: torch.nn.Module,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
    ):
        """
        Initialize the model block.

        Parameters
        ----------
        num_iterations: Number of reconstruction blocks.
        attention_model: Attention model.
        unet_model: UNet model.
        fft_centered: Whether to center the fft.
        fft_normalization: The normalization of the fft.
        spatial_dims: The spatial dimensions of the data.
        coil_dim: The dimension of the coil dimension.
        """
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
        Parameters
        ----------
        pred_reconstruction: Input data.
        pred_segmentation: Input segmentation.
        ref_kspace: Reference k-space data.
        sensitivity_maps: Sensitivity maps.
        mask: Mask to apply to the data.

        Returns
        -------
        Reconstructed image.
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
