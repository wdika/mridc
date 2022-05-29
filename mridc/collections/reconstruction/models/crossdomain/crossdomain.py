# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/crossdomain/crossdomain.py
# Copyright (c) DIRECT Contributors
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class CrossDomainNetwork(nn.Module):
    """This performs optimisation in both, k-space ("K") and image ("I") domains according to domain_sequence."""

    def __init__(
        self,
        image_model_list: nn.Module,
        kspace_model_list: Optional[Union[nn.Module, None]] = None,
        domain_sequence: str = "KIKI",
        image_buffer_size: int = 1,
        kspace_buffer_size: int = 1,
        normalize_image: bool = False,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        **kwargs,
    ):
        """
        Inits CrossDomainNetwork.

        Parameters
        ----------
        image_model_list: Image domain model list.
            torch.nn.Module
        kspace_model_list: K-space domain model list. If set to None, a correction step is applied.
            torch.nn.Module, Default: None.
        domain_sequence: Domain sequence containing only "K" (k-space domain) and/or "I" (image domain).
            str, Default: "KIKI".
        image_buffer_size: Image buffer size.
            int, Default: 1.
        kspace_buffer_size: K-space buffer size.
            int, Default: 1.
        normalize_image: If True, input is normalized.
            bool, Default: False.
        fft_centered: If True, FFT is centered.
            bool, Default: True.
        fft_normalization: FFT normalization.
            str, Default: "ortho".
        spatial_dims: Spatial dimensions.
            Tuple[int, int], Default: None.
        coil_dim: Coil dimension.
            int, Default: 1.
        kwargs:Keyword Arguments.
            dict
        """
        super().__init__()

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

        domain_sequence = list(domain_sequence.strip())  # type: ignore
        if not set(domain_sequence).issubset({"K", "I"}):
            raise ValueError(f"Invalid domain sequence. Got {domain_sequence}. Should only contain 'K' and 'I'.")

        if kspace_model_list is not None and len(kspace_model_list) != domain_sequence.count("K"):
            raise ValueError("K-space domain steps do not match k-space model list length.")

        if len(image_model_list) != domain_sequence.count("I"):
            raise ValueError("Image domain steps do not match image model list length.")

        self.domain_sequence = domain_sequence

        self.kspace_model_list = kspace_model_list
        self.kspace_buffer_size = kspace_buffer_size

        self.image_model_list = image_model_list
        self.image_buffer_size = image_buffer_size

    def kspace_correction(self, block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map, masked_kspace):
        """Performs k-space correction."""
        forward_buffer = [
            self._forward_operator(image.clone(), sampling_mask, sensitivity_map)
            for image in torch.split(image_buffer, 2, -1)
        ]
        forward_buffer = torch.cat(forward_buffer, -1)

        kspace_buffer = torch.cat([kspace_buffer, forward_buffer, masked_kspace], -1)

        if self.kspace_model_list is not None:
            kspace_buffer = self.kspace_model_list[block_idx](kspace_buffer.permute(0, 1, 4, 2, 3)).permute(
                0, 1, 3, 4, 2
            )
        else:
            kspace_buffer = kspace_buffer[..., :2] - kspace_buffer[..., 2:4]

        return kspace_buffer

    def image_correction(self, block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map):
        """Performs image correction."""
        backward_buffer = [
            self._backward_operator(kspace.clone(), sampling_mask, sensitivity_map)
            for kspace in torch.split(kspace_buffer, 2, -1)
        ]
        backward_buffer = torch.cat(backward_buffer, -1)

        image_buffer = torch.cat([image_buffer, backward_buffer], -1).permute(0, 3, 1, 2)
        image_buffer = self.image_model_list[block_idx](image_buffer).permute(0, 2, 3, 1)

        return image_buffer

    def _forward_operator(self, image, sampling_mask, sensitivity_map):
        """Forward operator."""
        return torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=image.dtype).to(image.device),
            fft2(
                complex_mul(image.unsqueeze(self.coil_dim), sensitivity_map),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ).type(image.type()),
        )

    def _backward_operator(self, kspace, sampling_mask, sensitivity_map):
        """Backward operator."""
        kspace = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype).to(kspace.device), kspace)
        return (
            complex_mul(
                ifft2(
                    kspace.float(),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                complex_conj(sensitivity_map),
            )
            .sum(self.coil_dim)
            .type(kspace.type())
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the forward pass of CrossDomainNetwork.

        Parameters
        ----------
        masked_kspace: Subsampled k-space data.
            torch.tenor, shape [batch_size, n_coil, height, width, 2]
        sensitivity_map: Sensitivity map.
            torch.tenor, shape [batch_size, n_coil, height, width, 2]
        sampling_mask: Sampling mask.
            torch.tenor, shape [batch_size, 1, height, width, 1]

        Returns
        -------
        Output image.
            torch.tenor, shape [batch_size, height, width, 2]
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)

        image_buffer = torch.cat([input_image] * self.image_buffer_size, -1).to(masked_kspace.device)
        kspace_buffer = torch.cat([masked_kspace] * self.kspace_buffer_size, -1).to(masked_kspace.device)

        kspace_block_idx, image_block_idx = 0, 0
        for block_domain in self.domain_sequence:
            if block_domain == "K":
                kspace_buffer = self.kspace_correction(
                    kspace_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map, masked_kspace
                )
                kspace_block_idx += 1
            else:
                image_buffer = self.image_correction(
                    image_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map
                )
                image_block_idx += 1

        return image_buffer[..., :2]
