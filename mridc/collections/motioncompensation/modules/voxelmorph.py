# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/wustl-cig/DeCoLearn/blob/main/decolearn/torch_util/module.py and
# https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/networks.py

from abc import ABC
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mridc.collections.reconstruction.models.unet_base.unet_block import ConvBlock, NormUnet


class VoxelMorph(nn.Module, ABC):
    """

    Implementation of the VoxelMorph, as presented in Gan, W., et al. & Balakrishnan, G. et al. (2019).

    References
    ----------

    ..

        Gan, W., Sun, Y., Eldeniz, C., Liu, J., An, H., & Kamilov, U. S. (2022). Deformation-Compensated Learning for
        Image Reconstruction without Ground Truth. IEEE Transactions on Medical Imaging, 0062(c), 1–14.
        https://doi.org/10.1109/TMI.2022.3163018

        Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. v. (2019). VoxelMorph: A Learning Framework
        for Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, 38(8), 1788–1800.
        https://doi.org/10.1109/TMI.2019.2897538

    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        bidirectional: bool = False,
        int_downsize: int = 2,
        int_steps: int = 7,
        shape: Optional[Sequence[int]] = None,
        mode: str = "bilinear",
    ):
        """
        Initializes the model.

        Parameters
        ----------
        chans: Number of channels in the input k-space data.
            int
        num_pools: Number of U-Net downsampling/upsampling operations.
            int
        in_chans: Number of channels in the input data.
            int
        out_chans: Number of channels in the output data.
            int
        drop_prob: Dropout probability.
            float
        padding_size: Size of the zero-padding.
            int
        normalize: Whether to normalize the input data.
            bool
        conv_kernels: Kernel size of the convolutional layers.
            list
        conv_dilations: Dilation of the convolutional layers.
            list
        conv_bias: Whether to use bias in the convolutional layers.
            bool
        conv_nonlinearity: Nonlinearity of the convolutional layers.
            str
        conv_dim: Dimension of the convolutional layers.
            int
        shape: Shape of the input data.
            Sequence[int]
        mode: Mode of the interpolation.
            str
        """
        super().__init__()

        ndims = len(shape)  # type: ignore
        if ndims not in [1, 2, 3]:
            raise ValueError(f"Input must be 1D, 2D, or 3D. Got {ndims}D.")

        self.model = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            padding_size=padding_size,
            normalize=normalize,
        )
        self.flow = ConvBlock(in_chans=out_chans, out_chans=out_chans, drop_prob=drop_prob)

        self.mode = mode

        if int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
            self.full_resize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.resize = None  # type: ignore
            self.full_resize = None  # type: ignore

        # configure bidirectional flow
        self.bidirectional = bidirectional

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in shape]  # type: ignore
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(shape)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        registration: bool = False,
    ) -> Tuple[Any, Optional[Any], Any]:
        """
        Forward pass of the model.

        Parameters
        ----------
        source: Source k-space data.
            torch.Tensor
        target: Target k-space data.
            torch.Tensor
        registration: Whether to perform registration.
            bool

        Returns
        -------
        Motion corrected data.
        """
        input = torch.cat([torch.view_as_real(source), torch.view_as_real(target)], dim=-1).permute(0, 3, 1, 2)
        output = self.model(input)
        flow = self.flow(output)

        pos_flow = flow.clone()
        if self.resize is not None:
            pos_flow = self.resize(pos_flow)  # type: ignore

        # negative flow for bidirectional model
        neg_flow = -pos_flow if self.bidirectional else None  # type: ignore

        # integrate to produce diffeomorphic warp
        if self.integrate is not None:
            pos_flow = self.integrate(pos_flow)  # type: ignore
            neg_flow = self.integrate(neg_flow) if self.bidirectional else None  # type: ignore

            # resize to original size
            if self.full_resize is not None:
                pos_flow = self.full_resize(pos_flow)  # type: ignore
                neg_flow = self.full_resize(neg_flow) if self.bidirectional else None  # type: ignore

        # wrap image with flow field
        y_source = self.transformer(source, pos_flow)  # type: ignore
        y_target = self.transformer(target, neg_flow) if self.bidirectional else None  # type: ignore

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, flow) if self.bidirectional else (y_source, None, flow)
        else:
            return (y_source, None, pos_flow)


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer
    """

    def __init__(
        self,
        shape: Optional[Sequence[int]],
        mode: str = "bilinear",
    ):
        """
        Initialize the spatial transformer.

        Parameters
        ----------
        shape: Shape of the input image
            Sequence[int]
        mode: Interpolation mode
            str
        """
        super().__init__()
        self.mode = mode

        grid = torch.stack(torch.meshgrid([torch.arange(0, s) for s in shape])).unsqueeze(0)  # type: ignore
        self.grid = torch.cat([grid, grid], dim=1)

    def forward(self, src, flow):
        """
        Forward pass of the model.

        Parameters
        ----------
        src: Source image
            torch.Tensor
        flow: Flow field
            torch.Tensor
        """
        new_locs = self.grid.to(flow) + flow
        shape = flow.shape[2:]
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if src.dtype in (torch.complex64, torch.complex128):
            src = torch.view_as_real(src).permute(0, 3, 1, 2)

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)  # .to(flow)


class ResizeTransform(nn.Module):
    """Resize a transform to a given scale."""

    def __init__(self, scale, ndims):
        """
        Initialize the ResizeTransform.

        Parameters
        ----------
        scale: Scale of the transform
            float
        ndims: Number of dimensions of the transform
            int
        """
        super().__init__()

        self.scale = 1.0 / scale
        self.mode = "linear"

        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x):
        """
        Forward pass of the ResizeTransform.

        Parameters
        ----------
        x: Input to the ResizeTransform
            torch.Tensor

        Returns
        -------
        Resized transform
            torch.Tensor
        """
        if self.scale < 1:
            x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
            x = x * self.scale
        elif self.scale > 1:
            x = x * self.scale
            x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        return x


class VecInt(nn.Module):
    """
    Integrator for diffeomorphic warp
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        if nsteps < 0:
            raise ValueError(f"Steps must be positive. Got {nsteps}.")
        self.nsteps = nsteps
        self.scale = 1.0 / (2**nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        """
        Forward pass of the VecInt.

        Parameters
        ----------
        vec: Vector to be integrated
            torch.Tensor

        Returns
        -------
        Integrated vector
            torch.Tensor
        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
