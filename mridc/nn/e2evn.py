# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mridc import complex_mul, rss, complex_abs, complex_conj
from mridc import fft2c, ifft2c
from mridc import rss_complex
from mridc.data import transforms
from .unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the input before the U-Net.
    This keeps the values more numerically stable during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
    ):
        """
        Initialize the model.

        Args:
            chans : Number of output channels of the first convolution layer.
            num_pools : Number of down-sampling and up-sampling layers.
            in_chans : Number of channels in the input to the U-Net model.
            out_chans : Number of channels in the output to the U-Net model.
            drop_prob : Dropout probability.
            padding_size: Size of the padding.
            normalize: Whether to normalize the input.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob
        )

        self.padding_size = padding_size
        self.normalize = normalize

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """
        Convert the last dimension of the input to complex.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with the last dimension converted to complex.
        """
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """
        Convert the last dimension of the input to complex.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with the last dimension converted to complex.
        """
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the input.

        Args:
            x: Input tensor.

        Returns:
            Tuple of mean, standard deviation, and normalized input.
        """
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the input.

        Args:
            x: Input tensor.
            mean: Mean of the input.
            std: Standard deviation of the input.

        Returns:
            Unnormalized input.
        """
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """
        Pad the input with zeros to make it square.

        Args:
            x: Input tensor.

        Returns:
            Padded input tensor and the padding.
        """
        _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """
        Unpad the input.

        Args:
            x: Input tensor.
            h_pad: Horizontal padding.
            w_pad: Vertical padding.
            h_mult: Horizontal multiplier.
            w_mult: Vertical multiplier.

        Returns:
            Unpadded input tensor.
        """
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            Normalized UNet output tensor.
        """
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        mean = 1.0
        std = 1.0

        x = self.complex_to_chan_dim(x)
        if self.normalize:
            x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)

        if self.normalize:
            x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    TODO: move this to a separate file.
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net to the coil images to estimate coil
    sensitivities. It can be used with the end-to-end variational network.
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_type: str = "2D",  # TODO: make this generalizable
        fft_type: str = "orthogonal",
        normalize: bool = True,
    ):
        """
        Initialize the model.

        Args:
            chans : Number of output channels of the first convolution layer.
            num_pools : Number of down-sampling and up-sampling layers.
            in_chan s: Number of channels in the input to the U-Net model.
            out_chans : Number of channels in the output to the U-Net model.
            drop_prob : Dropout probability.
            mask_type : Type of mask to use.
            fft_type : Type of FFT to use.
            normalize : Whether to normalize the data.
        """
        super().__init__()

        self.mask_type = mask_type
        self.fft_type = fft_type

        self.norm_unet = NormUnet(
            chans, num_pools, in_chans=in_chans, out_chans=out_chans, drop_prob=drop_prob, normalize=normalize
        )

    @staticmethod
    def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert the last dimension of the input to the batch dimension.

        Args:
            x: Input tensor.

        Returns:
            Tuple of the converted tensor and the original last dimension.
        """
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    @staticmethod
    def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Convert the batch dimension of the input to the last dimension.

        Args:
            x: Input tensor.
            batch_size: Original batch size.

        Returns:
            Converted tensor.
        """
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    @staticmethod
    def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
        """
        Divide the input by the root of the sum of squares of the magnitude of each complex number.

        Args:
            x: Input tensor.

        Returns:
            RSS output tensor.
        """
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            masked_kspace: Masked k-space data.
            mask: Mask to apply to the k-space data.

        Returns:
            Normalized UNet output tensor.
        """
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs, mask_type=self.mask_type)

        # convert to image space
        x = ifft2c(x, self.fft_type)
        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net regularizer.
    To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        unet_padding_size: int = 15,
        normalize: bool = True,
        use_sens_net: bool = True,
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_normalize: bool = True,
        sens_mask_type: str = "1D",
        fft_type: str = "orthogonal",
        output_type: str = "RSS",
        no_dc: bool = False,
    ):
        """
        Initialize the model.

        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade U-Net.
            unet_padding_size : Size of padding to use for U-Net.
            normalize : Whether to normalize the data.
            use_sens_net : Whether to use a sensitivity network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for sensitivity map U-Net.
            sens_normalize : Whether to normalize the sensitivity map.
            sens_mask_type : Type of mask to use for sensitivity map.
            fft_type : Type of FFT to use.
            output_type : Type of output to use.
            no_dc : Whether to remove the DC component.
        """
        super().__init__()

        self.fft_type = fft_type
        self.output_type = output_type
        self.sens_mask_type = sens_mask_type

        self.no_dc = no_dc

        self.use_sens_net = use_sens_net
        self.sens_net = (
            SensitivityModel(
                sens_chans, sens_pools, fft_type=self.fft_type, mask_type=self.sens_mask_type, normalize=sens_normalize
            )
            if self.use_sens_net
            else None
        )

        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet(chans, pools, padding_size=unet_padding_size, normalize=normalize),
                    fft_type=self.fft_type,
                    no_dc=self.no_dc,
                )
                for _ in range(num_cascades)
            ]
        )

        # TODO: replace print with logger
        print("No of parameters: {:,d}".format(self.get_num_params()))

    def get_num_params(self):
        """
        Get the number of parameters in the model.

        Returns:
            Number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, masked_kspace: torch.Tensor, sense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            masked_kspace: Masked k-space data.
            sense: Sensitivity maps.
            mask: Mask.

        Returns:
            Reconstructed image.
        """
        sens_maps = self.sens_net(masked_kspace, mask) if self.use_sens_net and self.sens_net is not None else sense

        pred_kspace = masked_kspace.clone()
        for _, cascade in enumerate(self.cascades):
            pred_kspace = cascade(pred_kspace, masked_kspace, mask, sens_maps)

        if self.output_type == "SENSE":
            pred = torch.sum(complex_mul(ifft2c(pred_kspace, fft_type=self.fft_type), complex_conj(sens_maps)), 1)
            pred = pred[..., 0] + 1j * pred[..., 1]
        elif self.output_type == "RSS":
            pred = rss(complex_abs(ifft2c(pred_kspace, fft_type=self.fft_type)), dim=1)
        else:
            raise NotImplementedError("Output type should be either SENSE or RSS.")

        return pred


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input model as a regularizer.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(self, model: nn.Module, fft_type: str = "orthogonal", no_dc: bool = False):
        """
        Initialize the model block.

        Args:
            model: Model to apply soft data consistency.
            fft_type: Type of FFT to use.
            no_dc: Whether to remove the DC component.
        """
        super().__init__()

        self.model = model
        self.fft_type = fft_type
        self.no_dc = no_dc
        self.dc_weight = nn.Parameter(torch.ones(1))

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
        return complex_mul(x, complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self, pred: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            pred: Predicted k-space data.
            ref_kspace: Reference k-space data.
            mask: Mask to apply to the data.
            sens_maps: Sensitivity maps.

        Returns
        -------
            Reconstructed image.
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(pred)
        soft_dc = torch.where(mask, pred - ref_kspace, zero) * self.dc_weight

        eta = self.sens_reduce(pred, sens_maps)
        eta = self.model(eta)
        eta = self.sens_expand(eta, sens_maps)

        if not self.no_dc:
            eta = pred - soft_dc - eta

        return eta
