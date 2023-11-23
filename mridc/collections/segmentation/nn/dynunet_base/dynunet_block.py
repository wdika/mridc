# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/dynunet.py

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.functional import interpolate

from mridc.collections.segmentation.nn.unetr_base.unetr_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    UnetUpBlock,
)

__all__ = ["DynUNet"]


class DynUNetSkipLayer(nn.Module):
    """
     Implementation of a Dynamic UNet (DynUNet) Skip Layer, based on [1].

     References
     ----------
     .. [1] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G, Norajitra T,
        Wirkert S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image segmentation. arXiv
        preprint arXiv:1809.10486. 2018 Sep 27.

     Parameters
     ----------
     index : int
         The index of the layer in the UNet structure.
     downsample : nn.Module
         The downsample layer of the skip connection.
     upsample : nn.Module
         The upsample layer of the skip connection.
     next_layer : nn.Module
         The next layer in the UNet structure.
     heads : List[torch.Tensor]
         The list of output tensors from the supervision heads. Default is ``None``.
    super_head : nn.Module
        The supervision head for this layer. Default is ``None``.

     .. note::
         This class is a wrapper of the original DynUNetSkipLayer class from MONAI.
         See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/dynunet.py

     .. note::
         Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip
         connection. The member `next_layer` may refer to instances of this class or the final bottleneck layer at the
         bottom the UNet structure. The purpose of using a recursive class like this is to get around the Torchscript
         restrictions on looping over lists of layers and accumulating lists of output tensors which must be indexed.
         The `heads` list is shared amongst all the instances of this class and is used to store the output from the
         supervision heads during forward passes of the network.
    """

    heads: Optional[List[torch.Tensor]]

    def __init__(  # noqa: C901
        self,
        index: int,
        downsample: nn.Module,
        upsample: nn.Module,
        next_layer: nn.Module,
        heads: List[torch.Tensor] = None,
        super_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer."""
        downout = self.downsample(x)
        nextout = self.next_layer(downout)
        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)

        return upout


class DynUNet(nn.Module):
    """
    Implementation of a Dynamic UNet (DynUNet) Skip Layer, based on [1].

    References
    ----------
    .. [1] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G, Norajitra T, Wirkert
        S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image segmentation. arXiv preprint
        arXiv:1809.10486. 2018 Sep 27.

    Parameters
    ----------
    spatial_dims : int
        The number of spatial dimensions of the input data.
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : Union[int, Sequence[int]]
        The kernel size for the convolutional layers.
    strides : Union[int, Sequence[int]]
        The stride for the convolutional layers.
    upsample_kernel_size : Union[int, Sequence[int]]
        Convolution kernel size for transposed convolution layers. The values should equal to strides[1:].
    filters : Sequence[int]
        The number of output channels for each block. Different from nnU-Net, in this implementation we add this
        argument to make the network more flexible. One way to determine this parameter is like:
        ``[64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]``. If not specified, the way which nnUNet used
        will be employed. Defaults to ``None``.
    dropout : float
        Dropout ratio. Defaults to no dropout.
    norm_name : str
        Feature normalization type and arguments. Defaults to ``INSTANCE``.
        `INSTANCE_NVFUSER` is a faster version of the instance norm layer, it can be used when:
        1) `spatial_dims=3`, 2) CUDA device is available, 3) `apex` is installed and 4) non-Windows OS is used.
    act_name : str
        Activation layer type and arguments. Defaults to ``leakyrelu``.
    deep_supervision : bool
        Whether to add deep supervision head before output. Defaults to ``False``. If ``True``, in training mode, the
        forward function will output not only the final feature map (from `output_block`), but also the feature maps
        that come from the intermediate up sample layers. In order to unify the return type (the restriction of
        TorchScript), all intermediate feature maps are interpolated into the same size as the final feature map and
        stacked together (with a new dimension in the first axis)into one single tensor. For instance, if there are two
         intermediate feature maps with shapes: (1, 2, 16, 12) and (1, 2, 8, 6), and the final feature map has the
        shape (1, 2, 32, 24), then all intermediate feature maps will be interpolated into (1, 2, 32, 24), and the
        stacked tensor will have the shape (1, 3, 2, 32, 24). When calculating the loss, you can use torch.unbind to
        get all feature maps can compute the loss one by one with the ground truth, then do a weighted average for all
        losses to achieve the final loss.
    deep_supr_num : int
        Number of feature maps that will output during deep supervision head. The value should be larger than 0 and
        less than the number of up sample layers. Defaults to ``1``.
    res_block : bool
        Whether to use residual connection based convolution blocks during the network. Defaults to ``False``.
    trans_bias : bool
        Whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.

    .. note::
        This class is a wrapper of the original DynUNetSkipLayer class from MONAI.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/dynunet.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

        def create_skips(
            index: int,
            downsamples: List[nn.Module],
            upsamples: List[nn.Module],
            bottleneck: nn.Module,
            superheads: List[nn.Module] = None,
        ) -> nn.Module:
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.

            Parameters
            ----------
            index : int
                The index of the current skip layer.
            downsamples : List[nn.Module]
                The list of downsample layers.
            upsamples : List[nn.Module]
                The list of upsample layers.
            bottleneck : nn.Module
                The bottleneck layer.
            superheads : List[nn.Module]
                The list of supervision heads. Default is ``None``.
            """
            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            if superheads is None:
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)
                return DynUNetSkipLayer(
                    index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer  # type: ignore
                )

            super_head_flag = False
            if index == 0:  # don't associate a supervision head with self.input_block
                rest_heads = superheads
            else:
                if len(superheads) > 0:
                    super_head_flag = True
                    rest_heads = superheads[1:]
                else:
                    rest_heads = nn.ModuleList()

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads)
            if super_head_flag:
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=self.heads,
                    super_head=superheads[0],
                )

            return DynUNetSkipLayer(
                index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer  # type: ignore
            )

        if not self.deep_supervision:
            self.skip_layers = create_skips(
                0, [self.input_block] + list(self.downsamples), self.upsamples[::-1], self.bottleneck
            )
        else:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

    def check_kernel_stride(self):
        """Check the length of kernel_size and strides."""
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        """Check the number of deep supervision heads."""
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        """Check the length of filters."""
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("Length of filters should be no less than the length of strides.")
        self.filters = filters[: len(self.strides)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1)
        return out

    def get_input_block(self) -> nn.Module:
        """Get the input block."""
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self) -> nn.Module:
        """Get the bottleneck block."""
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, idx: int) -> nn.Module:
        """Get the output block."""
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self) -> nn.ModuleList:
        """Get the downsampling blocks."""
        inp, out = self.filters[:-2], self.filters[1:-1]  # type: ignore
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)  # type: ignore

    def get_upsamples(self) -> nn.ModuleList:
        """Get the upsampling blocks."""
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]  # type: ignore
        return self.get_module_list(
            inp,  # type: ignore
            out,  # type: ignore
            kernel_size,
            strides,
            UnetUpBlock,
            upsample_kernel_size,
            trans_bias=self.trans_bias,
        )

    def get_module_list(  # noqa: C901
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ) -> nn.ModuleList:
        """
        Get the module list of the network.

        Parameters
        ----------
        in_channels : List[int]
            The number of input channels.
        out_channels : List[int]
            The number of output channels.
        kernel_size : Sequence[Union[Sequence[int], int]]
            The kernel size.
        strides : Sequence[Union[Sequence[int], int]]
            The strides size.
        conv_block : nn.Module
            The convolutional block.
        upsample_kernel_size : Optional[Sequence[Union[Sequence[int], int]]]
            The upsample kernel size.
        trans_bias : bool
            Whether to use bias in the transpose convolutional layer.

        Returns
        -------
        nn.ModuleList
            The module list of the network.
        """
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self) -> nn.ModuleList:
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module: nn.Module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
