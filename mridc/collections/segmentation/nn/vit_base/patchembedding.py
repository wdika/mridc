# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/patchembedding.py#L28

from typing import Sequence, Type, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mridc.collections.segmentation.nn.vit_base.utils import trunc_normal_

SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


class PatchEmbeddingBlock(nn.Module):
    """
    Implementation of a patch embedding block, as presented in [1].

    References
    ----------
    .. [1] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold
        G, Gelly S, Uszkoreit J. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv
        preprint arXiv:2010.11929. 2020 Oct 22.

    Parameters
    ----------
    in_channels : int
        Dimension of input channels.
    img_size : Union[Sequence[int], int]
        Dimension of input image.
    patch_size : Union[Sequence[int], int]
        Dimension of patch size.
    hidden_size : int
        Dimension of hidden layer.
    num_heads : int
        Number of attention heads.
    pos_embed : str
        Position embedding layer type.
    dropout_rate : float, optional
        Faction of the input units to drop. Default is ``0.0``.
    spatial_dims : int, optional
        Number of spatial dimensions. Default is ``3``.

    .. note::
        This is a wrapper for monai implementation of PatchEmbeddingBlock. See: https://github.com/Project-MONAI/MONAI/
        blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/patchembedding.py#L28
    """

    def __init__(  # noqa: C901
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = pos_embed

        # img_size = (img_size,) * spatial_dims  # type: ignore
        # patch_size = (patch_size,) * spatial_dims  # type: ignore
        for m, p in zip(img_size, patch_size):  # type: ignore
            m = list(m) if isinstance(m, tuple) else [m]  # type: ignore
            p = list(p) if isinstance(p, tuple) else [p]  # type: ignore
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")

        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])  # type: ignore
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            if spatial_dims == 2:
                self.patch_embeddings = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
            elif spatial_dims == 3:
                self.patch_embeddings = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
            else:
                raise ValueError(f"Convolutional patch embedding not supported for {spatial_dims}D.")
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i + 1}": p for i, p in enumerate(patch_size)}  # type: ignore
            self.patch_embeddings = nn.Sequential(
                einops.rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize the weights of the module."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of PatchEmbeddingBlock."""
        x = self.patch_embeddings(x)
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class PatchEmbed(nn.Module):
    """
    Implementation of a patch embedding block, as presented in [1].

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    References
    ----------
    .. [1] Liu Z, Lin Y, Cao Y, Hu H, Wei Y, Zhang Z, Lin S, Guo B. Swin transformer: Hierarchical vision transformer
        using shifted windows. InProceedings of the IEEE/CVF International Conference on Computer Vision 2021 (pp.
        10012-10022).

    Parameters
    ----------
    patch_size : Union[Sequence[int], int]
        Dimension of patch size. Default is ``2``.
    in_chans : int
        Dimension of input channels. Default is ``1``.
    embed_dim : int
        Dimension of embedding. Default is ``48``.
    norm_layer : Type[nn.LayerNorm]
        Normalization layer. Default is ``nn.InstanceNorm2d``.
    spatial_dims : int, optional
        Number of spatial dimensions. Default is ``3``.

    .. note::
        This is a wrapper for monai implementation of PatchEmbeddingBlock. See: https://github.com/Project-MONAI/MONAI/
        blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/patchembedding.py#L28
    """

    def __init__(  # noqa: C901
        self,
        patch_size: Union[Sequence[int], int] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[nn.LayerNorm] = nn.InstanceNorm2d,  # noqa: B008
        spatial_dims: int = 3,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("Spatial dimension should be 2 or 3.")

        patch_size = (patch_size,) * spatial_dims  # type: ignore
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        if spatial_dims == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif spatial_dims == 3:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"Convolutional patch embedding not supported for {spatial_dims}D.")
        if spatial_dims == 2:
            self.norm = nn.InstanceNorm2d(embed_dim)
        elif spatial_dims == 3:
            self.norm = nn.InstanceNorm3d(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of PatchEmbed."""
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:  # type: ignore
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))  # type: ignore
            if h % self.patch_size[1] != 0:  # type: ignore
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))  # type: ignore
            if d % self.patch_size[0] != 0:  # type: ignore
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))  # type: ignore

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:  # type: ignore
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))  # type: ignore
            if h % self.patch_size[0] != 0:  # type: ignore
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))  # type: ignore

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x
