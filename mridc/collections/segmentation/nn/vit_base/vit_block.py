# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vit.py

from typing import Sequence, Union

import torch
from torch import nn

from mridc.collections.segmentation.nn.vit_base.patchembedding import PatchEmbeddingBlock
from mridc.collections.segmentation.nn.vit_base.transformer_block import TransformerBlock

__all__ = ["ViT"]


class ViT(nn.Module):
    """
    Implementation of a Vision Transformer (ViT), as presented in [1].

    ViT supports Torchscript but only works for Pytorch after 1.8.

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
        Dimension of hidden layer. Default is ``768``.
    mlp_dim : int
        Dimension of MLP layer. Default is ``3072``.
    num_layers : int
        Number of transformer layers. Default is ``12``.
    num_heads : int
        Number of attention heads. Default is ``12``.
    pos_embed : str
        Position embedding layer type. Default is ``"conv"``.
    classification : bool
        Whether to add a classification head. Default is ``False``.
    dropout_rate : float, optional
        Faction of the input units to drop. Default is ``0.0``.
    spatial_dims : int, optional
        Number of spatial dimensions. Default is ``3``.
    post_activation : str, optional
        Post activation layer type. Default is ``"Tanh"``.
    qkv_bias : bool, optional
        Whether to add a bias to query, key, value. Default is ``False``.

    .. note::
        This is a wrapper for monai implementation of PatchEmbeddingBlock.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vit.py
    """

    def __init__(  # noqa: C901
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Forward pass of the network."""
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out
