# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/transformerblock.py#L18

from typing import Tuple, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}


class MLPBlock(nn.Module):
    """
    Implementation of a multi-layer perceptron block, as presented in [1].

    References
    ----------
    .. [1] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold
        G, Gelly S, Uszkoreit J. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv
        preprint arXiv:2010.11929. 2020 Oct 22.

    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layer.
    mlp_dim : int
        Dimension of MLP layer.
    dropout_rate : float, optional
        Faction of the input units to drop. Default is ``0.0``.
    act : Union[Tuple, str], optional
        Activation type and arguments. Default is ``"GELU"``.
    dropout_mode : str, optional
        Dropout mode, can be "vit" or "swin". Default is ``vit``.
        "vit" mode uses two dropout instances as implemented in
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
        "swin" corresponds to one instance as implemented in
        https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23

    .. note::
        This is a wrapper for monai implementation of a multi-layer perceptron block.
        See: https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/transformerblock.py#L18
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        act: Union[Tuple, str] = "GELU",
        dropout_mode="vit",
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        if dropout_mode == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_mode == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of MLPBlock."""
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class SABlock(nn.Module):
    """
    Implementation of a self-attention block, as presented in [1].

    References
    ----------
    .. [1] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold
        G, Gelly S, Uszkoreit J. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv
        preprint arXiv:2010.11929. 2020 Oct 22.

    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layer.
    num_heads : int
        Number of attention heads.
    dropout_rate : float, optional
        Faction of the input units to drop. Default is ``0.0``.
    qkv_bias : bool, optional
        Bias term for the qkv linear layer. Default is ``False``.

    .. note::
        This is a wrapper for monai implementation of self-attention block.
        See: https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/transformerblock.py#L18
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of SABlock."""
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x


class TransformerBlock(nn.Module):
    """
    Implementation of a transformer block, as presented in [1].

    References
    ----------
    .. [1] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, Dehghani M, Minderer M, Heigold
        G, Gelly S, Uszkoreit J. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv
        preprint arXiv:2010.11929. 2020 Oct 22.

    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layer.
    mlp_dim : int
        Dimension of the mlp layer.
    num_heads : int
        Number of attention heads.
    dropout_rate : float, optional
        Faction of the input units to drop. Default is ``0.0``.
    qkv_bias : bool, optional
        Bias term for the qkv linear layer. Default is ``False``.
    spatial_dims : int, optional
        Number of spatial dimensions. Default is ``2``.

    .. note::
        This is a wrapper for monai implementation of self-attention block.
        See: https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/blocks/transformerblock.py#L18
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        spatial_dims: int = 2,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)

        if spatial_dims == 2:
            self.norm1 = nn.InstanceNorm2d(hidden_size)
            self.norm2 = nn.InstanceNorm2d(hidden_size)
        elif spatial_dims == 3:
            self.norm1 = nn.InstanceNorm3d(hidden_size)
            self.norm2 = nn.InstanceNorm3d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of TransformerBlock."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
