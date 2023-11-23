# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn

# Taken and adapted from: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py


class ConvLSTMCell(nn.Module):
    """
    Wrapper for ConvLSTM cell.

    .. note::
        See: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether to add the bias.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(  # noqa: W0221
        self, input_tensor: torch.Tensor, cur_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvLSTM cell.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor. Shape [batch_size, input_dim, height, width]
        cur_state: torch.Tensor
            Current state of the hidden state. Shape [batch_size, hidden_dim, height, width]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the next hidden state and the cell state. Shape [batch_size, hidden_dim, height, width]

        """
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the hidden state.

        Parameters
        ----------
        batch_size: int
            Batch size.
        image_size: Tuple[int, int]
            Size of the image. Shape [height, width]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the next hidden state and the cell state. Shape [batch_size, hidden_dim, height, width]
        """
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    """
    Wrapper for ConvLSTM cell.

    .. note::
        See: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

    Parameters
    ----------
    input_dim: int
        Number of channels of input tensor.
    hidden_dim: int
        Number of channels of hidden state.
    kernel_size: Any
        Size of the convolutional kernel.
    num_layers: int
        Number of layers.
    batch_first: bool
        Whether the first dimension corresponds to the batch size. Default is ``False``.
    bias: bool
        Whether to add the bias. Default is ``True``.
    return_all_layers: bool
        Whether to return all layers or just the last layer. Default is ``False``.
    """

    def __init__(  # noqa: W0221
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Any,
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ):
        super().__init__()

        kernel_size = (kernel_size, kernel_size)

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:  # type: ignore
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]  # type: ignore
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],  # type: ignore
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, input_tensor: torch.Tensor, hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> tuple[list[Tensor], list[list[Any]]]:
        """
        Forward pass of the ConvLSTM.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape [batch_size, seq_len, channels, height, width] if ``batch_first`` is ``False``.
            Otherwise, the shape should be [seq_len, batch_size, channels, height, width].
        hidden_state : Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
            List of tuples of the hidden state and the cell state for each layer. The shape of each tensor should be
            [batch_size, hidden_dim, height, width]. If ``None``, the hidden state will be initialized to zero.

        Returns
        -------
        Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]
            Tuple of the output tensor and the list of the hidden state and the cell state for each layer. The shape
            of the output tensor is [batch_size, seq_len, hidden_dim, height, width] if ``batch_first`` is ``False``.
            Otherwise, the shape should be [seq_len, batch_size, hidden_dim, height, width]. The shape of each tensor
            in the list is [batch_size, hidden_dim, height, width].
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()

        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
