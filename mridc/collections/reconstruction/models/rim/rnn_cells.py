# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch
import torch.nn as nn


class ConvGRUCellBase(nn.Module):
    """
    Base class for Conv Gated Recurrent Unit (GRU) cells.
    # TODO: add paper reference
    """

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation, bias):
        super(ConvGRUCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)

        self.ih = nn.Conv2d(
            input_size,
            3 * hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=bias,
        )
        self.hh = nn.Conv2d(
            hidden_size,
            3 * hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters following the way proposed in the paper."""
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """Orthogonalize the weights of a convolutional layer."""
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """Determine the convolutional class to use."""
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        raise NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        """Extra information to be printed when printing the model."""
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """Check forward input."""
        if _input.size(1) != self.input_size:
            raise RuntimeError(f"input has inconsistent input_size: got {_input.size(1)}, expected {self.input_size}")

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """Check forward hidden."""
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {_input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}"
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}"
            )


class ConvGRUCell(ConvGRUCellBase):
    """A Convolutional GRU cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Initialize the ConvGRUCell.

        Parameters
        ----------
        input_size: The number of channels in the input.
        hidden_size: The number of channels in the hidden state.
        conv_dim: The number of dimensions of the convolutional layer.
        kernel_size: The size of the convolutional kernel.
        dilation: The dilation of the convolutional kernel.
        bias: Whether to add a bias.
        """
        super(ConvGRUCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)
        self.conv_dim = conv_dim

    def forward(self, _input, hx):
        """Forward pass of the ConvGRUCell."""
        if self.conv_dim == 3:
            _input = _input.unsqueeze(0)
            hx = hx.permute(1, 0, 2, 3).unsqueeze(0)

        ih = self.ih(_input).chunk(3, 1)
        hh = self.hh(hx).chunk(3, 1)

        r = torch.sigmoid(ih[0] + hh[0])
        z = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2] + r * hh[2])

        hx = n * (1 - z) + z * hx

        return hx


class ConvMGUCellBase(nn.Module):
    """
    A base class for a Convolutional Minimal Gated Unit cell.
    # TODO: add paper reference
    """

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation, bias):
        """
        Initialize the ConvMGUCellBase.

        Parameters
        ----------
        input_size: The number of channels in the input.
        hidden_size: The number of channels in the hidden state.
        conv_dim: The number of dimensions of the convolutional layer.
        kernel_size: The size of the convolutional kernel.
        dilation: The dilation of the convolutional kernel.
        bias: Whether to add a bias.
        """
        super(ConvMGUCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)

        self.ih = nn.Conv2d(
            input_size,
            2 * hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=bias,
        )
        self.hh = nn.Conv2d(
            hidden_size,
            2 * hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        nn.init.xavier_uniform_(self.ih.weight, nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.hh.weight)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """Orthogonalize the weights."""
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """Determine the convolutional class."""
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        raise ValueError(f"Convolution of: {n_dim} dims is not implemented")

    def extra_repr(self):
        """Extra information about the ConvMGUCellBase."""
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """Check the forward input."""
        if _input.size(1) != self.input_size:
            raise RuntimeError(f"input has inconsistent input_size: got {_input.size(1)}, expected {self.input_size}")

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """Check the forward hidden."""
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {_input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}"
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}"
            )


class ConvMGUCell(ConvMGUCellBase):
    """Convolutional Minimal Gated Unit cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Initialize the ConvMGUCell.

        Parameters
        ----------
        input_size: The input size.
        hidden_size: The hidden size.
        conv_dim: The convolutional dimension.
        kernel_size: The kernel size.
        dilation: The dilation.
        bias: Whether to use a bias.
        """
        super(ConvMGUCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)
        self.conv_dim = conv_dim

    def forward(self, _input, hx):
        """Forward the ConvMGUCell."""
        if self.conv_dim == 3:
            _input = _input.unsqueeze(0)
            hx = hx.permute(1, 0, 2, 3).unsqueeze(0)

        ih = self.ih(_input).chunk(2, dim=1)
        hh = self.hh(hx).chunk(2, dim=1)

        f = torch.sigmoid(ih[0] + hh[0])
        c = torch.tanh(ih[1] + f * hh[1])

        return c + f * (hx - c)


class IndRNNCellBase(nn.Module):
    """
    Base class for Independently RNN cells as presented in [1]_.

    References
    ----------
    .. [1] Li, S. et al. (2018) ‘Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN’, Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, (1), pp. 5457–5466. doi: 10.1109/CVPR.2018.00572.
    """

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation, bias):
        """
        Initialize the IndRNNCellBase.

        Parameters
        ----------
        input_size: The input size.
        hidden_size: The hidden size.
        conv_dim: The convolutional dimension.
        kernel_size: The kernel size.
        dilation: The dilation.
        bias: Whether to use a bias.
        """
        super(IndRNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)

        self.ih = self.conv_class(
            input_size,
            hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=bias,
        )

        if self.conv_dim == 2:
            self.hh = nn.Parameter(
                nn.init.normal_(torch.empty(1, hidden_size, 1, 1), std=1.0 / (hidden_size * (1 + kernel_size**2)))
            )
        elif self.conv_dim == 3:
            self.hh = nn.Parameter(
                nn.init.normal_(torch.empty(1, hidden_size, 1, 1, 1), std=1.0 / (hidden_size * (1 + kernel_size**2)))
            )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)

        nn.init.normal_(self.ih.weight, std=1.0 / (self.hidden_size * (1 + self.kernel_size**2)))

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """Orthogonalize the weights."""
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """Determine the convolutional class."""
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        raise NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        """Extra information about the module, used for printing."""
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """Check forward input."""
        if _input.size(1) != self.input_size:
            raise RuntimeError(f"input has inconsistent input_size: got {_input.size(1)}, expected {self.input_size}")

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """Check forward hidden."""
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {_input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}"
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}"
            )


class IndRNNCell(IndRNNCellBase):
    """Independently Recurrent Neural Network cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Parameters
        ----------
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        conv_dim: The dimension of the convolutional layer.
        kernel_size: The size of the convolved kernel.
        dilation: The spacing between the kernel points.
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        """
        super(IndRNNCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)
        self.conv_dim = conv_dim

    def forward(self, _input, hx):
        """Forward propagate the RNN cell."""
        if self.conv_dim == 3:
            # TODO: Check if this is correct
            _input = _input.unsqueeze(0)
            hx = hx.permute(1, 0, 2, 3).unsqueeze(0)

        return nn.ReLU()(self.ih(_input) + self.hh * hx)
