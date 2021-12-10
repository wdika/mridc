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
        """
        Initialize parameters following the way proposed in the paper.

        Returns:
            None
        """
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """
        Orthogonalize the weights of a convolutional layer.

        Args:
            weights: The weights of the convolutional layer.
            chunks: The number of chunks to split the weights into.

        Returns:
            The orthogonalized weights.
        """
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """
        Determine the convolutional class to use.

        Args:
            n_dim: The number of dimensions of the convolutional layer.

        Returns:
            The convolutional class to use.
        """
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        """
        Extra information to be printed when printing the model.

        Returns:
            The extra information.
        """
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """
        Check forward input.

        Args:
            _input: The input to check.

        Returns:
            None
        """
        if _input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(_input.size(1), self.input_size)
            )

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """
        Check forward hidden.

        Args:
            _input: The input to check.
            hx: The hidden to check.
            hidden_label: The label of the hidden.

        Returns:
            None
        """
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    _input.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size
                )
            )


class ConvGRUCell(ConvGRUCellBase):
    """A Convolutional GRU cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Initialize the ConvGRUCell.

        Args:
            input_size: The number of channels in the input.
            hidden_size: The number of channels in the hidden state.
            conv_dim: The number of dimensions of the convolutional layer.
            kernel_size: The size of the convolutional kernel.
            dilation: The dilation of the convolutional kernel.
            bias: Whether or not to add a bias.
        """
        super(ConvGRUCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)

    def forward(self, _input, hx):
        """
        Forward the ConvGRUCell.

        Args:
            _input: The input to the ConvGRUCell.
            hx: The hidden state of the ConvGRUCell.

        Returns:
            The new hidden state of the ConvGRUCell.
        """
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

        Args:
            input_size: The number of channels in the input.
            hidden_size: The number of channels in the hidden state.
            conv_dim: The number of dimensions of the convolutional layer.
            kernel_size: The size of the convolutional kernel.
            dilation: The dilation of the convolutional kernel.
            bias: Whether or not to add a bias.
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
        """
        Reset the parameters.

        Returns:
            None
        """
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        nn.init.xavier_uniform_(self.ih.weight, nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.hh.weight)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """
        Orthogonalize the weights.

        Args:
            weights: The weights to orthogonalize.
            chunks: The number of chunks to orthogonalize.

        Returns:
            The orthogonalized weights.
        """
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """
        Determine the convolutional class.

        Args:
            n_dim: The number of dimensions.

        Returns:
            The convolutional class.
        """
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        """
        Extra information about the ConvMGUCellBase.

        Returns:
            The extra information.
        """
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """
        Check the forward input.

        Args:
            _input: The input to check.

        Returns:
            None
        """
        if _input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(_input.size(1), self.input_size)
            )

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """
        Check the forward hidden.

        Args:
            _input: The input to check.
            hx: The hidden to check.
            hidden_label: The hidden label.

        Returns:
            None
        """
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    _input.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size
                )
            )


class ConvMGUCell(ConvMGUCellBase):
    """Convolutional Minimal Gated Unit cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Initialize the ConvMGUCell.

        Args:
            input_size: The input size.
            hidden_size: The hidden size.
            conv_dim: The convolutional dimension.
            kernel_size: The kernel size.
            dilation: The dilation.
            bias: Whether to use a bias.
        """
        super(ConvMGUCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)

    def forward(self, _input, hx):
        """
        Forward the ConvMGUCell.

        Args:
            _input: The input.
            hx: The hidden.

        Returns:
            The output.
        """
        ih = self.ih(_input).chunk(2, dim=1)
        hh = self.hh(hx).chunk(2, dim=1)

        f = torch.sigmoid(ih[0] + hh[0])
        c = torch.tanh(ih[1] + f * hh[1])

        hy = c + f * (hx - c)

        return hy


class IndRNNCellBase(nn.Module):
    """
    Base class for Independently RNN cells as presented in [1]_.

    References
    ----------

    .. [1] Li, S. et al. (2018) ‘Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN’,
    Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, (1),
    pp. 5457–5466. doi: 10.1109/CVPR.2018.00572.

    """

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation, bias):
        """
        Initialize the IndRNNCellBase.

        Args:
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

        self.ih = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size,
            padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item(),
            dilation=dilation,
            bias=bias,
        )
        self.hh = nn.Parameter(
            nn.init.normal_(torch.empty(1, hidden_size, 1, 1), std=1.0 / (hidden_size * (1 + kernel_size ** 2)))
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters.

        Returns:
            None
        """
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)

        nn.init.normal_(self.ih.weight, std=1.0 / (self.hidden_size * (1 + self.kernel_size ** 2)))

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """
        Orthogonalize weights.

        Args:
            weights: The weights to orthogonalize.
            chunks: The chunks.

        Returns:
            The orthogonalized weights.
        """
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    @staticmethod
    def determine_conv_class(n_dim):
        """
        Determine the convolutional class.

        Args:
            n_dim: The number of dimensions.

        Returns:
            The convolutional class.
        """
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        """
        Extra information about the module, used for printing.

        Returns:
            The extra information.
        """
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """
        Check forward input.

        Args:
            _input: The input.

        Returns:
            The input.
        """
        if _input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(_input.size(1), self.input_size)
            )

    def check_forward_hidden(self, _input, hx, hidden_label=""):
        """
        Check forward hidden.

        Args:
            _input: The input.
            hx: The hidden.
            hidden_label: The hidden label.

        Returns:
            The hidden.
        """
        if _input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    _input.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size
                )
            )


class IndRNNCell(IndRNNCellBase):
    """Independently Recurrent Neural Network cell."""

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        """
        Args:
            input_size: The number of expected features in the input.
            hidden_size: The number of features in the hidden state.
            conv_dim: The dimension of the convolutional layer.
            kernel_size: The size of the convolving kernel.
            dilation: The spacing between the kernel points.
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        """
        super(IndRNNCell, self).__init__(input_size, hidden_size, conv_dim, kernel_size, dilation, bias)

    def forward(self, _input, hx):
        """
        Args:
            _input: A (batch, input_size) tensor containing input features.
            hx: A (batch, hidden_size) tensor containing the initial hidden

        Returns:
            h_next: A (batch, hidden_size) tensor containing the next hidden state
        """
        return nn.ReLU()(self.ih(_input) + self.hh * hx)
