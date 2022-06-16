# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch
import torch.nn as nn


class ConvRNNStack(nn.Module):
    """A stack of convolutional RNNs."""

    def __init__(self, convs, rnn):
        """
        Parameters
        ----------
        convs: list of convolutional layers
        rnn: list of RNN layers
        """
        super(ConvRNNStack, self).__init__()
        self.convs = convs
        self.rnn = rnn

    def forward(self, x, hidden):
        """
        Parameters
        ----------
        x: [batch_size, seq_len, input_size]
        hidden: [num_layers * num_directions, batch_size, hidden_size

        Returns
        -------
        output: [batch_size, seq_len, hidden_size]
        """
        return self.rnn(self.convs(x), hidden)


class ConvNonlinear(nn.Module):
    """A convolutional layer with nonlinearity."""

    def __init__(self, input_size, features, conv_dim, kernel_size, dilation, bias, nonlinear="relu"):
        """
        Initializes the convolutional layer.

        Parameters
        ----------
        input_size: number of input channels.
        features: number of output channels.
        conv_dim: number of dimensions of the convolutional layer.
        kernel_size: size of the convolutional kernel.
        dilation: dilation of the convolutional kernel.
        bias: whether to use bias.
        nonlinear: nonlinearity of the convolutional layer.
        """
        super(ConvNonlinear, self).__init__()

        self.input_size = input_size
        self.features = features
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)

        if nonlinear is not None and nonlinear.upper() == "RELU":
            self.nonlinear = torch.nn.ReLU()
        elif nonlinear is not None and nonlinear.upper() == "LEAKYRELU":
            self.nonlinear = torch.nn.LeakyReLU()
        elif nonlinear is None:
            self.nonlinear = lambda x: x
        else:
            raise ValueError("Please specify a proper nonlinearity")

        self.padding = [
            torch.nn.ReplicationPad1d(torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item()),
            torch.nn.ReplicationPad2d(torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item()),
            torch.nn.ReplicationPad3d(torch.div(dilation * (kernel_size - 1), 2, rounding_mode="trunc").item()),
        ][conv_dim - 1]

        self.conv_layer = self.conv_class(
            in_channels=input_size,
            out_channels=features,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the convolutional layer."""
        torch.nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity="relu")

        if self.conv_layer.bias is not None:
            nn.init.zeros_(self.conv_layer.bias)

    @staticmethod
    def determine_conv_class(n_dim):
        """Determines the convolutional layer class."""
        if n_dim == 1:
            return nn.Conv1d
        if n_dim == 2:
            return nn.Conv2d
        if n_dim == 3:
            return nn.Conv3d
        raise ValueError(f"Convolution of: {n_dim} dims is not implemented")

    def extra_repr(self):
        """Extra information about the layer."""
        s = "{input_size}, {features}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinear" in self.__dict__ and self.nonlinear != "tanh":
            s += ", nonlinearity={nonlinear}"
        return s.format(**self.__dict__)

    def check_forward_input(self, _input):
        """Checks input for correct size and shape."""
        if _input.size(1) != self.input_size:
            raise RuntimeError(f"input has inconsistent input_size: got {_input.size(1)}, expected {self.input_size}")

    def forward(self, _input):
        """Forward pass of the convolutional layer."""
        return self.nonlinear(self.conv_layer(self.padding(_input)))
