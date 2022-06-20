import torch
from torch import nn

from mridc.collections.reconstruction.models.invrim_base.invert_to_learn import InvertibleLayer
from mridc.collections.reconstruction.models.invrim_base.residual_blocks import ResidualBlockPixelshuffle
from mridc.collections.reconstruction.models.invrim_base.utils import determine_conv_functional


class RevNetLayer(InvertibleLayer):
    def __init__(self, n_channels, n_hidden, dilation=1, conv_nd=2, residual_function=ResidualBlockPixelshuffle):
        """
        Initialize the layer.

        Parameters
        ----------
        n_channels : int
            Number of channels.
        n_hidden : int
            Number of hidden channels.
        dilation : int
            Dilation factor.
        conv_nd : int
            Number of dimensions of the convolution.
        residual_function : callable
            Residual function.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_left = self.n_channels // 2
        self.n_right = self.n_channels - self.n_left
        self.n_hidden = n_hidden
        self.conv_nd = conv_nd
        self.update_right = residual_function(
            self.n_left, self.n_right, self.n_hidden, dilation=dilation, conv_nd=conv_nd
        )

    def _forward(self, x, *args, **kwargs):
        """
        Forward pass of the layer.

        Forward function.

        Parameters
        ----------
        x : torch.Tensor
            The input to the layer.
        args : tuple
            The arguments to the forward function.
        kwargs : dict
            The keyword arguments to the forward function.
        """
        x_left, x_right = x[:, : self.n_left], x[:, self.n_left : self.n_channels]
        y_right = x_right + self.update_right(x_left)
        return torch.cat((x_left, y_right, x[:, self.n_channels :]), 1)

    def _reverse(self, y, *args, **kwargs):
        """
        Reverse function.

        Parameters
        ----------
        y : torch.Tensor
            The output of the forward function.
        args : tuple
            The arguments to the reverse function.
        kwargs : dict
            The keyword arguments to the reverse function.
        """
        x_left, y_right = y[:, : self.n_left], y[:, self.n_left : self.n_channels]
        x_right = y_right - self.update_right(x_left)
        return torch.cat((x_left, x_right, y[:, self.n_channels :]), 1)

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        """
        Gradient function.

        Parameters
        ----------
        forward_fun : callable
            The forward function.
        reverse_fun : callable
            The reverse function.
        x : torch.Tensor
            The input to the forward function.
        y : torch.Tensor
            The output of the forward function.
        grad_outputs : torch.Tensor
            The gradient of the output of the forward function.
        parameters : list
            The parameters of the forward function.
        args : tuple
            The arguments to the forward function.
        kwargs : dict
            The keyword arguments to the forward function.
        """
        with torch.enable_grad():
            if x is None and y is not None:
                y = y.detach().requires_grad_(True)
                x = forward_fun(y)
                grads = torch.autograd.grad(x, [y] + parameters, grad_outputs=grad_outputs)
                x = 2 * y - x
            elif x is not None:
                x = x.detach().requires_grad_(True)
                y = forward_fun(x)
                grads = torch.autograd.grad(y, [x] + parameters, grad_outputs=grad_outputs)

        grad_input = grads[0]
        param_grads = grads[1:]

        return x, grad_input, param_grads


class Housholder1x1(InvertibleLayer):
    def __init__(self, num_inputs, n_projections=3, conv_nd=2):
        """
        Initialize the layer.

        Parameters
        ----------
        num_inputs : int
            The number of inputs to the layer.
        n_projections : int
            The number of projections to use.
        conv_nd : int
            The number of dimensions to use for the convolution.
        """
        super(Housholder1x1, self).__init__()
        n_projections = min(n_projections, num_inputs)
        self.weights = nn.Parameter(torch.randn((n_projections, num_inputs, 1)))
        self.conv_nd = conv_nd
        self.conv = determine_conv_functional(conv_nd)
        self.register_buffer("I", torch.eye(num_inputs))

    def _forward(self, x, W=None):
        """
        Forward function.

        Parameters
        ----------
        x : torch.Tensor
            The input to the layer.
        args : tuple
            The arguments to the forward function.
        kwargs : dict
            The keyword arguments to the forward function.
        """
        if W is None:
            W = self._get_weight()
        for _ in range(self.conv_nd):
            W = W.unsqueeze(-1)
        return self.conv(x.to(W), W)

    def _reverse(self, y, W=None):
        """
        Reverse function.

        Parameters
        ----------
        y : torch.Tensor
            The output of the forward function.
        args : tuple
            The arguments to the reverse function.
        kwargs : dict
            The keyword arguments to the reverse function.
        """
        if W is None:
            W = self._get_weight()
        W = W.t()
        for _ in range(self.conv_nd):
            W = W.unsqueeze(-1)
        return self.conv(y, W.to(y))

    def _get_weight(self):
        """Get the weight matrix."""
        V = self.weights
        V_t = self.weights.transpose(1, 2)
        U = self.I - 2 * torch.bmm(V, V_t) / torch.bmm(V_t, V)
        return torch.chain_matmul(*U)

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        """
        Gradient function.

        Parameters
        ----------
        forward_fun : callable
            The forward function.
        reverse_fun : callable
            The reverse function.
        x : torch.Tensor
            The input to the forward function.
        y : torch.Tensor
            The output of the forward function.
        grad_outputs : torch.Tensor
            The gradient of the output of the forward function.
        parameters : list
            The parameters of the forward function.
        args : tuple
            The arguments to the forward function.
        kwargs : dict
            The keyword arguments to the forward function.
        """
        with torch.enable_grad():
            W = self._get_weight()
            if x is None and y is not None:
                x = reverse_fun(y.detach(), W)
                grad_input = reverse_fun(grad_outputs.detach(), W)
                param_grads = torch.autograd.grad(grad_input, parameters, grad_outputs=x.detach())
            else:
                x = x.detach().requires_grad_(True)
                y = forward_fun(x, W)
                grads = torch.autograd.grad(y, [x] + parameters, grad_outputs=grad_outputs)
                grad_input = grads[0]
                param_grads = grads[1:]
        return x, grad_input, param_grads
