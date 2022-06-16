import torch

from mridc.collections.reconstruction.models.invrim_base.invert_to_learn import InvertibleLayer, InvertibleModule


class IRIM(InvertibleModule):
    def __init__(self, rnn, grad_fun, fft_type, n_channels=1):
        super(IRIM, self).__init__()
        self.rnn = rnn
        self.grad_fun = InvertibleGradUpdate(grad_fun, n_channels, fft_type)

    def forward(self, x, data=None):
        x = self.grad_fun.forward(x, data)
        for i in range(len(self.rnn)):
            x = self.rnn[i].forward(x)
            x = self.grad_fun.forward(x, data)
        return x

    def reverse(self, x, data=None):
        x = self.grad_fun.reverse(x, data)
        for i in range(len(self.rnn)):
            x = self.rnn[-i].reverse(x)
            x = self.grad_fun.reverse(x, data)
        return x


class InvertibleGradUpdate(InvertibleLayer):
    def __init__(self, grad_fun, n_channels, fft_type):
        super().__init__()
        self.grad_fun = grad_fun
        self.n_channels = n_channels
        self.fft_type = fft_type

    def _forward(self, x, data):
        grad = self.grad_fun(x[:, : self.n_channels], data, self.fft_type)
        assert grad.size(1) <= x.size(1) - self.n_channels
        x = torch.cat(
            (
                x[:, : self.n_channels],
                x[:, self.n_channels : self.n_channels + grad.size(1)] + grad,
                x[:, self.n_channels + grad.size(1) :],
            ),
            1,
        )
        return x

    def _reverse(self, y, data):
        grad = self.grad_fun(y[:, : self.n_channels], data, self.fft_type)
        assert grad.size(1) <= y.size(1) - self.n_channels
        y = torch.cat(
            (
                y[:, : self.n_channels],
                y[:, self.n_channels : self.n_channels + grad.size(1)] - grad,
                y[:, self.n_channels + grad.size(1) :],
            ),
            1,
        )
        return y

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        with torch.enable_grad():
            if x is None and y is not None:
                y = y.detach().requires_grad_(True)
                x = forward_fun(y, *args, **kwargs)
                grads = torch.autograd.grad(x, [y] + parameters, grad_outputs=grad_outputs)
                x = 2 * y - x
            elif x is not None:
                x = x.detach().requires_grad_(True)
                y = forward_fun(x, *args, **kwargs)
                grads = torch.autograd.grad(y, [x] + parameters, grad_outputs=grad_outputs)

        grad_input = grads[0]
        param_grads = grads[1:]

        return x, grad_input, param_grads
