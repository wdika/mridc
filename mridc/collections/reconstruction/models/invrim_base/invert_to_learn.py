import itertools
from abc import ABC, abstractmethod

import torch
from torch.autograd import Function


def make_layer_memory_free(layer, save_input=False):
    """
    Makes a layer memory free by replacing the forward and reverse functions with a wrapper that saves the
    input and output of the layer. This is useful for layers that are not invertible, but can be made invertible
    by replacing the forward and reverse functions with the same function.
    """
    if isinstance(layer, InvertibleLayer):
        layer.memory_free = True
        layer.save_input = save_input


class InvertToLearnFunction(Function):
    @staticmethod
    def forward(ctx, n_args, layer, forward_fun, reverse_fun, args, kwargs, *tensors):
        x = tensors[0] if n_args == 1 else list(tensors[: 2 * n_args : 2])
        with torch.no_grad():
            y = forward_fun(x, *args, **kwargs)

        if any(x_.requires_grad for x_ in tensors):
            ctx.n_args = n_args
            ctx.layer = layer
            ctx.forward_fun = forward_fun
            ctx.reverse_fun = reverse_fun
            ctx.args = args
            ctx.kwargs = kwargs
            ctx.tensors = list(tensors[2 * n_args :])
            if layer.save_input:
                if n_args == 1:
                    ctx.save_for_backward(x)
                else:
                    ctx.save_for_backward(*x)

        if isinstance(y, list):
            y = [(y_, y_.detach()) for y_ in y]
            y = tuple(itertools.chain(*y))
        else:
            y = (y, y.detach())

        return y

    @staticmethod
    def backward(ctx, *out):
        if len(out) == 2:
            y, grad_outputs = out[1].detach(), out[0]
        else:
            y, grad_outputs = list(out[1::2]), list(out[::2])
            y = [y_.detach() for y_ in y]

        x = None
        if len(ctx.saved_tensors) > 0:
            x = ctx.saved_tensors[0] if ctx.n_args == 1 else list(ctx.saved_tensors)
        x, grad_inputs, param_grads = ctx.layer.gradfun(
            ctx.forward_fun, ctx.reverse_fun, x, y, grad_outputs, ctx.tensors, *ctx.args, **ctx.kwargs
        )
        if ctx.n_args == 1:
            input_gradients = (grad_inputs, x.detach())
        else:
            x = [x_.detach() for x_ in x]
            input_gradients = tuple(itertools.chain(*zip(grad_inputs, x)))
        parameter_gradients = tuple(param_grads)

        return (None, None, None, None, None, None) + input_gradients + parameter_gradients


class InvertibleModule(torch.nn.Module, ABC):
    """
    Abstract class to define any invertible Module, be it a layer or whole network.
    Inheriting class should implement forward and backward.
    """

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def reverse(self, y, *args, **kwargs):
        pass


class InvertibleLayer(InvertibleModule, ABC):
    """
    Abstract class for all invertible layers. This builds the core of all invertible networks. All sub-classes are
    required to implement _forward and _backward which implement the layers computations in the respective directions.
    """

    def __init__(self):
        super().__init__()
        self.memory_free = False
        self.save_input = False

    def forward(self, x, *args, **kwargs):
        """
        Forward operation of the invertible layer
        :param x: Tensor or list of Tensors. Only gradients for x will be valid in invert to learn.
        :param args: Additional Inputs
        :param kwargs: Keyword Arguments
        :return: Tensor or list of Tensors
        """
        if self.memory_free:
            if isinstance(x, list):
                n_args = len(x)
                x = list(itertools.chain(*x))
            else:
                n_args = 1
                x = list(x)
            tensors = x + list(self.parameters())
            for arg in args:
                if torch.is_tensor(arg) and arg.requires_grad:
                    tensors.append(arg)
            for arg in kwargs.values():
                if torch.is_tensor(arg) and arg.requires_grad:
                    tensors.append(arg)
            forward_fun = self._forward
            reverse_fun = self._reverse

            y = InvertToLearnFunction.apply(n_args, self, forward_fun, reverse_fun, args, kwargs, *tensors)
            if len(y) > 2:
                y = list(zip(y[::2], y[1::2]))
        else:
            y = self._forward(x, *args, **kwargs)

        return y

    def reverse(self, y, *args, **kwargs):
        """
        Reverse operation of the invertible layer
        :param y: Tensor or list of Tensors. Only gradients for y will be valid in invert to learn.
        :param args: Additional Inputs
        :param kwargs: Keyword Arguments
        :return: Tensor or list of Tensors
        """
        if self.memory_free:
            if isinstance(y, list):
                n_args = len(y)
                y = list(itertools.chain(*y))
            else:
                n_args = 1
                y = list(y)

            tensors = list(y) + list(self.parameters())
            for arg in args:
                if torch.is_tensor(arg) and arg.requires_grad:
                    tensors.append(arg)
            for arg in kwargs.values():
                if torch.is_tensor(arg) and arg.requires_grad:
                    tensors.append(arg)

            reverse_fun = self._forward
            forward_fun = self._reverse
            x = InvertToLearnFunction.apply(n_args, self, forward_fun, reverse_fun, args, kwargs, *tensors)
            if len(x) > 2:
                x = list(zip(x[::2], x[1::2]))
        else:
            x = self._reverse(y, *args, **kwargs)
        return x

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        """
        This function implements gradient calculations for the invert to learn case. It will be called by
        InvertToLearn.backward. This function will be valid for any invertible function, however computation
        cost might be not optimal. Inheriting classes can overwrite this method to implement more efficient
        gradient computation specific for teh respective layer. See invertible_layer.py for examples.
        :param forward_fun: The function that was used during the forward operation
        :param reverse_fun: The inverse of forward_fun
        :param x: Tensor or list of Tensors, Input of layer
        :param y: Tensor or list of Tensors, Output of layer
        :param grad_outputs: Tensor or list of Tensors, gradients passed from higher layers
        :param parameters: Tensor or list of Tensors, parameters of the layer
        :return: x, grad_x, grads_param
        """
        assert not (x is None and y is None)
        if x is None:
            with torch.no_grad():
                x = reverse_fun(y, *args, **kwargs)

        with torch.enable_grad():
            if isinstance(x, list):
                x = [x_.detach().requires_grad_(True) for x_ in x]
                grad_tensors = x + parameters
            else:
                x = x.detach().requires_grad_(True)
                grad_tensors = [x] + parameters
            y = forward_fun(x, *args, **kwargs)
            grads = torch.autograd.grad(y, grad_tensors, grad_outputs=grad_outputs)

        if isinstance(x, list):
            grad_inputs = grads[: len(x)]
            grads_param = grads[len(x) :]
        else:
            grad_inputs = grads[0]
            grads_param = grads[1:]

        return x, grad_inputs, grads_param

    @abstractmethod
    def _forward(self, x, data, *args, **kwargs):
        pass

    @abstractmethod
    def _reverse(self, y, data, *args, **kwargs):
        pass


class IdentityLayer(InvertibleLayer):
    def _forward(self, x, *args, **kwargs):
        return x

    def _reverse(self, y, *args, **kwargs):
        return y

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        if x is None:
            x = y
        return x, grad_outputs, []


class MemoryFreeInvertibleModule(InvertibleModule):
    """
    A wrapper class that turns an invertible module into a module that utilizes invert to  learn during training,
    i.e. removing intermediate memory storage for back-propagation.
    """

    def __init__(self, model):
        """
        :param model: Model to be wrapped.
        """
        super().__init__()
        assert isinstance(model, InvertibleModule)
        self.model = model.apply(make_layer_memory_free)
        self.save_layer = IdentityLayer()
        make_layer_memory_free(self.save_layer, save_input=True)

    def forward(self, x, *args, **kwargs):
        x = [(x_, x_.detach()) for x_ in x] if isinstance(x, list) else (x, x.detach())
        x = self.model.forward(x, *args, **kwargs)
        x = self.save_layer.forward(x, None)
        x = [x_[0] for x_ in x] if isinstance(x, list) else x[0]
        return x

    def reverse(self, y, *args, **kwargs):
        y = [(y_, y_.detach()) for y_ in y] if isinstance(y, list) else (y, y.detach())
        y = self.model.reverse(y, *args, **kwargs)
        y = self.save_layer.reverse(y, None)
        y = [y_[0] for y_ in y] if isinstance(y, list) else y[0]
        return y
