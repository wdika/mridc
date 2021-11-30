# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/config/optimizers.py

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

from omegaconf import MISSING, OmegaConf

__all__ = [
    "OptimizerParams",
    "AdamParams",
    "NovogradParams",
    "SGDParams",
    "AdadeltaParams",
    "AdamaxParams",
    "AdagradParams",
    "AdamWParams",
    "RMSpropParams",
    "RpropParams",
    "get_optimizer_config",
    "register_optimizer_params",
]


@dataclass
class OptimizerParams:
    """Base Optimizer params with no values. User can chose it to explicitly override via command line arguments."""

    lr: Optional[float] = MISSING


@dataclass
class SGDParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=sgd#torch.optim.SGD
    """

    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class AdamParams(OptimizerParams):
    """
    Default configuration for Adam optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam
    """

    # betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdamWParams(OptimizerParams):
    """
    Default configuration for AdamW optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class AdadeltaParams(OptimizerParams):
    """
    Default configuration for Adadelta optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta
    """

    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0


@dataclass
class AdamaxParams(OptimizerParams):
    """
    Default configuration for Adamax optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adamax
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdagradParams(OptimizerParams):
    """
    Default configuration for Adagrad optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adagrad
    """

    lr_decay: float = 0
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    eps: float = 1e-10


@dataclass
class RMSpropParams(OptimizerParams):
    """
    Default configuration for RMSprop optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
    """

    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class RpropParams(OptimizerParams):
    """
    Default configuration for RpropParams optimizer.

    .. note::
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/optim.html#torch.optim.Rprop
    """

    etas: Tuple[float, float] = (0.5, 1.2)
    step_sizes: Tuple[float, float] = (1e-6, 50)


@dataclass
class NovogradParams(OptimizerParams):
    """
    Configuration of the Novograd optimizer. It has been proposed in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks" (https://arxiv.org/abs/1905.11286). The OptimizerParams is a Base
    Optimizer params with no values. User can choose to explicitly override it via command line arguments.
    """

    betas: Tuple[float, float] = (0.95, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0
    grad_averaging: bool = False
    amsgrad: bool = False
    lr: float = 1e-3
    luc: bool = False
    luc_trust: float = 1e-3
    luc_eps: float = 1e-8


@dataclass
class AdafactorParams(OptimizerParams):
    """
    Configuration of the Adafactor optimizer.
    It has been proposed  in "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    (https://arxiv.org/abs/1804.04235)

    Parameters
    ----------
    lr: Learning rate.
         (float, optional), (default: 1e-3)
    beta1: Coefficients used for computing running averages of gradient and its square.
        (float, optional), (default: None)
    eps: Term added to the denominator to improve numerical stability.
        (Tuple [float, float] optional)
    weight_decay: Weight decay (L2 penalty).
        (float, optional), (default: 0)
    scale_parameter: Scale parameter.
        (float, optional), (default: False)
    relative_step: Whether to use relative step sizes.
        (bool, optional), (default: False)
    warmup_init: Whether to warm up the learning rate linearly.
        (bool, optional) (default: False)
    """

    beta1: Optional[float] = None
    eps: Tuple[float, float] = (1e-30, 1e-3)
    clip_threshold: float = 1.0
    decay_rate: float = 0.8
    weight_decay: float = 0
    scale_parameter: bool = True
    relative_step: bool = False
    warmup_init: bool = False


def register_optimizer_params(name: str, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer param name exists in the registry, and if it doesn't, adds it.
    This allows custom optimizer params to be added and called by name during instantiation.

    Parameters
    ----------
    name: Name of the optimizer. Will be used as key to retrieve the optimizer.
    optimizer_params: Optimizer class
    """
    if name in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZER_PARAMS[name] = optimizer_params  # type: ignore


def get_optimizer_config(
    name: str, **kwargs: Optional[Dict[str, Any]]
) -> Union[Dict[str, Optional[Dict[str, Any]]], partial]:
    """
    Convenience method to obtain a OptimizerParams class and partially instantiate it with optimizer kwargs.

    Parameters
    ----------
    name: Name of the OptimizerParams in the registry.
    kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns
    -------
    A partially instantiated OptimizerParams.
    """
    if name is None:
        return kwargs

    if name not in AVAILABLE_OPTIMIZER_PARAMS:
        raise ValueError(
            f"Cannot resolve optimizer parameters '{name}'. Available optimizer parameters are : "
            f"{AVAILABLE_OPTIMIZER_PARAMS.keys()}"
        )

    scheduler_params = AVAILABLE_OPTIMIZER_PARAMS[name]

    if kwargs is not None and kwargs:
        kwargs = OmegaConf.create(kwargs)
        OmegaConf.merge(scheduler_params(), kwargs)

    scheduler_params = partial(scheduler_params, **kwargs)  # type: ignore
    return scheduler_params  # type: ignore


AVAILABLE_OPTIMIZER_PARAMS = {
    "optim_params": OptimizerParams,
    "adam_params": AdamParams,
    "novograd_params": NovogradParams,
    "sgd_params": SGDParams,
    "adadelta_params": AdadeltaParams,
    "adamax_params": AdamaxParams,
    "adagrad_params": AdagradParams,
    "adamw_params": AdamWParams,
    "rmsprop_params": RMSpropParams,
    "rprop_params": RpropParams,
    "adafactor_params": AdafactorParams,
}
