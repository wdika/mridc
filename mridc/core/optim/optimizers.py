# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/optimizers.py

import copy
from functools import partial
from typing import Any, Dict, Optional, Union

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer

from mridc.core.conf.optimizers import OptimizerParams, get_optimizer_config, register_optimizer_params
from mridc.core.optim.adafactor import Adafactor
from mridc.core.optim.novograd import Novograd
from mridc.utils.model_utils import maybe_update_config_version

AVAILABLE_OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adadelta": adadelta.Adadelta,
    "adamax": adamax.Adamax,
    "adagrad": adagrad.Adagrad,
    "rmsprop": rmsprop.RMSprop,
    "rprop": rprop.Rprop,
    "novograd": Novograd,
    "adafactor": Adafactor,
}

__all__ = ["AVAILABLE_OPTIMIZERS", "get_optimizer", "register_optimizer", "parse_optimizer_args"]


def parse_optimizer_args(
    optimizer_name: str, optimizer_kwargs: Union[DictConfig, Dict[str, Any]]
) -> Union[Dict[str, Any], DictConfig]:
    """
    Parses a list of strings, of the format "key=value" or "key2=val1,val2,..."
    into a dictionary of type {key=value, key2=[val1, val2], ...}
    This dictionary is then used to instantiate the chosen Optimizer.

    Parameters
    ----------
    optimizer_name: string name of the optimizer, used for auto resolution of params.
    optimizer_kwargs: Either a list of strings in a specified format, or a dictionary. If a dictionary is provided, it
    is assumed the dictionary is the final parsed value, and simply returned. If a list of strings is provided, each
    item in the list is parsed into a new dictionary.

    Returns
    -------
    A dictionary of the parsed arguments.
    """
    kwargs: Dict[Any, Any] = {}

    if optimizer_kwargs is None:
        return kwargs

    optimizer_kwargs = copy.deepcopy(optimizer_kwargs)
    optimizer_kwargs = maybe_update_config_version(optimizer_kwargs)

    if isinstance(optimizer_kwargs, DictConfig):
        optimizer_kwargs = OmegaConf.to_container(optimizer_kwargs, resolve=True)

    # If it is a dictionary, perform stepwise resolution
    if hasattr(optimizer_kwargs, "keys"):
        # Attempt class path resolution
        if "_target_" in optimizer_kwargs:  # captures (target, _target_)
            optimizer_kwargs_config = OmegaConf.create(optimizer_kwargs)
            optimizer_instance = hydra.utils.instantiate(optimizer_kwargs_config)  # type: DictConfig
            optimizer_instance = vars(optimizer_instance)  # type: ignore
            return optimizer_instance

        # If class path was not provided, perhaps `name` is provided for resolution
        if "name" in optimizer_kwargs:
            # If `auto` is passed as name for resolution of optimizer name,
            # then lookup optimizer name and resolve its parameter config
            if optimizer_kwargs["name"] == "auto":
                optimizer_params_name = f"{optimizer_name}_params"
                optimizer_kwargs.pop("name")
            else:
                optimizer_params_name = optimizer_kwargs.pop("name")

            # Override arguments provided in the config yaml file
            if "params" in optimizer_kwargs:
                # If optimizer kwarg overrides are wrapped in yaml `params`
                optimizer_params_override = optimizer_kwargs.get("params")
            else:
                # If the kwargs themselves are a DictConfig
                optimizer_params_override = optimizer_kwargs

            if isinstance(optimizer_params_override, DictConfig):
                optimizer_params_override = OmegaConf.to_container(optimizer_params_override, resolve=True)

            optimizer_params_cls = get_optimizer_config(optimizer_params_name, **optimizer_params_override)

            # If we are provided just a Config object, simply return the dictionary of that object
            if optimizer_params_name is None:
                optimizer_params = vars(optimizer_params_cls)
                return optimizer_params
            # If we are provided a partial class instantiation of a Config, instantiate it and retrieve its vars
            # as a dictionary.
            optimizer_params = vars(optimizer_params_cls)  # instantiate the parameters object
            return optimizer_params

        # simply return the dictionary that was provided
        return optimizer_kwargs

    return kwargs


def register_optimizer(name: str, optimizer: Optimizer, optimizer_params: OptimizerParams):
    """
    Checks if the optimizer name exists in the registry, and if it doesn't, adds it.
    This allows custom optimizers to be added and called by name during instantiation.

    Parameters
    ----------
    name: Name of the optimizer. Will be used as key to retrieve the optimizer.
    optimizer: Optimizer class.
    optimizer_params: The parameters as a dataclass of the optimizer.
    """
    if name in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_OPTIMIZERS[name] = optimizer

    optim_name = f"{optimizer.__name__}_params"
    register_optimizer_params(name=optim_name, optimizer_params=optimizer_params)


def get_optimizer(name: str, **kwargs: Optional[Dict[str, Any]]) -> partial:
    """
    Convenience method to obtain an Optimizer class and partially instantiate it with optimizer kwargs.

    Parameters
    ----------
    name: Name of the Optimizer in the registry.
    kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns
    -------
    A partially instantiated Optimizer.
    """
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Cannot resolve optimizer '{name}'. Available optimizers are : " f"{AVAILABLE_OPTIMIZERS.keys()}"
        )
    if name == "fused_adam" and not torch.cuda.is_available():
        raise ValueError("CUDA must be available to use fused_adam.")

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer
