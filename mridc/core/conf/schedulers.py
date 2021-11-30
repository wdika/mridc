# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/config/schedulers.py

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional


@dataclass
class SchedulerParams:
    """Base configuration for all schedulers."""

    last_epoch: int = -1


@dataclass
class SquareRootConstantSchedulerParams(SchedulerParams):
    """
    Base configuration for all schedulers.
    It is not derived from Config as it is not a mridc object (and in particular it doesn't need a name).
    """

    constant_steps: Optional[float] = None
    constant_ratio: Optional[float] = None


@dataclass
class WarmupSchedulerParams(SchedulerParams):
    """Base configuration for all schedulers."""

    max_steps: int = 0
    warmup_steps: Optional[float] = None
    warmup_ratio: Optional[float] = None


@dataclass
class WarmupHoldSchedulerParams(WarmupSchedulerParams):
    """Base configuration for all schedulers."""

    hold_steps: Optional[float] = None
    hold_ratio: Optional[float] = None
    min_lr: float = 0.0


@dataclass
class WarmupAnnealingHoldSchedulerParams(WarmupSchedulerParams):
    """Base configuration for all schedulers."""

    constant_steps: Optional[float] = None
    constant_ratio: Optional[float] = None
    min_lr: float = 0.0


@dataclass
class SquareAnnealingParams(WarmupSchedulerParams):
    """Square Annealing parameter config"""

    min_lr: float = 1e-5


@dataclass
class SquareRootAnnealingParams(WarmupSchedulerParams):
    """Square Root Annealing parameter config"""

    min_lr: float = 0.0


@dataclass
class CosineAnnealingParams(WarmupAnnealingHoldSchedulerParams):
    """Cosine Annealing parameter config"""

    min_lr: float = 0.0


@dataclass
class NoamAnnealingParams(WarmupSchedulerParams):
    """Cosine Annealing parameter config"""

    min_lr: float = 0.0


@dataclass
class WarmupAnnealingParams(WarmupSchedulerParams):
    """Warmup Annealing parameter config"""

    warmup_ratio: Optional[float] = None


@dataclass
class InverseSquareRootAnnealingParams(WarmupSchedulerParams):
    """Inverse Square Root Annealing parameter config"""


@dataclass
class PolynomialDecayAnnealingParams(WarmupSchedulerParams):
    """Polynomial Decay Annealing parameter config"""

    power: float = 1.0
    cycle: bool = False


@dataclass
class PolynomialHoldDecayAnnealingParams(WarmupSchedulerParams):
    """Polynomial Hold Decay Annealing parameter config"""

    power: float = 1.0
    cycle: bool = False


@dataclass
class StepLRParams(SchedulerParams):
    """Config for StepLR."""

    step_size: float = 0.1
    gamma: float = 0.1


@dataclass
class ExponentialLRParams(SchedulerParams):
    """Config for ExponentialLR."""

    gamma: float = 0.9


@dataclass
class ReduceLROnPlateauParams:
    """Config for ReduceLROnPlateau."""

    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    verbose: bool = False
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8


@dataclass
class CyclicLRParams(SchedulerParams):
    """Config for CyclicLR."""

    base_lr: float = 0.001
    max_lr: float = 0.1
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"
    gamma: float = 1.0
    scale_mode: str = "cycle"
    # scale_fn is not supported
    cycle_momentum: bool = True
    base_momentum: float = 0.8
    max_momentum: float = 0.9


def register_scheduler_params(name: str, scheduler_params: SchedulerParams):
    """
    Checks if the scheduler config name exists in the registry, and if it doesn't, adds it.
    This allows custom schedulers to be added and called by name during instantiation.

    Parameters
    ----------
    name: Name of the optimizer. Will be used as key to retrieve the optimizer.
    scheduler_params: SchedulerParams class
    """
    if name in AVAILABLE_SCHEDULER_PARAMS:
        raise ValueError(f"Cannot override pre-existing optimizers. Conflicting optimizer name = {name}")

    AVAILABLE_SCHEDULER_PARAMS[name] = scheduler_params  # type: ignore


def get_scheduler_config(name: str, **kwargs: Optional[Dict[str, Any]]) -> partial:
    """
    Convenience method to obtain a SchedulerParams class and partially instantiate it with optimizer kwargs.

    Parameters
    ----------
    name: Name of the SchedulerParams in the registry.
    kwargs: Optional kwargs of the optimizer used during instantiation.

    Returns
    -------
    A partially instantiated SchedulerParams.
    """
    if name not in AVAILABLE_SCHEDULER_PARAMS:
        raise ValueError(
            f"Cannot resolve scheduler parameters '{name}'. Available scheduler parameters are : "
            f"{AVAILABLE_SCHEDULER_PARAMS.keys()}"
        )

    return partial(AVAILABLE_SCHEDULER_PARAMS[name], **kwargs)


AVAILABLE_SCHEDULER_PARAMS = {
    "SchedulerParams": SchedulerParams,
    "WarmupPolicyParams": WarmupSchedulerParams,
    "WarmupHoldPolicyParams": WarmupHoldSchedulerParams,
    "WarmupAnnealingHoldSchedulerParams": WarmupAnnealingHoldSchedulerParams,
    "SquareAnnealingParams": SquareAnnealingParams,
    "SquareRootAnnealingParams": SquareRootAnnealingParams,
    "InverseSquareRootAnnealingParams": InverseSquareRootAnnealingParams,
    "SquareRootConstantSchedulerParams": SquareRootConstantSchedulerParams,
    "CosineAnnealingParams": CosineAnnealingParams,
    "NoamAnnealingParams": NoamAnnealingParams,
    "WarmupAnnealingParams": WarmupAnnealingParams,
    "PolynomialDecayAnnealingParams": PolynomialDecayAnnealingParams,
    "PolynomialHoldDecayAnnealingParams": PolynomialHoldDecayAnnealingParams,
}
