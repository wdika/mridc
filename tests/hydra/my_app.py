# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/hydra/my_app.py

from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf

from mridc.core.conf.hydra_runner import hydra_runner


@dataclass
class DefaultConfig:
    """This is structured config for this application. It provides the schema used for validation of user-written \
    spec file as well as default values of the selected parameters."""

    dataset_name: str = MISSING


@hydra_runner(config_name="DefaultConfig", schema=DefaultConfig)
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    # Get dataset_name.
    dataset_name = cfg.dataset_name


if __name__ == "__main__":
    my_app()
