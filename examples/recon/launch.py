# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from mridc.collections.reconstruction.models.cirim import CIRIM
from mridc.collections.reconstruction.models.pics import PICS
from mridc.collections.reconstruction.models.zf import ZF
from mridc.core.conf.hydra_runner import hydra_runner
from mridc.utils import logging
from mridc.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for training and running a model."""
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model_name = (cfg.model["model_name"]).upper()

    if model_name == "ZF":
        model = ZF(cfg.model, trainer=trainer)
    elif model_name == "PICS":
        model = PICS(cfg.model, trainer=trainer)
    elif model_name == "CIRIM":
        model = CIRIM(cfg.model, trainer=trainer)
    elif model_name in ("E2EVN", "UNET"):
        raise NotImplementedError(f"{model_name} is supported but not properly implemented yet.")
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")

    if cfg.get("pretrained", None):
        checkpoint = cfg.get("checkpoint", None)
        logging.info(f"Loading pretrained model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint)["state_dict"])

    if cfg.get("mode", None) == "train":
        logging.info("Training")
        trainer.fit(model)
    else:
        logging.info("Evaluating")
        trainer.test(model)


if __name__ == "__main__":
    main()
