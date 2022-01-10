# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from mridc.collections.recon.models.cirim import CIRIM
from mridc.core.conf.hydra_runner import hydra_runner
from mridc.utils import logging
from mridc.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="base_cirim_train")
def main(cfg: DictConfig) -> None:
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    cirim = CIRIM(cfg.model, trainer=trainer)

    if cfg.get("pretrained", None):
        checkpoint = cfg.get("checkpoint", None)
        logging.info(f"Loading pretrained model from {checkpoint}")
        cirim.load_state_dict(torch.load(checkpoint)["state_dict"])

    if cfg.get("mode", None) == "train":
        logging.info("Training")
        trainer.fit(cirim)
    else:
        logging.info("Evaluating")
        trainer.test(cirim)


if __name__ == "__main__":
    main()
