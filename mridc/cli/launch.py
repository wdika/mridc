# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from mridc.collections.multitask.rs.models import IDSLR, IDSLRUNet, MTLRS, RecSegUNet, SegNet, SERANet
from mridc.collections.quantitative.models import qCIRIM, qVarNet
from mridc.collections.reconstruction.models import (
    CIRIM,
    CascadeNet,
    CRNNet,
    DUNet,
    JointICNet,
    KIKINet,
    LPDNet,
    MultiDomainNet,
    RecurrentVarNet,
    UNet,
    VarNet,
    VSNet,
    XPDNet,
    ZF,
)
from mridc.collections.segmentation.models import (
    SegmentationAttentionUNet,
    SegmentationDYNUNet,
    SegmentationLambdaUNet,
    SegmentationUNet,
    SegmentationUNetR,
    Segmentation3DUNet,
    SegmentationVNet,
)
from mridc.core.conf.hydra_runner import hydra_runner
from mridc.utils import logging
from mridc.utils.exp_manager import exp_manager


def register_parser(parser: argparse._SubParsersAction):
    """Register parser for the launch command."""
    parser_launch = parser.add_parser(
        "run",
        help="Launch MRIDC through cli given a configuration (yaml) file, e.g. mridc run -c /path/to/config.yaml",
    )
    parser_launch.add_argument(
        "-c",
        "--config-path",
        required=True,
        type=str,
        help="Path to the configuration file.",
    )
    parser_launch.set_defaults(func=main)


@hydra_runner(config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for training and running a model

    Parameters
    ----------
    cfg: Configuration (yaml) file.
        DictConfig
    """
    cfg = OmegaConf.load(f"{cfg.config_path}")

    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model_name = (cfg.model["model_name"]).upper()

    if model_name == "CASCADENET":
        model = CascadeNet(cfg.model, trainer=trainer)
    elif model_name == "CIRIM":
        model = CIRIM(cfg.model, trainer=trainer)
    elif model_name == "CRNNET":
        model = CRNNet(cfg.model, trainer=trainer)
    elif model_name == "DUNET":
        model = DUNet(cfg.model, trainer=trainer)
    elif model_name in ("E2EVN", "VN"):
        model = VarNet(cfg.model, trainer=trainer)
    elif model_name == "IDSLR":
        model = IDSLR(cfg.model, trainer=trainer)
    elif model_name == "IDSLRUNET":
        model = IDSLRUNet(cfg.model, trainer=trainer)
    elif model_name == "JOINTICNET":
        model = JointICNet(cfg.model, trainer=trainer)
    elif model_name == "MTLRS":
        model = MTLRS(cfg.model, trainer=trainer)
    elif model_name == "KIKINET":
        model = KIKINet(cfg.model, trainer=trainer)
    elif model_name == "LPDNET":
        model = LPDNet(cfg.model, trainer=trainer)
    elif model_name == "MULTIDOMAINNET":
        model = MultiDomainNet(cfg.model, trainer=trainer)
    elif model_name == "QCIRIM":
        model = qCIRIM(cfg.model, trainer=trainer)
    elif model_name == "QVN":
        model = qVarNet(cfg.model, trainer=trainer)
    elif model_name == "RECSEGNET":
        model = RecSegUNet(cfg.model, trainer=trainer)
    elif model_name == "RVN":
        model = RecurrentVarNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONATTENTIONUNET":
        model = SegmentationAttentionUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONDYNUNET":
        model = SegmentationDYNUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONLAMBDAUNET":
        model = SegmentationLambdaUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONUNET":
        model = SegmentationUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONUNETR":
        model = SegmentationUNetR(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATION3DUNET":
        model = Segmentation3DUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONVNET":
        model = SegmentationVNet(cfg.model, trainer=trainer)
    elif model_name == "SEGNET":
        model = SegNet(cfg.model, trainer=trainer)
    elif model_name == "SERANET":
        model = SERANet(cfg.model, trainer=trainer)
    elif model_name == "UNET":
        model = UNet(cfg.model, trainer=trainer)
    elif model_name == "VSNET":
        model = VSNet(cfg.model, trainer=trainer)
    elif model_name == "XPDNET":
        model = XPDNet(cfg.model, trainer=trainer)
    elif model_name == "ZF":
        model = ZF(cfg.model, trainer=trainer)
    else:
        raise NotImplementedError(
            f"{model_name} is not implemented in MRIDC. You can use one of the following methods: "
            "CASCADENET, CIRIM, CRNNET, DUNET, E2EVN, IDSLR, JOINTICNET, MTLRS, KIKINET, LPDNET, MULTIDOMAINNET, "
            "qCIRIM, qVN, RECSEGNET, RVN, SEGMENTATIONATTENTIONUNET, SEGMENTATIONLAMBDAUNET, SEGMENTATIONUNET, "
            "SEGMENTATION3DUNET, SEGMENTATIONVNET, SEGNET, UNET, VSNET, XPDNET, or Zero-Filled. /n"
            "If you are interested in another model, please consider opening an issue on GitHub."
        )

    if cfg.get("pretrained", None):
        checkpoint = cfg.get("checkpoint", None)
        logging.info(f"Loading pretrained model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])

    if cfg.get("mode", None) == "train":
        logging.info("Validating")
        trainer.validate(model)
        logging.info("Training")
        trainer.fit(model)
    else:
        logging.info("Evaluating")
        trainer.test(model)


if __name__ == "__main__":
    main()
