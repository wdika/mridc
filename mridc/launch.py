# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from mridc.collections.multitask.models import MTLMRIRS
from mridc.collections.quantitative.models.qcirim import qCIRIM
from mridc.collections.quantitative.models.qvn import qVarNet
from mridc.collections.reconstruction.models.ccnn import CascadeNet
from mridc.collections.reconstruction.models.cirim import CIRIM
from mridc.collections.reconstruction.models.crnn import CRNNet
from mridc.collections.reconstruction.models.dunet import DUNet
from mridc.collections.reconstruction.models.jointicnet import JointICNet
from mridc.collections.reconstruction.models.kikinet import KIKINet
from mridc.collections.reconstruction.models.lpd import LPDNet
from mridc.collections.reconstruction.models.multidomainnet import MultiDomainNet
from mridc.collections.reconstruction.models.pics import PICS
from mridc.collections.reconstruction.models.rvn import RecurrentVarNet
from mridc.collections.reconstruction.models.unet import UNet
from mridc.collections.reconstruction.models.vn import VarNet
from mridc.collections.reconstruction.models.vsnet import VSNet
from mridc.collections.reconstruction.models.xpdnet import XPDNet
from mridc.collections.reconstruction.models.zf import ZF
from mridc.collections.segmentation.models.attention_unet import SegmentationAttentionUNet
from mridc.collections.segmentation.models.dynunet import DYNUNet
from mridc.collections.segmentation.models.idslr import IDSLR
from mridc.collections.segmentation.models.idslr_unet import IDSLRUNET
from mridc.collections.segmentation.models.jrscirim import JRSCIRIM
from mridc.collections.segmentation.models.lambda_unet import SegmentationLambdaUNet
from mridc.collections.segmentation.models.recseg_unet import RecSegUNet
from mridc.collections.segmentation.models.segnet import SegNet
from mridc.collections.segmentation.models.seranet import SERANet
from mridc.collections.segmentation.models.unet import SegmentationUNet
from mridc.collections.segmentation.models.unet3d import Segmentation3DUNet
from mridc.collections.segmentation.models.unetr import SegmentationUNetR
from mridc.collections.segmentation.models.vnet import SegmentationVNet
from mridc.core.conf.hydra_runner import hydra_runner
from mridc.utils import logging
from mridc.utils.exp_manager import exp_manager


@hydra_runner(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for training and running a model

    Parameters
    ----------
    cfg: Configuration (yaml) file.
        DictConfig
    """
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
    elif model_name == "DYNUNET":
        model = DYNUNet(cfg.model, trainer=trainer)
    elif model_name in ("E2EVN", "VN"):
        model = VarNet(cfg.model, trainer=trainer)
    elif model_name == "IDSLR":
        model = IDSLR(cfg.model, trainer=trainer)
    elif model_name == "IDSLRUNET":
        model = IDSLRUNET(cfg.model, trainer=trainer)
    elif model_name == "JOINTICNET":
        model = JointICNet(cfg.model, trainer=trainer)
    elif model_name == "JRSCIRIM":
        model = JRSCIRIM(cfg.model, trainer=trainer)
    elif model_name == "KIKINET":
        model = KIKINet(cfg.model, trainer=trainer)
    elif model_name == "LPDNET":
        model = LPDNet(cfg.model, trainer=trainer)
    elif model_name == "MTLMRIRS":
        model = MTLMRIRS(cfg.model, trainer=trainer)
    elif model_name == "MULTIDOMAINNET":
        model = MultiDomainNet(cfg.model, trainer=trainer)
    elif model_name == "PICS":
        model = PICS(cfg.model, trainer=trainer)
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
            "CASCADENET, CIRIM, CRNNET, DUNET, E2EVN, IDSLR, JOINTICNET, JRSCIRIM, KIKINET, LPDNET, MTLMRIRS, "
            "MULTIDOMAINNET, PICS, qCIRIM, qVN, RECSEGNET, RVN, SEGMENTATIONATTENTIONUNET, SEGMENTATIONLAMBDAUNET, "
            "SEGMENTATIONUNET, SEGMENTATION3DUNET, SEGMENTATIONVNET, SEGNET, UNET, VSNET, XPDNET, or Zero-Filled. /n"
            "If you implemented a new model, please consider adding it through a PR on GitHub."
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
