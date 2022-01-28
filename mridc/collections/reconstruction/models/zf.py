# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import check_stacked_complex, coil_combination
from mridc.collections.reconstruction.data.mri_data import FastMRISliceDataset
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.parts.transforms import PhysicsInformedDataTransform
from mridc.core.classes.common import typecheck
from mridc.core.classes.modelPT import ModelPT
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["ZF"]


class ZF(ModelPT, ABC):
    """
    Zero-Filled reconstruction using either root-sum-of-squares (RSS) or SENSE (SENSitivity Encoding) [1].

    References
    ----------

    .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI.
    Magn Reson Med 1999; 42:952-962.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cirim_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.zf_method = cirim_cfg_dict.get("zf_method")
        self.fft_type = cirim_cfg_dict.get("fft_type")

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Zero-Filling.

        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
        Returns:
            torch.Tensor, shape [batch_size, n_x, n_y], reconstructed data
        """
        pred = coil_combination(
            ifft2c(y, fft_type=self.fft_type), sensitivity_maps, method=self.zf_method.upper(), dim=1
        )
        pred = check_stacked_complex(pred)
        return pred

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for PICS."""
        y, sensitivity_maps, _, _, target, fname, slice_num, _, _, _ = batch
        prediction = self.forward(y, sensitivity_maps)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        output = torch.abs(prediction).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, prediction.detach().cpu().numpy()

    def log_image(self, name, image):
        """Log an image."""
        # TODO: Add support for wandb logging
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def test_epoch_end(self, outputs):
        """Test epoch end for the the CIRIM."""
        reconstructions = defaultdict(list)
        for fname, slice_num, output in outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])

        out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)
        for fname, recons in reconstructions.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("reconstruction", data=recons)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        """Pass the setup of the training data."""

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        """Pass the setup of the validation data."""

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        """Setup the test data."""
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """Setup the dataloader from the config."""
        if cfg.get("dataset_type") == "FastMRI":

            mask_args = cfg.get("mask_args")
            mask_type = mask_args.get("type")
            shift_mask = mask_args.get("shift_mask")

            if mask_type is not None and mask_type != "None":
                accelerations = mask_args.get("accelerations")
                center_fractions = mask_args.get("center_fractions")
                mask_center_scale = mask_args.get("scale")

                if len(accelerations) > 2:
                    mask_func = [
                        create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                        for acc, cf in zip(accelerations, center_fractions)
                    ]
                else:
                    mask_func = [create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            else:
                mask_func = None  # type: ignore
                mask_center_scale = 0.02

            dataset = FastMRISliceDataset(
                root=cfg.get("data_path"),
                sense_root=cfg.get("sense_data_path"),
                challenge=cfg.get("challenge"),
                transform=PhysicsInformedDataTransform(
                    mask_func=mask_func,
                    shift_mask=shift_mask,
                    mask_center_scale=mask_center_scale,
                    normalize_inputs=cfg.get("normalize_inputs"),
                    crop_size=cfg.get("crop_size"),
                    crop_before_masking=cfg.get("crop_before_masking"),
                    kspace_zero_filling_size=cfg.get("kspace_zero_filling_size"),
                    fft_type=cfg.get("fft_type"),
                    use_seed=cfg.get("use_seed"),
                ),
                sample_rate=cfg.get("sample_rate"),
            )
        else:
            raise ValueError(f"Unknown dataset type: {cfg.get('dataset_type')}")

        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )
