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
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import rss_complex
from mridc.collections.reconstruction.data.mri_data import FastMRISliceDataset
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.transforms import MRIDataTransforms
from mridc.collections.reconstruction.parts.utils import batched_mask_center
from mridc.core.classes.modelPT import ModelPT
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["BaseMRIReconstructionModel", "BaseSensitivityModel"]


class BaseMRIReconstructionModel(ModelPT, ABC):
    """Base class for all MRIReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for the model."""

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step for the model."""

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for the model."""

    def log_image(self, name, image):
        """Log an image."""
        # TODO: Add support for wandb logging
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        """Validation epoch end. Called at the end of validation to aggregate outputs."""
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

    def test_epoch_end(self, outputs):
        """Test epoch end."""
        reconstructions = defaultdict(list)
        for fname, slice_num, output in outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])  # type: ignore

        out_dir = Path(os.path.join(self.logger.log_dir, "reconstructions"))
        out_dir.mkdir(exist_ok=True, parents=True)
        for fname, recons in reconstructions.items():
            with h5py.File(out_dir / fname, "w") as hf:
                hf.create_dataset("reconstruction", data=recons)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        """Setup the training data."""
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        """Setup the validation data."""
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

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
                transform=MRIDataTransforms(
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


class BaseSensitivityModel(nn.Module, ABC):
    """
    TODO: appears to be broken. Convert this to ModelPT ?
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net to the coil images to estimate coil
    sensitivities. It can be used with the end-to-end variational network.
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        mask_type: str = "2D",  # TODO: make this generalizable
        fft_type: str = "orthogonal",
        normalize: bool = True,
        mask_center: bool = True,
    ):
        """
        Initialize the model.

        Args:
            chans : Number of output channels of the first convolution layer.
            num_pools : Number of down-sampling and up-sampling layers.
            in_chan s: Number of channels in the input to the U-Net model.
            out_chans : Number of channels in the output to the U-Net model.
            drop_prob : Dropout probability.
            padding_size: Size of the zero-padding.
            mask_type : Type of mask to use.
            fft_type : Type of FFT to use.
            normalize : Whether to normalize the data.
            mask_center: Whether to mask the center of the sensitivity map.
        """
        super().__init__()

        self.mask_type = mask_type
        self.fft_type = fft_type

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            padding_size=padding_size,
            normalize=normalize,
        )

        self.mask_center = mask_center
        self.normalize = normalize

    @staticmethod
    def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert the last dimension of the input to the batch dimension.

        Args:
            x: Input tensor.

        Returns:
            Tuple of the converted tensor and the original last dimension.
        """
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    @staticmethod
    def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Convert the batch dimension of the input to the last dimension.

        Args:
            x: Input tensor.
            batch_size: Original batch size.

        Returns:
            Converted tensor.
        """
        bc, _, h, w, comp = x.shape
        c = torch.div(bc, batch_size, rounding_mode="trunc")

        return x.view(batch_size, c, h, w, comp)

    @staticmethod
    def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
        """
        Divide the input by the root of the sum of squares of the magnitude of each complex number.

        Args:
            x: Input tensor.

        Returns:
            RSS output tensor.
        """
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    @staticmethod
    def get_pad_and_num_low_freqs(
        mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the padding to apply to the input to make it square and the number of low frequencies to keep.

        Args:
            mask (): Mask to use.
            num_low_frequencies (): Number of low frequencies to keep. If None, keep all.

        Returns:
            Tuple of the padding and the number of low frequencies to keep.
        """
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = torch.div(squeezed_mask.shape[1], 2, rounding_mode="trunc")
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = torch.div(mask.shape[-2] - num_low_frequencies_tensor + 1, 2, rounding_mode="trunc")

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            masked_kspace: Masked k-space data.
            mask: Mask to apply to the k-space data.
            num_low_frequencies: Number of low frequencies to use.

        Returns:
            Normalized UNet output tensor.
        """
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs, mask_type=self.mask_type)

        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        # estimate sensitivities
        images = self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        return self.divide_root_sum_of_squares(images)
