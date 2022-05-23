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
    """Base class of all MRIReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

    # skipcq: PYL-R0201
    def process_loss(self, target, pred, _loss_fn):
        """
        Processes the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        target = torch.abs(target / torch.max(torch.abs(target)))
        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                return _loss_fn(
                    x.unsqueeze(dim=1),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=1),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return _loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        return loss_fn(target, pred)

    @staticmethod
    def process_inputs(y, mask):
        """
        Processes the inputs to the method.

        Parameters
        ----------
        y: Subsampled k-space data.
            list of torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            list of torch.Tensor, shape [1, 1, n_x, n_y, 1]

        Returns
        -------
        y: Subsampled k-space data.
            randomly selected y
        mask: Sampling mask.
            randomly selected mask
        r: Random index.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch: Batch of data.
            Dict[str, torch.Tensor], with keys,

            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        y, sensitivity_maps, mask, init_pred, target, _, _, acc = batch
        y, mask, r = self.process_inputs(y, mask)
        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass

            train_loss = sum(self.process_loss(target, preds, _loss_fn=self.train_loss_fn))
        else:
            train_loss = self.process_loss(target, preds, _loss_fn=self.train_loss_fn)

        acc = r if r != 0 else acc
        tensorboard_logs = {
            f"train_loss_{acc}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """
        Performs a validation step.

        Parameters
        ----------
        batch: Batch of data.
            Dict[str, torch.Tensor], with keys,

            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        Dict[str, torch.Tensor], with keys,
        'loss': loss,
            torch.Tensor, shape [1]
        'log': log,
            dict, shape [1]
        """
        y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass

            val_loss = sum(self.process_loss(target, preds, _loss_fn=self.eval_loss_fn))
        else:
            val_loss = self.process_loss(target, preds, _loss_fn=self.eval_loss_fn)

        # Cascades
        if isinstance(preds, list):
            preds = preds[-1]

        # Time-steps
        if isinstance(preds, list):
            preds = preds[-1]

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(preds).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return {"val_loss": val_loss}

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch: Batch of data.
            Dict[str, torch.Tensor], with keys,

            'y': subsampled kspace,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'sensitivity_maps': sensitivity_maps,
                torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            'mask': sampling mask,
                torch.Tensor, shape [1, 1, n_x, n_y, 1]
            'init_pred': initial prediction. For example zero-filled or PICS.
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'target': target data,
                torch.Tensor, shape [batch_size, n_x, n_y, 2]
            'fname': filename,
                str, shape [batch_size]
            'slice_idx': slice_idx,
                torch.Tensor, shape [batch_size]
            'acc': acceleration factor,
                torch.Tensor, shape [batch_size]
            'max_value': maximum value of the magnitude image space,
                torch.Tensor, shape [batch_size]
            'crop_size': crop size,
                torch.Tensor, shape [n_x, n_y]
        batch_idx: Batch index.
            int

        Returns
        -------
        name: Name of the volume.
            str
        slice_num: Slice number.
            int
        pred: Predicted data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        """
        y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        preds = self.forward(y, sensitivity_maps, mask, init_pred, target)

        if self.accumulate_estimates:
            try:
                preds = next(preds)
            except StopIteration:
                pass

        # Cascades
        if isinstance(preds, list):
            preds = preds[-1]

        # Time-steps
        if isinstance(preds, list):
            preds = preds[-1]

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore

        output = torch.abs(preds).detach().cpu()
        output = output / output.max()  # type: ignore

        target = torch.abs(target).detach().cpu()
        target = target / target.max()  # type: ignore

        error = torch.abs(target - output)

        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, preds.detach().cpu().numpy()

    def log_image(self, name, image):
        """
        Logs an image.

        Parameters
        ----------
        name: Name of the image.
            str
        image: Image to log.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        """
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the validation batches.
            list of dicts

        Returns
        -------
        metrics: Dictionary of metrics.
            dict
        """
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

    def test_epoch_end(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs.

        Parameters
        ----------
        outputs: List of outputs of the test batches.
            list of dicts

        Returns
        -------
        Saves the reconstructed images to .h5 files.
        """
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
        """
        Setups the training data.

        Parameters
        ----------
        train_data_config: Training data configuration.
            dict

        Returns
        -------
        train_data: Training data.
            torch.utils.data.DataLoader
        """
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        """
        Setups the validation data.

        Parameters
        ----------
        val_data_config: Validation data configuration.
            dict

        Returns
        -------
        val_data: Validation data.
            torch.utils.data.DataLoader
        """
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        """
        Setups the test data.

        Parameters
        ----------
        test_data_config: Test data configuration.
            dict

        Returns
        -------
        test_data: Test data.
            torch.utils.data.DataLoader
        """
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg: Configuration file.
            dict

        Returns
        -------
        dataloader: DataLoader.
            torch.utils.data.DataLoader
        """
        if cfg.get("dataset_type") != "FastMRI":
            raise ValueError(f"Unknown dataset type: {cfg.get('dataset_type')}")

        mask_args = cfg.get("mask_args")
        mask_type = mask_args.get("type")
        shift_mask = mask_args.get("shift_mask")

        if mask_type is not None and mask_type != "None":
            accelerations = mask_args.get("accelerations")
            center_fractions = mask_args.get("center_fractions")
            mask_center_scale = mask_args.get("scale")

            mask_func = (
                [
                    create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) > 2
                else [create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            )

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
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net to the coil images to estimate coil
    sensitivities.
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
        Initializes the model.

        Parameters
        ----------
        chans: Number of channels in the input k-space data.
            int
        num_pools: Number of U-Net downsampling/upsampling operations.
            int
        in_chans: Number of channels in the input data.
            int
        out_chans: Number of channels in the output data.
            int
        drop_prob: Dropout probability.
            float
        padding_size: Size of the zero-padding.
            int
        mask_type: Type of mask to use.
            str
        fft_type: Type of FFT to use.
            str
        normalize: Whether to normalize the input data.
            bool
        mask_center: Whether mask the center of the image.
            bool
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
        Converts the number of channels in a tensor to the batch dimension.

        Parameters
        ----------
        x: Tensor to convert.
            torch.Tensor

        Returns
        -------
        Tuple of the converted tensor and the original last dimension.
            Tuple[torch.Tensor, int]
        """
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    @staticmethod
    def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Converts the number of channels in a tensor to the channel dimension.

        Parameters
        ----------
        x: Tensor to convert.
            torch.Tensor
        batch_size: Original batch size.
            int

        Returns
        -------
        Converted tensor.
            torch.Tensor
        """
        bc, _, h, w, comp = x.shape
        c = torch.div(bc, batch_size, rounding_mode="trunc")

        return x.view(batch_size, c, h, w, comp)

    @staticmethod
    def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
        """
        Divide the input by the root of the sum of squares of the magnitude of each complex number.

        Parameters
        ----------
        x: Tensor to divide.
            torch.Tensor

        Returns
        -------
        RSS output tensor.
            torch.Tensor
        """
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    @staticmethod
    def get_pad_and_num_low_freqs(
        mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the padding to apply to the input to make it square and the number of low frequencies to keep.

        Parameters
        ----------
        mask: Mask to use.
            torch.Tensor
        num_low_frequencies: Number of low frequencies to keep.
            int

        Returns
        -------
        Tuple of the padding and the number of low frequencies to keep.
            Tuple[torch.Tensor, torch.Tensor]
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

        Parameters
        ----------
        masked_kspace: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 1]
        num_low_frequencies: Number of low frequencies to keep.
            int

        Returns
        -------
        Normalized UNet output tensor.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        """
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs, mask_type=self.mask_type)

        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        # estimate sensitivities
        images = self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        if self.normalize:
            images = self.divide_root_sum_of_squares(images)
        return images
