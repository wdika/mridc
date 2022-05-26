# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import is_none, rss_complex, sense
from mridc.collections.reconstruction.data.mri_data import FastMRISliceDataset
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.metrics.evaluate import mse, nmse, psnr, ssim
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.transforms import MRIDataTransforms
from mridc.collections.reconstruction.parts.utils import batched_mask_center
from mridc.core.classes.modelPT import ModelPT
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["BaseMRIReconstructionModel", "BaseSensitivityModel"]


class DistributedMetricSum(Metric):
    """
    A metric that sums the values of a metric across all workers.
    Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
    """

    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        """Update the metric with a batch of data."""
        self.quantity += batch

    def compute(self):
        """Compute the metric value."""
        return self.quantity


class BaseMRIReconstructionModel(ModelPT, ABC):
    """Base class of all MRIReconstruction models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = convert_model_config_to_dict_config(cfg)
        cfg = maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
                mask_center=cfg_dict.get("sens_mask_center"),
            )

        self.MSE = DistributedMetricSum()
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()

        # Set evaluation metrics dictionaries
        self.mse_vals: Dict = defaultdict(dict)
        self.nmse_vals: Dict = defaultdict(dict)
        self.ssim_vals: Dict = defaultdict(dict)
        self.psnr_vals: Dict = defaultdict(dict)

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
                    x.unsqueeze(dim=self.coil_dim),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return _loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        return loss_fn(target, pred)

    @staticmethod
    def process_inputs(y, mask, init_pred):
        """
        Processes the inputs to the method.

        Parameters
        ----------
        y: Subsampled k-space data.
            list of torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            list of torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        y: Subsampled k-space data.
            randomly selected y
        mask: Sampling mask.
            randomly selected mask
        init_pred: Initial prediction.
            randomly selected init_pred
        r: Random index.
        """
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, init_pred, r

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
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
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
        kspace, y, sensitivity_maps, mask, init_pred, target, _, _, acc = batch
        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)
            if self.coil_combination_method.upper() == "SENSE":
                target = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_maps,
                    dim=self.coil_dim,
                )

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
        batch: Batch of data. Dict[str, torch.Tensor], with keys,
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
            'phase_shift': phase shift for simulated motion,
                torch.Tensord
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
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)
            if self.coil_combination_method.upper() == "SENSE":
                target = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_maps,
                    dim=self.coil_dim,
                )

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

        target = target.numpy()  # type: ignore
        output = output.numpy()  # type: ignore
        self.mse_vals[fname][slice_num] = torch.tensor(mse(target, output)).view(1)
        self.nmse_vals[fname][slice_num] = torch.tensor(nmse(target, output)).view(1)
        self.ssim_vals[fname][slice_num] = torch.tensor(ssim(target, output, maxval=output.max() - output.min())).view(
            1
        )
        self.psnr_vals[fname][slice_num] = torch.tensor(psnr(target, output, maxval=output.max() - output.min())).view(
            1
        )

        return {"val_loss": val_loss}

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch: Batch of data. Dict[str, torch.Tensor], with keys,
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
            'phase_shift': phase shift for simulated motion,
                torch.Tensor
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
        kspace, y, sensitivity_maps, mask, init_pred, target, fname, slice_num, _ = batch
        y, mask, init_pred, r = self.process_inputs(y, mask, init_pred)

        if self.use_sens_net:
            sensitivity_maps = self.sens_net(kspace, mask)
            if self.coil_combination_method.upper() == "SENSE":
                target = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_maps,
                    dim=self.coil_dim,
                )

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
        if image.dim() > 3:
            image = image[0, 0, :, :].unsqueeze(0)
        elif image.shape[0] != 1:
            image = image[0].unsqueeze(0)

        if "wandb" in self.logger.__module__.lower():
            self.logger.experiment.log({name: wandb.Image(image.numpy())})
        else:
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

        # Log metrics.
        # Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py
        mse_vals = defaultdict(dict)
        nmse_vals = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        psnr_vals = defaultdict(dict)

        for k in self.mse_vals.keys():
            mse_vals[k].update(self.mse_vals[k])
        for k in self.nmse_vals.keys():
            nmse_vals[k].update(self.nmse_vals[k])
        for k in self.ssim_vals.keys():
            ssim_vals[k].update(self.ssim_vals[k])
        for k in self.psnr_vals.keys():
            psnr_vals[k].update(self.psnr_vals[k])

        # apply means across image volumes
        metrics = {"MSE": 0, "NMSE": 0, "SSIM": 0, "PSNR": 0}
        local_examples = 0
        for fname in mse_vals:
            local_examples += 1
            metrics["MSE"] = metrics["MSE"] + torch.mean(torch.cat([v.view(-1) for _, v in mse_vals[fname].items()]))
            metrics["NMSE"] = metrics["NMSE"] + torch.mean(
                torch.cat([v.view(-1) for _, v in nmse_vals[fname].items()])
            )
            metrics["SSIM"] = metrics["SSIM"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )
            metrics["PSNR"] = metrics["PSNR"] + torch.mean(
                torch.cat([v.view(-1) for _, v in psnr_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["MSE"] = self.MSE(metrics["MSE"])
        metrics["NMSE"] = self.NMSE(metrics["NMSE"])
        metrics["SSIM"] = self.SSIM(metrics["SSIM"])
        metrics["PSNR"] = self.PSNR(metrics["PSNR"])

        tot_examples = self.TotExamples(torch.tensor(local_examples))
        for metric, value in metrics.items():
            self.log(f"{metric}", value / tot_examples)

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
        mask_root = cfg.get("mask_path")
        mask_args = cfg.get("mask_args")
        shift_mask = mask_args.get("shift_mask")
        mask_type = mask_args.get("type")

        mask_func = None  # type: ignore
        mask_center_scale = 0.02

        if is_none(mask_root) and not is_none(mask_type):
            accelerations = mask_args.get("accelerations")
            center_fractions = mask_args.get("center_fractions")
            mask_center_scale = mask_args.get("scale")

            mask_func = (
                [
                    create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) >= 2
                else [create_mask_for_mask_type(mask_type, center_fractions, accelerations)]
            )

        dataset = FastMRISliceDataset(
            root=cfg.get("data_path"),
            sense_root=cfg.get("sense_path"),
            mask_root=cfg.get("mask_path"),
            challenge=cfg.get("challenge"),
            transform=MRIDataTransforms(
                coil_combination_method=cfg.get("coil_combination_method"),
                dimensionality=cfg.get("dimensionality"),
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=mask_center_scale,
                remask=cfg.get("remask"),
                normalize_inputs=cfg.get("normalize_inputs"),
                crop_size=cfg.get("crop_size"),
                crop_before_masking=cfg.get("crop_before_masking"),
                kspace_zero_filling_size=cfg.get("kspace_zero_filling_size"),
                fft_centered=cfg.get("fft_centered"),
                fft_normalization=cfg.get("fft_normalization"),
                max_norm=cfg.get("max_norm"),
                spatial_dims=cfg.get("spatial_dims"),
                coil_dim=cfg.get("coil_dim"),
                use_seed=cfg.get("use_seed"),
            ),
            sample_rate=cfg.get("sample_rate"),
            consecutive_slices=cfg.get("consecutive_slices"),
        )
        if cfg.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.get("batch_size"),
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
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 1,
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
        fft_centered: Whether to center the FFT.
            bool
        fft_normalization: Type of FFT normalization to use.
            str
        spatial_dims: Spatial dimensions of the data.
            tuple
        coil_dim: Coil dimension.
            int
        normalize: Whether to normalize the input data.
            bool
        mask_center: Whether mask the center of the image.
            bool
        """
        super().__init__()

        self.mask_type = mask_type

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
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
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
    def divide_root_sum_of_squares(x: torch.Tensor, coil_dim: int) -> torch.Tensor:
        """
        Divide the input by the root of the sum of squares of the magnitude of each complex number.

        Parameters
        ----------
        x: Tensor to divide.
            torch.Tensor
        coil_dim: Coil dimension.
            int

        Returns
        -------
        RSS output tensor.
            torch.Tensor
        """
        return x / rss_complex(x, dim=coil_dim).unsqueeze(-1).unsqueeze(coil_dim)

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
        images, batches = self.chans_to_batch_dim(
            ifft2(
                masked_kspace,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        )

        # estimate sensitivities
        images = self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        if self.normalize:
            images = self.divide_root_sum_of_squares(images, self.coil_dim)
        return images
