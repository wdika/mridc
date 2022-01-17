# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
import os
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.data.mri_data import FastMRISliceDataset
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.models.e2evn import SensitivityModel
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.parts.transforms import PhysicsInformedDataTransform
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck
from mridc.core.classes.modelPT import ModelPT
from mridc.utils.model_utils import convert_model_config_to_dict_config, maybe_update_config_version

__all__ = ["CIRIM"]


class CIRIM(ModelPT, ABC):
    """
    Cascades of Independently Recurrent Inference Machines implementation as presented in [1]_.

    References
    ----------

    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent
    Inference Machines for fast and robust accelerated MRI reconstruction’.
    Available at: https://arxiv.org/abs/2111.15498v1
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

        # Cascades of RIM blocks
        cirim_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.recurrent_filters = cirim_cfg_dict.get("recurrent_filters")

        # make time-steps size divisible by 8 for fast fp16 training
        self.time_steps = 8 * math.ceil(cirim_cfg_dict.get("time_steps") / 8)

        self.no_dc = cirim_cfg_dict.get("no_dc")
        self.fft_type = cirim_cfg_dict.get("fft_type")
        self.num_cascades = cirim_cfg_dict.get("num_cascades")

        self.cirim = nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=cirim_cfg_dict.get("recurrent_layer"),
                    conv_filters=cirim_cfg_dict.get("conv_filters"),
                    conv_kernels=cirim_cfg_dict.get("conv_kernels"),
                    conv_dilations=cirim_cfg_dict.get("conv_dilations"),
                    conv_bias=cirim_cfg_dict.get("conv_bias"),
                    recurrent_filters=self.recurrent_filters,
                    recurrent_kernels=cirim_cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cirim_cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cirim_cfg_dict.get("recurrent_bias"),
                    depth=cirim_cfg_dict.get("depth"),
                    time_steps=self.time_steps,
                    conv_dim=cirim_cfg_dict.get("conv_dim"),
                    no_dc=self.no_dc,
                    fft_type=self.fft_type,
                )
                for _ in range(self.num_cascades)
            ]
        )

        # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
        self.keep_eta = cirim_cfg_dict.get("keep_eta")
        self.output_type = cirim_cfg_dict.get("output_type")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cirim_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = SensitivityModel(
                cirim_cfg_dict.get("sens_chans"),
                cirim_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=cirim_cfg_dict.get("sens_mask_type"),
                normalize=cirim_cfg_dict.get("sens_normalize"),
            )

        std_init_range = 1 / self.recurrent_filters[0] ** 0.5

        # initialize weights if not using pretrained encoder
        if not cirim_cfg_dict.get("pretrained", False):
            self.cirim.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.train_loss_fn = SSIMLoss() if cirim_cfg_dict.get("loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cirim_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.accumulate_estimates = cirim_cfg_dict.get("accumulate_estimates")

        # Initialize data consistency term
        # TODO: make this configurable
        self.dc_weight = nn.Parameter(torch.ones(1))

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        target: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            eta: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for eta
            hx: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for hx
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
            sigma: float, noise level
        Returns:
             Final estimation of the network.
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        estimation = y.clone()

        cascades_etas = []
        for i, cascade in enumerate(self.cirim):
            # Forward pass through the cascades
            estimation, hx = cascade(
                estimation, y, sensitivity_maps, mask, eta, hx, sigma, keep_eta=False if i == 0 else self.keep_eta
            )

            if self.accumulate_estimates:
                time_steps_etas = [
                    self.process_intermediate_eta(pred, sensitivity_maps, target) for pred in estimation
                ]
                cascades_etas.append(time_steps_etas)

        if self.accumulate_estimates:
            yield cascades_etas
        else:
            return self.process_intermediate_eta(estimation, sensitivity_maps, target).detach()

    def process_intermediate_eta(self, eta, sensitivity_maps, target):
        """Process the intermediate eta to be used in the loss function."""
        if not self.no_dc:
            eta = ifft2c(eta, fft_type=self.fft_type)
            eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta

    def process_loss(self, target, eta, set_loss_fn):
        """Calculate the loss."""
        target = torch.abs(target / torch.max(torch.abs(target)))

        if "ssim" in str(set_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)
            loss_fn = lambda x, y: set_loss_fn(
                x.unsqueeze(dim=1),
                torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=1),
                data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
            )
        else:
            loss_fn = lambda x, y: set_loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        if self.accumulate_estimates:
            cascades_loss = []
            for cascade_eta in eta:
                time_steps_loss = [loss_fn(target, time_step_eta) for time_step_eta in cascade_eta]
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.time_steps).to(time_steps_loss[0]) for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.time_steps))
            yield sum(list(cascades_loss)) / len(self.cirim)
        else:
            return loss_fn(target, eta)

    @staticmethod
    def process_inputs(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for the CIRIM.

        Args:
            batch: A dictionary of the form {
                'y': subsampled kspace,
                'sensitivity_maps': sensitivity_maps,
                'mask': mask,
                'eta': initial estimation,
                'hx': hidden states,
                'target': target,
                'sigma': sigma
                }
            batch_idx: The index of the batch.

        Returns:
            A dictionary of the form {
                'loss': loss value,
                'log': log,
            }
        """
        y, sensitivity_maps, mask, _, target, _, _, acc, _, _ = batch
        y, mask, r = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        if self.accumulate_estimates:
            try:
                etas = next(etas)
            except StopIteration:
                pass

            train_loss = sum(self.process_loss(target, etas, set_loss_fn=self.train_loss_fn))
        else:
            train_loss = self.process_loss(target, etas, set_loss_fn=self.train_loss_fn)

        acc = r if r != 0 else acc

        tensorboard_logs = {
            f"train_loss_{str(acc)}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step for the CIRIM."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        if self.accumulate_estimates:
            try:
                etas = next(etas)
            except StopIteration:
                pass

            val_loss = sum(self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn))
        else:
            val_loss = self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn)

        if isinstance(etas, list):
            etas = etas[-1][-1]

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(etas).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return {
            "val_loss": val_loss,
        }

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for the CIRIM."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        if self.accumulate_estimates:
            try:
                etas = next(etas)
            except StopIteration:
                pass

        if isinstance(etas, list):
            etas = etas[-1][-1]

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore
        output = torch.abs(etas).detach().cpu()
        target = torch.abs(target).detach().cpu()
        _output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - _output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", _output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, output

    def log_image(self, name, image):
        """Log an image."""
        # TODO: Add support for wandb logging
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        """Validation epoch end for the the CIRIM. Called at the end of validation to aggregate outputs."""
        self.log("val_loss", torch.stack([x["val_loss"] for x in outputs]).mean())

    def test_epoch_end(self, outputs):
        """Test epoch end for the the CIRIM."""
        reconstructions = defaultdict(list)
        for fname, slice_num, output in outputs:
            reconstructions[fname].append((slice_num, output))

        for fname in reconstructions:
            reconstructions[fname] = np.abs(np.stack([out for _, out in sorted(reconstructions[fname])]))

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
