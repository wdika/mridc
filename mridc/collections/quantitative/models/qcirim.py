# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

import math
from abc import ABC
from typing import Generator, List, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from mridc.collections.common.parts.fft import fft2
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import complex_mul
from mridc.collections.quantitative.models.base import BaseqMRIReconstructionModel
from mridc.collections.quantitative.models.qrim.qrim_block import qRIMBlock
from mridc.collections.quantitative.models.qrim.utils import RescaleByMax, SignalForwardModel
from mridc.collections.quantitative.parts.transforms import R2star_B0_real_S0_complex_mapping
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.core.classes.common import typecheck

__all__ = ["qCIRIM"]


class qCIRIM(BaseqMRIReconstructionModel, ABC):
    """
    Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in \
    Karkalousos, D. et al.

    References
    ----------

    ..

        Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent \
        Inference Machines for fast and robust accelerated MRI reconstruction’. Available at: \
        https://arxiv.org/abs/2111.15498v1

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        quantitative_module_dimensionality = cfg_dict.get("quantitative_module_dimensionality")
        if quantitative_module_dimensionality != 2:
            raise ValueError(
                f"Only 2D is currently supported for qMRI models.Found {quantitative_module_dimensionality}"
            )

        quantitative_module_no_dc = cfg_dict.get("quantitative_module_no_dc")
        if not quantitative_module_no_dc:
            raise ValueError("qCIRIM does not support explicit DC component.")

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")
        self.shift_B0_input = cfg_dict.get("shift_B0_input")

        self.cirim = torch.nn.ModuleList([])

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
            self.reconstruction_module_time_steps = 8 * math.ceil(cfg_dict.get("reconstruction_module_time_steps") / 8)
            self.reconstruction_module_no_dc = cfg_dict.get("reconstruction_module_no_dc")
            self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")

            for _ in range(self.reconstruction_module_num_cascades):
                self.cirim.append(
                    RIMBlock(
                        recurrent_layer=cfg_dict.get("reconstruction_module_recurrent_layer"),
                        conv_filters=cfg_dict.get("reconstruction_module_conv_filters"),
                        conv_kernels=cfg_dict.get("reconstruction_module_conv_kernels"),
                        conv_dilations=cfg_dict.get("reconstruction_module_conv_dilations"),
                        conv_bias=cfg_dict.get("reconstruction_module_conv_bias"),
                        recurrent_filters=self.reconstruction_module_recurrent_filters,
                        recurrent_kernels=cfg_dict.get("reconstruction_module_recurrent_kernels"),
                        recurrent_dilations=cfg_dict.get("reconstruction_module_recurrent_dilations"),
                        recurrent_bias=cfg_dict.get("reconstruction_module_recurrent_bias"),
                        depth=cfg_dict.get("reconstruction_module_depth"),
                        time_steps=self.reconstruction_module_time_steps,
                        conv_dim=cfg_dict.get("reconstruction_module_conv_dim"),
                        no_dc=self.reconstruction_module_no_dc,
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim - 1,
                        dimensionality=cfg_dict.get("reconstruction_module_dimensionality"),
                    )
                )

            # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
            self.reconstruction_module_keep_eta = cfg_dict.get("reconstruction_module_keep_eta")

            # initialize weights if not using pretrained cirim
            if not cfg_dict.get("pretrained", False):
                std_init_range = 1 / self.reconstruction_module_recurrent_filters[0] ** 0.5
                self.cirim.apply(lambda module: rnn_weights_init(module, std_init_range))

            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.reconstruction_module_accumulate_estimates = cfg_dict.get(
                "reconstruction_module_accumulate_estimates"
            )

        quantitative_module_num_cascades = cfg_dict.get("quantitative_module_num_cascades")
        self.qcirim = torch.nn.ModuleList(
            [
                qRIMBlock(
                    recurrent_layer=cfg_dict.get("quantitative_module_recurrent_layer"),
                    conv_filters=cfg_dict.get("quantitative_module_conv_filters"),
                    conv_kernels=cfg_dict.get("quantitative_module_conv_kernels"),
                    conv_dilations=cfg_dict.get("quantitative_module_conv_dilations"),
                    conv_bias=cfg_dict.get("quantitative_module_conv_bias"),
                    recurrent_filters=cfg_dict.get("quantitative_module_recurrent_filters"),
                    recurrent_kernels=cfg_dict.get("quantitative_module_recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("quantitative_module_recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("quantitative_module_recurrent_bias"),
                    depth=cfg_dict.get("quantitative_module_depth"),
                    time_steps=cfg_dict.get("quantitative_module_time_steps"),
                    conv_dim=cfg_dict.get("quantitative_module_conv_dim"),
                    no_dc=cfg_dict.get("quantitative_module_no_dc"),
                    linear_forward_model=SignalForwardModel(
                        sequence=cfg_dict.get("quantitative_module_signal_forward_model_sequence")
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    coil_combination_method=self.coil_combination_method,
                    dimensionality=quantitative_module_dimensionality,
                )
                for _ in range(quantitative_module_num_cascades)
            ]
        )

        self.accumulate_estimates = cfg_dict.get("quantitative_module_accumulate_estimates")

        self.gamma = torch.tensor(cfg_dict.get("quantitative_module_gamma_regularization_factors"))
        self.preprocessor = RescaleByMax

    @typecheck()
    def forward(
        self,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask_brain: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        R2star_map_init: Initial R2* map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map_init: Initial S0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        B0_map_init: Initial B0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        phi_map_init: Initial phi map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        TEs: List of echo times.
            List of float, shape [n_echoes]
        y: Data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask_brain: Mask of the brain.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        sampling_mask: Mask of the sampling.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]

        Returns
        -------
        pred: list of list of torch.Tensor, shape [qmaps][batch_size, n_x, n_y, 2],
                or torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        if self.use_reconstruction_module:
            echoes_etas = []
            cascades_echoes_etas = []
            sigma = 1.0
            for echo in range(y.shape[1]):
                prediction = y[:, echo, ...].clone()
                init_pred = None
                hx = None
                cascades_etas = []
                for i, cascade in enumerate(self.cirim):
                    # Forward pass through the cascades
                    prediction, hx = cascade(
                        prediction,
                        y[:, echo, ...],
                        sensitivity_maps,
                        sampling_mask.squeeze(1),
                        init_pred,
                        hx,
                        sigma,
                        keep_eta=False if i == 0 else self.reconstruction_module_keep_eta,
                    )
                    cascades_etas.append([torch.view_as_complex(x) for x in prediction])
                cascades_echoes_etas.append(cascades_etas)
                echoes_etas.append(cascades_etas[-1][-1])

            eta = torch.stack(echoes_etas, dim=1)
            if eta.shape[-1] != 2:
                eta = torch.view_as_real(eta)
            y = fft2(
                complex_mul(eta.unsqueeze(self.coil_dim), sensitivity_maps.unsqueeze(self.coil_dim - 1)),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            for batch_idx in range(eta.shape[0]):
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_real_S0_complex_mapping(
                    eta[batch_idx],
                    TEs,
                    mask_brain,
                    torch.ones_like(mask_brain),
                    fully_sampled=True,
                    shift=self.shift_B0_input,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                R2star_maps_init.append(R2star_map_init.squeeze(0))
                S0_maps_init.append(S0_map_init.squeeze(0))
                B0_maps_init.append(B0_map_init.squeeze(0))
                phi_maps_init.append(phi_map_init.squeeze(0))
            R2star_map_init = torch.stack(R2star_maps_init, dim=0).to(y)
            S0_map_init = torch.stack(S0_maps_init, dim=0).to(y)
            B0_map_init = torch.stack(B0_maps_init, dim=0).to(y)
            phi_map_init = torch.stack(phi_maps_init, dim=0).to(y)

        R2star_map_pred = R2star_map_init / self.gamma[0]
        S0_map_pred = S0_map_init / self.gamma[1]
        B0_map_pred = B0_map_init / self.gamma[2]
        phi_map_pred = phi_map_init / self.gamma[3]

        prediction = y.clone()
        eta = None
        hx = None

        cascades_R2star_maps = []
        cascades_S0_maps = []
        cascades_B0_maps = []
        cascades_phi_maps = []
        for i, cascade in enumerate(self.qcirim):
            # Forward pass through the cascades
            prediction, hx = cascade(
                prediction,
                y,
                R2star_map_pred,
                S0_map_pred,
                B0_map_pred,
                phi_map_pred,
                TEs,
                sensitivity_maps,
                sampling_mask,
                eta,
                hx,
                self.gamma,
                keep_eta=i != 0,
            )
            R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                prediction[-1][:, 0],
                prediction[-1][:, 1],
                prediction[-1][:, 2],
                prediction[-1][:, 3],
            )
            if R2star_map_pred.shape[-1] == 2:
                R2star_map_pred = torch.view_as_complex(R2star_map_pred)
            if S0_map_pred.shape[-1] == 2:
                S0_map_pred = torch.view_as_complex(S0_map_pred)
            if B0_map_pred.shape[-1] == 2:
                B0_map_pred = torch.view_as_complex(B0_map_pred)
            if phi_map_pred.shape[-1] == 2:
                phi_map_pred = torch.view_as_complex(phi_map_pred)

            time_steps_R2star_maps = []
            time_steps_S0_maps = []
            time_steps_B0_maps = []
            time_steps_phi_maps = []
            for pred in prediction:
                _R2star_map_pred, _S0_map_pred, _B0_map_pred, _phi_map_pred = self.process_intermediate_pred(
                    torch.abs(pred), None, None, False
                )
                time_steps_R2star_maps.append(_R2star_map_pred)
                time_steps_S0_maps.append(_S0_map_pred)
                time_steps_B0_maps.append(_B0_map_pred)
                time_steps_phi_maps.append(_phi_map_pred)
            cascades_R2star_maps.append(time_steps_R2star_maps)
            cascades_S0_maps.append(time_steps_S0_maps)
            cascades_B0_maps.append(time_steps_B0_maps)
            cascades_phi_maps.append(time_steps_phi_maps)

        pred = cascades_echoes_etas if self.use_reconstruction_module else torch.empty([])

        yield [pred, cascades_R2star_maps, cascades_S0_maps, cascades_B0_maps, cascades_phi_maps]

    def process_intermediate_pred(self, pred, sensitivity_maps, target, do_coil_combination=False):
        """
        Process the intermediate prediction.

        Parameters
        ----------
        pred: Intermediate prediction.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        target: Target data to crop to size.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        do_coil_combination: Whether to do coil combination.
            bool, default False

        Returns
        -------
        pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        x = self.preprocessor.reverse(pred, self.gamma)
        R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
            x[:, 0, ...],
            x[:, 1, ...],
            x[:, 2, ...],
            x[:, 3, ...],
        )
        return R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred

    def process_quantitative_loss(self, target, pred, mask_brain, map, _loss_fn):
        """
        Processes the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        mask_brain: Mask for brain.
            torch.Tensor, shape [batch_size, n_x, n_y, 1]
        map: Type of map to regularize the loss.
            str in {"R2star", "S0", "B0", "phi"}
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        if "ssim" in str(_loss_fn).lower():

            def loss_fn(x, y, m):
                """Calculate the ssim loss."""
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                max_value = torch.max(torch.abs(y)) - torch.min(torch.abs(y)).unsqueeze(dim=0)
                m = torch.abs(m).to(x)

                loss = _loss_fn(x * m, y * m, data_range=max_value) * self.loss_regularization_factors[map]
                return loss

        else:

            def loss_fn(x, y, m):
                """Calculate other loss."""
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                m = torch.abs(m).to(x)

                if "mse" in str(_loss_fn).lower():
                    x = x.float()
                    y = y.float()
                    m = m.float()
                return _loss_fn(x * m, y * m) / self.loss_regularization_factors[map]

        if self.accumulate_estimates:
            cascades_loss = []
            for cascade_pred in pred:
                time_steps_loss = [loss_fn(target, time_step_pred, mask_brain) for time_step_pred in cascade_pred]
                cascades_loss.append(torch.sum(torch.stack(time_steps_loss, dim=0)) / len(pred))
            yield sum(cascades_loss) / len(self.qcirim)
        else:
            return loss_fn(target, pred, mask_brain)

    def process_reconstruction_loss(self, target, pred, _loss_fn=None):
        """
        Process the loss.

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
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(
                    x,
                    y,
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                x = torch.abs(x / torch.max(torch.abs(x)))
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(x, y)

        if self.reconstruction_module_accumulate_estimates:
            echoes_loss = []
            for echo_time in range(len(pred)):
                cascades_loss = []
                for cascade_pred in pred[echo_time]:
                    time_steps_loss = [
                        loss_fn(target[:, echo_time, :, :], time_step_pred).mean() for time_step_pred in cascade_pred
                    ]
                    _loss = [
                        x * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(time_steps_loss[0])
                        for x in time_steps_loss
                    ]
                    cascades_loss.append(sum(sum(_loss) / self.reconstruction_module_time_steps))
                echoes_loss.append(sum(list(cascades_loss)) / len(self.cirim))
            yield sum(echoes_loss) / len(pred)
        else:
            return loss_fn(target, pred)
