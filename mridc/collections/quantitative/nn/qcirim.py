# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Dict, Generator, List, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.quantitative.nn.base as base_quantitative_models
import mridc.collections.quantitative.nn.qrim.utils as qrim_utils
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.quantitative.nn.qrim import qrim_block
from mridc.collections.quantitative.parts import transforms
from mridc.collections.reconstruction.nn.rim import rim_block

__all__ = ["qCIRIM"]


class qCIRIM(base_quantitative_models.BaseqMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the quantitative Recurrent Inference Machines (qRIM), as presented in [1].

    Also implements the qCIRIM model, which is a qRIM model with cascades.

    References
    ----------
    .. [1] Zhang C, Karkalousos D, Bazin PL, Coolen BF, Vrenken H, Sonke JJ, Forstmann BU, Poot DH, Caan MW. A unified
        model for reconstruction and R2* mapping of accelerated 7T data using the quantitative recurrent inference
        machine. NeuroImage. 2022 Dec 1;264:119680.
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
                    rim_block.RIMBlock(
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
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim - 1,
                        dimensionality=cfg_dict.get("reconstruction_module_dimensionality"),
                    )
                )

            # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
            self.reconstruction_module_keep_prediction = cfg_dict.get("reconstruction_module_keep_prediction")

            # initialize weights if not using pretrained cirim
            if not cfg_dict.get("pretrained", False):
                std_init_range = 1 / self.reconstruction_module_recurrent_filters[0] ** 0.5
                self.cirim.apply(lambda module: utils.rnn_weights_init(module, std_init_range))

            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.reconstruction_module_accumulate_predictions = cfg_dict.get(
                "reconstruction_module_accumulate_predictions"
            )

        quantitative_module_num_cascades = cfg_dict.get("quantitative_module_num_cascades")
        self.qcirim = torch.nn.ModuleList(
            [
                qrim_block.qRIMBlock(
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
                    linear_forward_model=base_quantitative_models.SignalForwardModel(
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

        self.accumulate_predictions = cfg_dict.get("quantitative_module_accumulate_predictions")

        self.gamma = torch.tensor(cfg_dict.get("quantitative_module_gamma_regularization_factors"))
        self.preprocessor = qrim_utils.RescaleByMax

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
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
        R2star_map_init : torch.Tensor
            Initial R2* map of shape [batch_size, n_x, n_y].
        S0_map_init : torch.Tensor
            Initial S0 map of shape [batch_size, n_x, n_y].
        B0_map_init : torch.Tensor
            Initial B0 map of shape [batch_size, n_x, n_y].
        phi_map_init : torch.Tensor
            Initial phase map of shape [batch_size, n_x, n_y].
        TEs : List
            List of echo times.
        y : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_echoes, n_coils, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        mask_brain : torch.Tensor
            Brain mask of shape [batch_size, 1, n_x, n_y, 1].
        sampling_mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].

        Returns
        -------
        List of list of torch.Tensor or torch.Tensor
             If self.accumulate_loss is True, returns a list of all intermediate predictions.
             If False, returns the final estimate.
        """
        if self.use_reconstruction_module:
            echoes_predictions = []
            cascades_echoes_predictions = []
            sigma = 1.0
            for echo in range(y.shape[1]):
                prediction = y[:, echo, ...].clone()
                init_pred = None
                hx = None
                cascades_predictions = []
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
                        keep_prediction=False if i == 0 else self.reconstruction_module_keep_prediction,
                    )
                    cascades_predictions.append([torch.view_as_complex(x) for x in prediction])
                cascades_echoes_predictions.append(cascades_predictions)
                echoes_predictions.append(cascades_predictions[-1][-1])

            prediction = torch.stack(echoes_predictions, dim=1)
            if prediction.shape[-1] != 2:
                prediction = torch.view_as_real(prediction)
            y = fft.fft2(
                utils.complex_mul(prediction.unsqueeze(self.coil_dim), sensitivity_maps.unsqueeze(self.coil_dim - 1)),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            for batch_idx in range(prediction.shape[0]):
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = transforms.R2star_B0_S0_phi_mapping(
                    prediction[batch_idx],
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

        prediction = None
        hx = None

        cascades_R2star_maps = []
        cascades_S0_maps = []
        cascades_B0_maps = []
        cascades_phi_maps = []
        for i, cascade in enumerate(self.qcirim):
            # Forward pass through the cascades
            prediction, hx = cascade(
                y,
                R2star_map_pred,
                S0_map_pred,
                B0_map_pred,
                phi_map_pred,
                TEs,
                sensitivity_maps,
                sampling_mask,
                prediction,
                hx,
                self.gamma,
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
                    torch.abs(pred)
                )
                time_steps_R2star_maps.append(_R2star_map_pred)
                time_steps_S0_maps.append(_S0_map_pred)
                time_steps_B0_maps.append(_B0_map_pred)
                time_steps_phi_maps.append(_phi_map_pred)
            cascades_R2star_maps.append(time_steps_R2star_maps)
            cascades_S0_maps.append(time_steps_S0_maps)
            cascades_B0_maps.append(time_steps_B0_maps)
            cascades_phi_maps.append(time_steps_phi_maps)

            prediction = torch.stack(
                [
                    cascades_R2star_maps[-1][-1],
                    cascades_S0_maps[-1][-1],
                    cascades_B0_maps[-1][-1],
                    cascades_phi_maps[-1][-1],
                ],
                dim=1,
            )

        pred = cascades_echoes_predictions if self.use_reconstruction_module else torch.empty([])

        yield [pred, cascades_R2star_maps, cascades_S0_maps, cascades_B0_maps, cascades_phi_maps]

    def process_intermediate_pred(self, x):
        """
        Process the intermediate prediction.

        Parameters
        ----------
        x : torch.Tensor
            Prediction of shape [batch_size, n_coils, n_x, n_y, 2].

        Returns
        -------
        torch.Tensor
            Processed prediction of shape [batch_size, n_x, n_y, 2].
        """
        x = self.preprocessor.reverse(x, self.gamma)
        R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
            x[:, 0, ...],
            x[:, 1, ...],
            x[:, 2, ...],
            x[:, 3, ...],
        )
        return R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred

    def process_quantitative_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        mask_brain: torch.Tensor,
        quantitative_map: str,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Processes the quantitative loss for the qRIM and qCIRIM models.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        mask_brain : torch.Tensor
            Mask for brain of shape [batch_size, n_x, n_y, 1].
        quantitative_map : str
            Type of quantitative map to regularize the loss. Must be one of {"R2star", "S0", "B0", "phi"}.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if isinstance(target, list):
            target = target[-1]
        if isinstance(target, list):
            target = target[-1]

        if "ssim" in str(loss_func).lower():

            def compute_quantitative_loss(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.FloatTensor:
                """
                Wrapper for SSIM loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].
                m : torch.Tensor
                    Mask of shape [batch_size, n_x, n_y, 1].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                max_value = torch.max(torch.abs(y)) - torch.min(torch.abs(y)).unsqueeze(dim=0)
                m = torch.abs(m).to(x)

                loss = (
                    loss_func(x * m, y * m, data_range=max_value) * self.loss_regularization_factors[quantitative_map]
                )
                return loss

        else:

            def compute_quantitative_loss(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.FloatTensor:
                """
                Wrapper for any (expect the SSIM) loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].
                m : torch.Tensor
                    Mask of shape [batch_size, n_x, n_y, 1].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                x = x / torch.max(torch.abs(x))
                y = (y / torch.max(torch.abs(y))).to(x)
                m = torch.abs(m).to(x)

                if "mse" in str(loss_func).lower():
                    x = x.float()
                    y = y.float()
                    m = m.float()
                return loss_func(x * m, y * m) / self.loss_regularization_factors[quantitative_map]

        if self.accumulate_predictions:
            cascades_loss = []
            for cascade_pred in prediction:
                time_steps_loss = [
                    compute_quantitative_loss(target, time_step_pred, mask_brain) for time_step_pred in cascade_pred
                ]
                cascades_loss.append(torch.sum(torch.stack(time_steps_loss, dim=0)) / len(prediction))
            yield sum(cascades_loss) / len(self.qcirim) * self.quantitative_loss_regularization_factor
        else:
            return (
                compute_quantitative_loss(target, prediction, mask_brain)
                * self.quantitative_loss_regularization_factor
            )

    def process_reconstruction_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Dict,
        r: int,
        loss_func: torch.nn.Module,
        kspace_reconstruction_loss: bool = False,
    ) -> torch.Tensor:
        """
        Processes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.
        kspace_reconstruction_loss : bool
            If True, the loss will be computed on the k-space data. Otherwise, the loss will be computed on the
            image space data. Default is ``False``. Note that this is different from
            ``self.kspace_reconstruction_loss``, so it can be used with multiple losses.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if self.unnormalize_loss_inputs:
            if self.n2r and not attrs["n2r_supervised"]:
                target = utils.unnormalize(
                    target,
                    {
                        "min": attrs["prediction_min"] if "prediction_min" in attrs else attrs[f"prediction_min_{r}"],
                        "max": attrs["prediction_max"] if "prediction_max" in attrs else attrs[f"prediction_max_{r}"],
                        "mean": (
                            attrs["prediction_mean"] if "prediction_mean" in attrs else attrs[f"prediction_mean_{r}"]
                        ),
                        "std": attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"],
                    },
                    self.normalization_type,
                )
                prediction = utils.unnormalize(
                    prediction,
                    {
                        "min": (
                            attrs["noise_prediction_min"]
                            if "noise_prediction_min" in attrs
                            else attrs[f"noise_prediction_min_{r}"]
                        ),
                        "max": (
                            attrs["noise_prediction_max"]
                            if "noise_prediction_max" in attrs
                            else attrs[f"noise_prediction_max_{r}"]
                        ),
                        attrs["noise_prediction_mean"] if "noise_prediction_mean" in attrs else "mean": attrs[
                            f"noise_prediction_mean_{r}"
                        ],
                        attrs["noise_prediction_std"] if "noise_prediction_std" in attrs else "std": attrs[
                            f"noise_prediction_std_{r}"
                        ],
                    },
                    self.normalization_type,
                )
            else:
                target = utils.unnormalize(
                    target,
                    {
                        "min": attrs["target_min"],
                        "max": attrs["target_max"],
                        "mean": attrs["target_mean"],
                        "std": attrs["target_std"],
                    },
                    self.normalization_type,
                )
                prediction = utils.unnormalize(
                    prediction,
                    {
                        "min": attrs["prediction_min"] if "prediction_min" in attrs else attrs[f"prediction_min_{r}"],
                        "max": attrs["prediction_max"] if "prediction_max" in attrs else attrs[f"prediction_max_{r}"],
                        "mean": (
                            attrs["prediction_mean"] if "prediction_mean" in attrs else attrs[f"prediction_mean_{r}"]
                        ),
                        "std": attrs["prediction_std"] if "prediction_std" in attrs else attrs[f"prediction_std_{r}"],
                    },
                    self.normalization_type,
                )

            sensitivity_maps = utils.unnormalize(
                sensitivity_maps,
                {
                    "min": attrs["sensitivity_maps_min"],
                    "max": attrs["sensitivity_maps_max"],
                    "mean": attrs["sensitivity_maps_mean"],
                    "std": attrs["sensitivity_maps_std"],
                },
                self.normalization_type,
            )

        if not self.kspace_reconstruction_loss and not kspace_reconstruction_loss and not self.unnormalize_loss_inputs:
            target = torch.abs(target / torch.max(torch.abs(target)))
        else:
            if target.shape[-1] != 2:
                target = torch.view_as_real(target)
            if self.ssdu or kspace_reconstruction_loss:
                target = utils.expand_op(target, sensitivity_maps, self.coil_dim)
            target = fft.fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)

        if "ssim" in str(loss_func).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for SSIM loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                y = torch.abs(y / torch.max(torch.abs(y)))
                return loss_func(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for any (expect the SSIM) loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                if (
                    not self.kspace_reconstruction_loss
                    and not kspace_reconstruction_loss
                    and not self.unnormalize_loss_inputs
                ):
                    y = torch.abs(y / torch.max(torch.abs(y)))
                else:
                    if y.shape[-1] != 2:
                        y = torch.view_as_real(y)
                    if self.ssdu or kspace_reconstruction_loss:
                        y = utils.expand_op(y, sensitivity_maps, self.coil_dim)
                    y = fft.fft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                    if self.ssdu or kspace_reconstruction_loss:
                        y = y * mask
                return loss_func(x, y)

        if self.reconstruction_module_accumulate_predictions:
            echoes_loss = []
            for echo_time, item in enumerate(prediction):
                cascades_loss = []
                for cascade_pred in item:
                    time_steps_loss = [
                        compute_reconstruction_loss(target[:, echo_time, :, :], time_step_pred).mean()
                        for time_step_pred in cascade_pred
                    ]
                    _loss = [
                        x * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(time_steps_loss[0])
                        for x in time_steps_loss
                    ]
                    cascades_loss.append(sum(sum(_loss) / self.reconstruction_module_time_steps))
                echoes_loss.append(sum(list(cascades_loss)) / len(self.cirim))
            yield sum(echoes_loss) / len(prediction) * self.reconstruction_loss_regularization_factor
        else:
            return compute_reconstruction_loss(target, prediction) * self.reconstruction_loss_regularization_factor
