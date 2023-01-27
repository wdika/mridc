# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Generator, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
import mridc.collections.reconstruction.models.rim.rim_block as rim_block
import mridc.core.classes.common as common_classes

__all__ = ["CIRIM"]


class CIRIM(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Cascades of Independently Recurrent Inference Machines, as presented in [1].

    References
    ----------
    .. [1] Karkalousos D, Noteboom S, Hulst HE, Vos FM, Caan MWA. Assessment of data consistency through cascades of
        independently recurrent inference machines for fast and robust accelerated MRI reconstruction. Phys Med Biol.
        2022 Jun 8;67(12). doi: 10.1088/1361-6560/ac6cc2. PMID: 35508147.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.recurrent_filters = cfg_dict.get("recurrent_filters")

        # make time-steps size divisible by 8 for fast fp16 training
        self.time_steps = 8 * math.ceil(cfg_dict.get("time_steps") / 8)

        self.no_dc = cfg_dict.get("no_dc")
        self.num_cascades = cfg_dict.get("num_cascades")

        self.cirim = torch.nn.ModuleList(
            [
                rim_block.RIMBlock(
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=cfg_dict.get("conv_filters"),
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=self.recurrent_filters,
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    time_steps=self.time_steps,
                    conv_dim=cfg_dict.get("conv_dim"),
                    no_dc=self.no_dc,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    dimensionality=cfg_dict.get("dimensionality"),
                )
                for _ in range(self.num_cascades)
            ]
        )

        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.keep_prediction = cfg_dict.get("keep_prediction")

    @common_classes.typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_pred : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        target : torch.Tensor
            Target data to compute the loss. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        List of torch.Tensor or torch.Tensor
            If self.keep_prediction is True, returns a list of the intermediate predictions for each cascade.
            Otherwise, returns the final prediction. Shape [batch_size, n_x, n_y, 2]
        """
        prediction = y.clone()
        init_pred = None if init_pred is None or init_pred.dim() < 4 else init_pred
        hx = None
        sigma = 1.0
        cascades_predictions = []
        for i, cascade in enumerate(self.cirim):
            # Forward pass through the cascades
            prediction, _ = cascade(
                prediction,
                y,
                sensitivity_maps,
                mask,
                init_pred,
                hx,
                sigma,
                keep_prediction=False if i == 0 else self.keep_prediction,
            )
            time_steps_predictions = [
                self.process_intermediate_pred(pred, sensitivity_maps, target) for pred in prediction
            ]
            cascades_predictions.append(time_steps_predictions)
        yield cascades_predictions

    def process_intermediate_pred(
        self,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        target: torch.Tensor,
        do_coil_combination: bool = False,
    ) -> torch.Tensor:
        """
        Processes the intermediate prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Intermediate prediction. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        target : torch.Tensor
            Target data to crop to size. Shape [batch_size, n_x, n_y, 2]
        do_coil_combination : bool
            Whether to do coil combination. In this case the prediction is in k-space. Default is ``False``.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        # Take the last time step of the predictions
        if not self.no_dc or do_coil_combination:
            prediction = fft.ifft2(
                prediction,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            prediction = utils.coil_combination_method(
                prediction, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
            )
        prediction = torch.view_as_complex(prediction)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction

    def process_reconstruction_loss(
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Processes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        target = torch.abs(target / torch.max(torch.abs(target)))

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
                y = torch.abs(y / torch.max(torch.abs(y)))
                return loss_func(x, y)

        if self.accumulate_predictions:
            cascades_loss = []
            for cascade_pred in prediction:
                time_steps_loss = [
                    compute_reconstruction_loss(target, time_step_pred) for time_step_pred in cascade_pred
                ]
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.time_steps).to(time_steps_loss[0]) for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.time_steps))
            yield sum(list(cascades_loss)) / len(self.cirim)
        else:
            return compute_reconstruction_loss(target, prediction)
