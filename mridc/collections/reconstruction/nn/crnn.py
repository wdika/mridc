# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Dict, Generator, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.nn.conv import gruconv2d
from mridc.collections.reconstruction.nn.convrecnet import crnn_block

__all__ = ["CRNNet"]


class CRNNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the Convolutional Recurrent Neural Network, inspired by [1].

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent
        Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1,
        pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.num_iterations = cfg_dict.get("num_iterations")

        self.crnn = crnn_block.RecurrentConvolutionalNetBlock(
            gruconv2d.GRUConv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("hidden_channels"),
                n_convs=cfg_dict.get("n_convs"),
                batchnorm=cfg_dict.get("batchnorm"),
            ),
            num_iterations=self.num_iterations,
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
            no_dc=self.no_dc,
        )

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,  # noqa: W0613
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
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        pred = self.crnn(y, sensitivity_maps, mask)
        yield [self.process_intermediate_pred(x, sensitivity_maps, target) for x in pred]

    def process_intermediate_pred(
        self,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process the intermediate prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Intermediate prediction. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        target : torch.Tensor
            Target data to crop to size. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
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
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction

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

        iterations_loss = [compute_reconstruction_loss(target, iteration_pred) for iteration_pred in prediction]
        _loss = [x * torch.logspace(-1, 0, steps=self.num_iterations).to(iterations_loss[0]) for x in iterations_loss]
        yield sum(list(_loss)) / len(self.num_iterations) * self.reconstruction_loss_regularization_factor
