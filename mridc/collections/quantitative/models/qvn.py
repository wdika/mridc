# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

from abc import ABC
from typing import Any, List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import Tensor

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import coil_combination, complex_conj, complex_mul
from mridc.collections.quantitative.models.base import BaseqMRIReconstructionModel
from mridc.collections.quantitative.models.qrim.utils import RescaleByMax, SignalForwardModel
from mridc.collections.quantitative.models.qvarnet.qvn_block import qVarNetBlock
from mridc.collections.quantitative.parts.transforms import R2star_B0_real_S0_complex_mapping
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.models.varnet.vn_block import VarNetBlock
from mridc.core.classes.common import typecheck

__all__ = ["qVarNet"]


class qVarNet(BaseqMRIReconstructionModel, ABC):
    """
    Implementation of the quantitative End-to-end Variational Network (qVN), as presented in Zhang, C. et al.

    References
    ----------

    ..

        Zhang, C. et al. (2022) ‘A unified model for reconstruction and R2 mapping of accelerated 7T data using \
        quantitative Recurrent Inference Machine’. In review.

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

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")
        self.shift_B0_input = cfg_dict.get("shift_B0_input")

        self.vn = torch.nn.ModuleList([])

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module")
        if self.use_reconstruction_module:
            self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
            self.reconstruction_module_no_dc = cfg_dict.get("reconstruction_module_no_dc")

            for _ in range(self.reconstruction_module_num_cascades):
                self.vn.append(
                    VarNetBlock(
                        NormUnet(
                            chans=cfg_dict.get("reconstruction_module_channels"),
                            num_pools=cfg_dict.get("reconstruction_module_pooling_layers"),
                            in_chans=cfg_dict.get("reconstruction_module_in_channels"),
                            out_chans=cfg_dict.get("reconstruction_module_out_channels"),
                            padding_size=cfg_dict.get("reconstruction_module_padding_size"),
                            normalize=cfg_dict.get("reconstruction_module_normalize"),
                        ),
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim - 1,
                        no_dc=self.reconstruction_module_no_dc,
                    )
                )

            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.reconstruction_module_accumulate_estimates = cfg_dict.get(
                "reconstruction_module_accumulate_estimates"
            )

        quantitative_module_num_cascades = cfg_dict.get("quantitative_module_num_cascades")
        self.qvn = torch.nn.ModuleList(
            [
                qVarNetBlock(
                    NormUnet(
                        chans=cfg_dict.get("quantitative_module_channels"),
                        num_pools=cfg_dict.get("quantitative_module_pooling_layers"),
                        in_chans=cfg_dict.get("quantitative_module_in_channels"),
                        out_chans=cfg_dict.get("quantitative_module_out_channels"),
                        padding_size=cfg_dict.get("quantitative_module_padding_size"),
                        normalize=cfg_dict.get("quantitative_module_normalize"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=cfg_dict.get("quantitative_module_no_dc"),
                    linear_forward_model=SignalForwardModel(
                        sequence=cfg_dict.get("quantitative_module_signal_forward_model_sequence")
                    ),
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
    ) -> List[Union[Tensor, List[Any]]]:
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
            cascades_echoes_etas = []
            for echo in range(y.shape[1]):
                prediction = y[:, echo, ...].clone()
                for cascade in self.vn:
                    # Forward pass through the cascades
                    prediction = cascade(prediction, y[:, echo, ...], sensitivity_maps, sampling_mask.squeeze(1))
                estimation = ifft2(
                    prediction,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                estimation = coil_combination(
                    estimation, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim - 1
                )
                cascades_echoes_etas.append(torch.view_as_complex(estimation))

            eta = torch.stack(cascades_echoes_etas, dim=1)
            if eta.shape[-1] != 2:
                eta = torch.view_as_real(eta)
            y = fft2(
                complex_mul(eta.unsqueeze(self.coil_dim), sensitivity_maps.unsqueeze(self.coil_dim - 1)),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )
            recon_eta = torch.view_as_complex(eta).clone()

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
        for cascade in self.qvn:
            # Forward pass through the cascades
            prediction = cascade(
                prediction,
                y,
                R2star_map_pred,
                S0_map_pred,
                B0_map_pred,
                phi_map_pred,
                TEs,
                sensitivity_maps,
                sampling_mask,
                self.gamma,
            )
            R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
                prediction[:, 0],
                prediction[:, 1],
                prediction[:, 2],
                prediction[:, 3],
            )
            if R2star_map_pred.shape[-1] == 2:
                R2star_map_pred = torch.view_as_complex(R2star_map_pred)
            if S0_map_pred.shape[-1] == 2:
                S0_map_pred = torch.view_as_complex(S0_map_pred)
            if B0_map_pred.shape[-1] == 2:
                B0_map_pred = torch.view_as_complex(B0_map_pred)
            if phi_map_pred.shape[-1] == 2:
                phi_map_pred = torch.view_as_complex(phi_map_pred)

        R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = self.process_intermediate_pred(
            torch.abs(torch.view_as_complex(prediction)), None, None, False
        )

        return [
            recon_eta if self.use_reconstruction_module else torch.empty([]),
            R2star_map_pred,
            S0_map_pred,
            B0_map_pred,
            phi_map_pred,
        ]

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
