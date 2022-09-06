# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

from typing import List, Sequence, Union

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import coil_combination, complex_mul


class RescaleByMax(object):
    def __init__(self, slack=1e-6):
        self.slack = slack

    def forward(self, data):
        """Apply scaling."""
        gamma = torch.max(torch.max(torch.abs(data), 3, keepdim=True)[0], 2, keepdim=True)[0] + self.slack
        data = data / gamma
        return data, gamma

    @staticmethod
    def reverse(data, gamma):
        """Reverse scaling."""
        return torch.stack([data[i] * gamma[i] for i in range(data.shape[0])], 0)


class SignalForwardModel(object):
    """Defines a signal forward model"""

    def __init__(self, sequence: Union[str, None] = None):
        super(SignalForwardModel, self).__init__()
        self.sequence = sequence.lower() if isinstance(sequence, str) else None
        self.scaling = 1e-3

    def __call__(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs=None,
    ):
        """
        Defines forward model based on sequence.

        Parameters
        ----------
        R2star_map: R2* map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map: S0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        B0_map: B0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        phi_map: phi map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        TEs: List of echo times.
            List of float, shape [n_echoes]
        """
        if TEs is None:
            TEs = torch.Tensor([3.0, 11.5, 20.0, 28.5])
        if self.sequence == "megre":
            return self.MEGRESignalModel(R2star_map, S0_map, B0_map, phi_map, TEs)
        elif self.sequence == "megre_no_phase":
            return self.MEGRENoPhaseSignalModel(R2star_map, S0_map, TEs)
        else:
            raise ValueError(
                "Only MEGRE and MEGRE no phase are supported are signal forward model at the moment. "
                f"Found {self.sequence}"
            )

    def MEGRESignalModel(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        B0_map: torch.Tensor,
        phi_map: torch.Tensor,
        TEs: List,
    ):
        """
        MEGRE forward model.

        Parameters
        ----------
        R2star_map: R2* map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map: S0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        B0_map: B0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        phi_map: phi map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        TEs: List of echo times.
            List of float, shape [n_echoes]
        """
        S0_map_real = S0_map
        S0_map_imag = phi_map

        first_term = lambda i: torch.exp(-TEs[i] * self.scaling * R2star_map)
        second_term = lambda i: torch.cos(B0_map * self.scaling * -TEs[i])
        third_term = lambda i: torch.sin(B0_map * self.scaling * -TEs[i])

        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map_real * first_term(i) * second_term(i) - S0_map_imag * first_term(i) * third_term(i),
                        S0_map_real * first_term(i) * third_term(i) + S0_map_imag * first_term(i) * second_term(i),
                    ),
                    -1,
                )
                for i in range(len(TEs))
            ],
            1,
        )
        pred[pred != pred] = 0.0
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])

    def MEGRENoPhaseSignalModel(
        self,
        R2star_map: torch.Tensor,
        S0_map: torch.Tensor,
        TEs: List,
    ):
        """
        MEGRE no phase forward model.

        Parameters
        ----------
        R2star_map: R2* map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        S0_map: S0 map.
            torch.Tensor, shape [batch_size, n_x, n_y]
        TEs: List of echo times.
            List of float, shape [n_echoes]
        """
        pred = torch.stack(
            [
                torch.stack(
                    (
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),
                        S0_map * torch.exp(-TEs[i] * self.scaling * R2star_map),
                    ),
                    -1,
                )
                for i in range(len(TEs))
            ],
            1,
        )
        pred[pred != pred] = 0.0
        return torch.view_as_real(pred[..., 0] + 1j * pred[..., 1])


def expand_op(x, sensitivity_maps):
    """Expand a coil-combined image to multicoil."""
    x = complex_mul(x, sensitivity_maps)
    if torch.isnan(x).any():
        x[x != x] = 0
    return x


def analytical_log_likelihood_gradient(
    linear_forward_model: SignalForwardModel,
    R2star_map: torch.Tensor,
    S0_map: torch.Tensor,
    B0_map: torch.Tensor,
    phi_map: torch.Tensor,
    TEs: List,
    sensitivity_maps: torch.Tensor,
    masked_kspace: torch.Tensor,
    sampling_mask: torch.Tensor,
    fft_centered: bool,
    fft_normalization: str,
    spatial_dims: Sequence[int],
    coil_dim: int,
    coil_combination_method: str = "SENSE",
    scaling: float = 1e-3,
) -> torch.Tensor:
    """
    Computes the analytical gradient of the log-likelihood function.

    Parameters
    ----------
    linear_forward_model: SignalForwardModel
        Signal forward model to use.
    R2star_map: R2* map.
        torch.Tensor, shape [batch_size, n_x, n_y]
    S0_map: S0 map.
        torch.Tensor, shape [batch_size, n_x, n_y]
    B0_map: B0 map.
        torch.Tensor, shape [batch_size, n_x, n_y]
    phi_map: phi map.
        torch.Tensor, shape [batch_size, n_x, n_y]
    TEs: List of echo times.
        List of float, shape [n_echoes]
    sensitivity_maps: Coil sensitivity maps.
        torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
    masked_kspace: Data.
        torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
    sampling_mask: Mask of the sampling.
        torch.Tensor, shape [batch_size, 1, n_x, n_y, 1]
    fft_centered: If True, the FFT is centered.
        bool
    fft_normalization: Normalization of the FFT.
        str, one of "ortho", "forward", "backward", None
    spatial_dims: Spatial dimensions of the input.
        Sequence of int, shape [n_dims]
    coil_dim: Coils dimension of the input.
        int
    coil_combination_method: Method to use for coil combination.
        str, one of "SENSE", "RSS"
    scaling: Scaling factor.
        float

    Returns
    -------
    Analytical gradient of the log-likelihood function.
    """

    nr_TEs = len(TEs)

    R2star_map = R2star_map.unsqueeze(0)
    S0_map = S0_map.unsqueeze(0)
    B0_map = B0_map.unsqueeze(0)
    phi_map = phi_map.unsqueeze(0)

    pred = linear_forward_model(R2star_map, S0_map, B0_map, phi_map, TEs)

    S0_map_real = S0_map
    S0_map_imag = phi_map

    pred_kspace = fft2(
        expand_op(pred.unsqueeze(coil_dim), sensitivity_maps.unsqueeze(0).unsqueeze(coil_dim - 1)),
        fft_centered,
        fft_normalization,
        spatial_dims,
    )

    diff_data = (pred_kspace - masked_kspace) * sampling_mask
    diff_data_inverse = coil_combination(
        ifft2(diff_data, fft_centered, fft_normalization, spatial_dims),
        sensitivity_maps.unsqueeze(0).unsqueeze(coil_dim - 1),
        method=coil_combination_method,
        dim=coil_dim,
    )

    first_term = lambda i: torch.exp(-TEs[i] * scaling * R2star_map)
    second_term = lambda i: torch.cos(B0_map * scaling * -TEs[i])
    third_term = lambda i: torch.sin(B0_map * scaling * -TEs[i])

    S0_part_der = torch.stack(
        [torch.stack((first_term(i) * second_term(i), -first_term(i) * third_term(i)), -1) for i in range(nr_TEs)], 1
    )

    R2str_part_der = torch.stack(
        [
            torch.stack(
                (
                    -TEs[i] * scaling * first_term(i) * (S0_map_real * second_term(i) - S0_map_imag * third_term(i)),
                    -TEs[i] * scaling * first_term(i) * (-S0_map_real * third_term(i) - S0_map_imag * second_term(i)),
                ),
                -1,
            )
            for i in range(nr_TEs)
        ],
        1,
    )

    S0_map_real_grad = (
        diff_data_inverse[..., 0] * S0_part_der[..., 0] - diff_data_inverse[..., 1] * S0_part_der[..., 1]
    )
    S0_map_imag_grad = (
        diff_data_inverse[..., 0] * S0_part_der[..., 1] + diff_data_inverse[..., 1] * S0_part_der[..., 0]
    )
    R2star_map_real_grad = (
        diff_data_inverse[..., 0] * R2str_part_der[..., 0] - diff_data_inverse[..., 1] * R2str_part_der[..., 1]
    )
    R2star_map_imag_grad = (
        diff_data_inverse[..., 0] * R2str_part_der[..., 1] + diff_data_inverse[..., 1] * R2str_part_der[..., 0]
    )

    S0_map_grad = torch.stack([S0_map_real_grad, S0_map_imag_grad], -1).squeeze()
    S0_map_grad = torch.mean(S0_map_grad, 0)
    R2star_map_grad = torch.stack([R2star_map_real_grad, R2star_map_imag_grad], -1).squeeze()
    R2star_map_grad = torch.mean(R2star_map_grad, 0)

    return torch.stack([R2star_map_grad[..., 0], S0_map_grad[..., 0], R2star_map_grad[..., 1], S0_map_grad[..., 1]], 0)
