# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Union

import torch
from torch import nn

from mridc import ifft2c, complex_mul, complex_conj
from .e2evn import SensitivityModel
from .rim.rim_block import RIMBlock
from ..data.transforms import center_crop_to_smallest


class CIRIM(nn.Module):
    """Cascades of RIM blocks."""

    def __init__(
        self,
        recurrent_layer: str = "IndRNN",
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        loss_fn: Union[nn.Module, str] = "l1",
        num_cascades: int = 1,
        no_dc: bool = False,
        keep_eta: bool = False,
        use_sens_net: bool = False,
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_normalize: bool = True,
        sens_mask_type: str = "2D",
        fft_type: str = "orthogonal",
        output_type: str = "SENSE",
    ):
        """
        Args:
            recurrent_layer: Recurrent Layer selected from rnn_cells
            conv_filters: Number of filters in the convolutional layers
            conv_kernels: Kernel size in the convolutional layers
            conv_dilations: Dilation in the convolutional layers
            conv_biased: Whether to use bias in the convolutional layers
            recurrent_filters: Number of filters in the recurrent layers
            recurrent_kernels: Kernel size in the recurrent layers
            recurrent_dilations: Dilation in the recurrent layers
            recurrent_biased: Whether to use bias in the recurrent layers
            depth: Number of layers in the network
            time_steps: Number of time steps in the input
            conv_dim: Dimension of the input
            loss_fn: Loss function to use
            num_cascades: Number of cascades
            no_dc: Whether to remove the DC component
            keep_eta: Whether to keep the eta term
            use_sens_net: Whether to use the sensitivity network
            sens_chans: Number of channels in the sensitivity network
            sens_pools: Number of pools in the sensitivity network
            sens_normalize: Whether to normalize the sensitivity network
            sens_mask_type: Type of mask to use for the sensitivity network, 1D or 2D
            fft_type: Type of FFT to use, data/orthogonal or numpy-like
            output_type: Type of output to use, SENSE or RSS
        """
        super(CIRIM, self).__init__()

        # Initialize the cascades with RIM blocks
        if recurrent_bias is None:
            recurrent_bias = [True, True, False]
        if recurrent_dilations is None:
            recurrent_dilations = [1, 1, 0]
        if recurrent_kernels is None:
            recurrent_kernels = [1, 1, 0]
        if recurrent_filters is None:
            recurrent_filters = [64, 64, 0]
        if conv_bias is None:
            conv_bias = [True, True, False]
        if conv_dilations is None:
            conv_dilations = [1, 2, 1]
        if conv_kernels is None:
            conv_kernels = [5, 3, 3]
        if conv_filters is None:
            conv_filters = [64, 64, 2]

        self.fft_type = fft_type
        self.no_dc = no_dc
        self.time_steps = time_steps

        self.cascades = nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=recurrent_layer,
                    conv_filters=conv_filters,
                    conv_kernels=conv_kernels,
                    conv_dilations=conv_dilations,
                    conv_bias=conv_bias,
                    recurrent_filters=recurrent_filters,
                    recurrent_kernels=recurrent_kernels,
                    recurrent_dilations=recurrent_dilations,
                    recurrent_bias=recurrent_bias,
                    depth=depth,
                    time_steps=self.time_steps,
                    conv_dim=conv_dim,
                    no_dc=self.no_dc,
                    fft_type=self.fft_type,
                )
                for _ in range(num_cascades)
            ]
        )

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = use_sens_net
        if self.use_sens_net:
            self.sens_net = SensitivityModel(
                sens_chans, sens_pools, fft_type=self.fft_type, mask_type=sens_mask_type, normalize=sens_normalize
            )

        self.loss_fn = loss_fn

        # Initialize data consistency term
        self.dc_weight = nn.Parameter(torch.ones(1))

        # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
        self.keep_eta = keep_eta

        # Initialize the output layer
        self.output_type = output_type

        # TODO: replace print with logger
        print("No of parameters: {:,d}".format(self.get_num_params()))

    def get_num_params(self):
        """
        Get the number of parameters in the model.

        Returns:
            Number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        target: torch.Tensor = None,
        max_value: float = 1.0,
        sigma: float = 1.0,
        accumulate_loss: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            masked_kspace: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sense: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            eta: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for eta
            hx: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for hx
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
            max_value: float, maximum value of the data
            sigma: float, noise level
            accumulate_loss: bool, accumulate loss or not

        Returns:
            eta: torch.Tensor, shape [batch_size, n_x, n_y, 2], estimated eta
            hx: torch.Tensor, shape [batch_size, n_x, n_y, 2], estimated hx
            loss: torch.Tensor, shape [1], loss value
        """
        sense = self.sens_net(masked_kspace, mask) if self.use_sens_net and self.sens_net is not None else sense

        pred = masked_kspace.clone()

        # Accumulate loss over cascades
        cascade_time_steps_loss = []
        for i, cascade in enumerate(self.cascades):
            # Forward pass through cascade
            pred, hx = cascade(
                pred, masked_kspace, sense, mask, eta, hx, sigma, keep_eta=False if i == 0 else self.keep_eta
            )

            # Accumulate loss over time steps
            if accumulate_loss:
                time_steps_loss = []

                for p in pred:

                    if self.no_dc is False and self.keep_eta is False:
                        p = ifft2c(p, fft_type=self.fft_type)

                        if self.output_type == "SENSE":
                            p = complex_mul(p, complex_conj(sense)).sum(dim=1)
                        elif self.output_type == "RSS":
                            p = torch.sqrt((p**2).sum(dim=1))
                        else:
                            raise ValueError("Output type not supported.")

                    output = torch.view_as_complex(p)
                    target, output = center_crop_to_smallest(target, output)

                    loss = (
                        self.loss_fn(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)  # type: ignore
                        if "ssim" in str(self.loss_fn).lower()
                        else self.loss_fn(output, target)  # type: ignore
                    )
                    time_steps_loss.append(loss)

                # Add weighted loss for each cascade. Loss is weighted for total number of time-steps on range 0-1.
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.time_steps).to(time_steps_loss[0]) for x in time_steps_loss
                ]

                # Take average of all time-steps loss
                cascade_time_steps_loss.append(sum(sum(_loss) / self.time_steps))  # type: ignore

        # Take average of all cascades loss
        if accumulate_loss:
            yield sum(list(cascade_time_steps_loss)) / len(self.cascades)
        else:
            if isinstance(pred, list):
                # Use the prediction of the last time-step.
                pred = pred[-1].detach()

            if self.no_dc is False and self.keep_eta is False:
                pred = ifft2c(pred, fft_type=self.fft_type)

                if self.output_type == "SENSE":
                    pred = complex_mul(pred, complex_conj(sense)).sum(dim=1)
                elif self.output_type == "RSS":
                    pred = torch.sqrt((pred**2).sum(dim=1))
                else:
                    raise ValueError("Output type not supported.")

            pred = torch.view_as_complex(pred)
            pred = torch.abs(pred / torch.max(torch.abs(pred)))

            return pred

    def inference(
        self,
        masked_kspace: torch.Tensor,
        sense: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
        accumulate_estimates: bool = False,
    ) -> torch.Tensor:
        """
        Inference step of the model.

        Args:
            masked_kspace: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sense: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            eta: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for eta
            hx: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for hx
            sigma: float, noise level
            accumulate_estimates: bool, if True, accumulate estimates for all time-steps

        Returns
        -------
            pred: torch.Tensor, shape [batch_size, n_x, n_y, 2], predicted kspace data
        """
        sense = self.sens_net(masked_kspace, mask) if self.use_sens_net and self.sens_net is not None else sense

        preds = []

        pred = masked_kspace.clone()
        for i, cascade in enumerate(self.cascades):
            pred, hx = cascade(
                pred, masked_kspace, sense, mask, eta, hx, sigma, keep_eta=False if i == 0 else self.keep_eta
            )

            if self.no_dc is False and self.keep_eta is False:
                output = []
                for p in pred:
                    p = ifft2c(p, fft_type=self.fft_type)

                    if self.output_type == "SENSE":
                        p = complex_mul(p, complex_conj(sense)).sum(dim=1)
                    elif self.output_type == "RSS":
                        p = torch.sqrt((p**2).sum(dim=1))
                    else:
                        raise ValueError("Output type not supported.")

                    output.append(p)
                pred = output

            if accumulate_estimates:
                preds.append(pred)

            pred = pred[-1].detach()

        if accumulate_estimates:
            yield preds
        else:
            return torch.view_as_complex(pred)
