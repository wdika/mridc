# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/khammernik/sigmanet/blob/master/reconstruction/common/mytorch/models/datalayer.py

from typing import Any, List, Optional, Tuple

import torch

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils


class DataIDLayer(torch.nn.Module):
    """Placeholder for the data layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class DataGDLayer(torch.nn.Module):
    """
    DataLayer computing the gradient on the L2 dataterm.

    Parameters
    ----------
    lambda_init : float
        Initial value of data term weight lambda.
    learnable : bool
        If True, the data term weight lambda is learnable. Default is ``True``.
    fft_centered : bool
        If True, the FFT is centered. Default is ``False``.
    fft_normalization : str
        If "ortho", the FFT is normalized. Default is ``"backward"``.
    spatial_dims : tuple of int
        If not None, the spatial dimensions of the FFT. Default is ``None``.
    """

    def __init__(
        self,
        lambda_init: float,
        learnable: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        super(DataGDLayer, self).__init__()
        self.lambda_init = lambda_init
        self.data_weight = torch.nn.Parameter(torch.Tensor(1))
        self.data_weight.data = torch.tensor(
            lambda_init,
            dtype=self.data_weight.dtype,
        )
        self.data_weight.requires_grad = learnable

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, sensitivity_maps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the data layer.

        Parameters
        ----------
        x : torch.Tensor
            Prediction. Shape [batch_size, num_coils, height, width, 2].
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, num_coils, height, width, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, num_coils, height, width, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1, num_channels, height, width, 1].

        Returns
        -------
        torch.Tensor
            Data loss term.
        """
        A_x_y = (
            torch.sum(
                fft.fft2(
                    utils.complex_mul(x.unsqueeze(-5).expand_as(sensitivity_maps), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                * mask,
                -4,
                keepdim=True,
            )
            - y
        )
        gradD_x = torch.sum(
            utils.complex_mul(
                fft.ifft2(
                    A_x_y * mask,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                utils.complex_conj(sensitivity_maps),
            ),
            dim=(-5),
        )
        return x - self.data_weight * gradD_x


class DataProxCGLayer(torch.nn.Module):
    """
    Solving the prox wrt. dataterm using Conjugate Gradient as proposed in [1].

    References
    ----------
    .. [1] Aggarwal HK, Mani MP, Jacob M. MoDL: Model-based deep learning architecture for inverse problems. IEEE
        transactions on medical imaging. 2018 Aug 13;38(2):394-405.

    Parameters
    ----------
    lambda_init : float
        Initial value of data term weight lambda.
    tol : float
        Tolerance for the Conjugate Gradient solver. Default is ``1e-6``.
    iter : int
        Number of iterations for the Conjugate Gradient solver. Default is ``10``.
    learnable : bool
        If True, the data term weight lambda is learnable. Default is ``True``.
    fft_centered : bool
        If True, the FFT is centered. Default is ``False``.
    fft_normalization : str
        FFT normalization. Default is ``"backward"``.
    spatial_dims : tuple of int
        Spatial dimensions of the FFT. Default is ``None``.
    """

    def __init__(
        self,
        lambda_init: float,
        tol: float = 1e-6,
        iter: int = 10,
        learnable: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        super(DataProxCGLayer, self).__init__()

        self._lambda = torch.nn.Parameter(torch.Tensor(1))
        self._lambda.data = torch.tensor(lambda_init)
        self._lambda_init = lambda_init
        self._lambda.requires_grad = learnable

        self.tol = tol
        self.iter = iter

        self.op = ConjugateGradient

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(self, x: torch.Tensor, y: torch.Tensor, sensitivity_maps: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the data layer.

        Parameters
        ----------
        x : torch.Tensor
            Prediction. Shape [batch_size, num_coils, height, width, 2].
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, num_coils, height, width, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, num_coils, height, width, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1, num_channels, height, width, 1].

        Returns
        -------
        torch.Tensor
            Data loss term.
        """
        return self.op.apply(
            x,
            self._lambda,
            y,
            sensitivity_maps,
            mask,
            self.tol,
            self.iter,
            self.fft_centered,
            self.fft_normalization,
            self.spatial_dims,
        )

    def set_learnable(self, flag: bool):
        """Set the learnability of the data term weight lambda."""
        self._lambda.requires_grad = flag


class ConjugateGradient(torch.autograd.Function):
    """Conjugate Gradient solver for the prox of the data term."""

    @staticmethod
    def complexDot(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """Complex dot product of two tensors."""
        nBatch = data1.shape[0]
        mult = utils.complex_mul(data1, utils.complex_conj(data2))
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re.view(nBatch, -1), dim=-1), torch.sum(im.view(nBatch, -1), dim=-1)], -1)

    @staticmethod
    def solve(x0: torch.Tensor, M: torch.Tensor, tol: float, max_iter: int) -> torch.Tensor:
        """Solve the linear system Mx=b using conjugate gradient."""
        nBatch = x0.shape[0]
        x = torch.zeros(x0.shape).to(x0.device)
        r = x0.clone()
        p = x0.clone()
        x0x0 = (x0.pow(2)).view(nBatch, -1).sum(-1)
        rr = torch.stack([(r.pow(2)).view(nBatch, -1).sum(-1), torch.zeros(nBatch).to(x0.device)], dim=-1)

        it = 0
        while torch.min(rr[..., 0] / x0x0) > tol and it < max_iter:
            it += 1
            q = M(p)

            data1 = rr
            data2 = ConjugateGradient.complexDot(p, q)

            re1, im1 = torch.unbind(data1, -1)
            re2, im2 = torch.unbind(data2, -1)
            alpha = torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], -1) / utils.complex_abs(data2) ** 2

            x += utils.complex_mul(alpha.reshape(nBatch, 1, 1, 1, -1), p.clone())
            r -= utils.complex_mul(alpha.reshape(nBatch, 1, 1, 1, -1), q.clone())
            rr_new = torch.stack([(r.pow(2)).view(nBatch, -1).sum(-1), torch.zeros(nBatch).to(x0.device)], dim=-1)
            beta = torch.stack([rr_new[..., 0] / rr[..., 0], torch.zeros(nBatch).to(x0.device)], dim=-1)
            p = r.clone() + utils.complex_mul(beta.reshape(nBatch, 1, 1, 1, -1), p)
            rr = rr_new.clone()
        return x

    @staticmethod
    def forward(
        ctx: torch.autograd.function,
        z: torch.Tensor,
        _lambda: torch.Tensor,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        tol: float,
        max_iter: int,
        fft_centered: bool,
        fft_normalization: str,
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """
        Forward pass of the conjugate gradient solver.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object.
        z : torch.Tensor
            Input image. Shape [batch_size, height, width, 2].
        _lambda : torch.Tensor
            Regularization parameter.
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, num_coils, height, width, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, num_coils, height, width, 2].
        mask : torch.Tensor
            Subsampling mask. Shape [batch_size, 1, height, width, 1].
        tol : float
            Tolerance for the stopping criterion.
        max_iter : int
            Maximum number of iterations.
        fft_centered : bool
            Whether to center the FFT.
        fft_normalization : str
            FFT normalization.
        spatial_dims : List[int]
            Spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output image. Shape [batch_size, height, width, 2].
        """
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.fft_centered = fft_centered
        ctx.fft_normalization = fft_normalization
        ctx.spatial_dims = spatial_dims

        def A(x):
            x = (
                fft.fft2(
                    utils.complex_mul(x.expand_as(sensitivity_maps), sensitivity_maps),
                    centered=fft_centered,
                    normalization=fft_normalization,
                    spatial_dims=spatial_dims,
                )
                * mask
            )
            return torch.sum(x, dim=-4, keepdim=True)

        def AT(x):
            return torch.sum(
                utils.complex_mul(
                    fft.ifft2(
                        x * mask, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims
                    ),
                    utils.complex_conj(sensitivity_maps),
                ),
                dim=(-5),
            )

        def M(p):
            return _lambda * AT(A(p)) + p

        x0 = _lambda * AT(y) + z
        ctx.save_for_backward(AT(y), x0, sensitivity_maps, mask, _lambda)

        return ConjugateGradient.solve(x0, M, ctx.tol, ctx.max_iter)  # type: ignore

    @staticmethod
    def backward(
        ctx: torch.autograd.function, grad_x: torch.Tensor
    ) -> tuple[torch.Tensor, Any, None, None, None, None, None, None, None, None]:
        """
        Backward pass of the conjugate gradient solver.

        Parameters
        ----------
        ctx : torch.autograd.function
            Context object.
        grad_x : torch.Tensor
            Gradient of the output image. Shape [batch_size, height, width, 2].

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Gradient of the input image, regularization parameter, ...
        """
        ATy, rhs, sensitivity_maps, mask, _lambda = ctx.saved_tensors

        def A(x):
            x = (
                fft.fft2(
                    utils.complex_mul(x.expand_as(sensitivity_maps), sensitivity_maps),
                    centered=ctx.fft_centered,
                    normalization=ctx.fft_normalization,
                    spatial_dims=ctx.spatial_dims,
                )
                * mask
            )
            return torch.sum(x, dim=-4, keepdim=True)

        def AT(x):
            return torch.sum(
                utils.complex_mul(
                    fft.ifft2(
                        x * mask,
                        centered=ctx.fft_centered,
                        normalization=ctx.fft_normalization,
                        spatial_dims=ctx.spatial_dims,
                    ),
                    utils.complex_conj(sensitivity_maps),
                ),
                dim=(-5),
            )

        def M(p):
            return _lambda * AT(A(p)) + p

        Qe = ConjugateGradient.solve(grad_x, M, ctx.tol, ctx.max_iter)  # type: ignore
        QQe = ConjugateGradient.solve(Qe, M, ctx.tol, ctx.max_iter)  # type: ignore

        grad_z = Qe

        grad_lambda = (
            utils.complex_mul(
                fft.ifft2(
                    Qe, centered=ctx.fft_centered, normalization=ctx.fft_normalization, spatial_dims=ctx.spatial_dims
                ),
                utils.complex_conj(ATy),
            ).sum()
            - utils.complex_mul(
                fft.ifft2(
                    QQe, centered=ctx.fft_centered, normalization=ctx.fft_normalization, spatial_dims=ctx.spatial_dims
                ),
                utils.complex_conj(rhs),
            ).sum()
        )

        return grad_z, grad_lambda, None, None, None, None, None, None, None, None


class DataVSLayer(torch.nn.Module):
    """
    DataLayer using variable splitting formulation.

    Parameters
    ----------
    alpha_init : float
        Initial value for the regularization parameter alpha.
    beta_init : float
        Initial value for the regularization parameter beta.
    learnable : bool
        If True, the data term weight lambda is learnable. Default is ``True``.
    fft_centered : bool
        If True, the FFT is centered. Default is ``False``.
    fft_normalization : str
        If "ortho", the FFT is normalized. Default is ``"backward"``.
    spatial_dims : tuple of int
        If not None, the spatial dimensions of the FFT. Default is ``None``.
    """

    def __init__(
        self,
        alpha_init: float,
        beta_init: float,
        learnable: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        super(DataVSLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.alpha.data = torch.tensor(alpha_init, dtype=self.alpha.dtype)

        self.beta = torch.nn.Parameter(torch.Tensor(1))
        self.beta.data = torch.tensor(beta_init, dtype=self.beta.dtype)

        self.learnable = learnable
        self.set_learnable(learnable)

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, sensitivity_maps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the data layer.

        Parameters
        ----------
        x : torch.Tensor
            Prediction. Shape [batch_size, num_coils, height, width, 2].
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, num_coils, height, width, 2].
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, num_coils, height, width, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1, num_channels, height, width, 1].

        Returns
        -------
        torch.Tensor
            Data loss term.
        """
        A_x = torch.sum(
            fft.fft2(
                utils.complex_mul(x.unsqueeze(-5).expand_as(sensitivity_maps), sensitivity_maps),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ),
            -4,
            keepdim=True,
        )
        k_dc = (1 - mask) * A_x + mask * (self.alpha * A_x + (1 - self.alpha) * y)
        x_dc = torch.sum(
            utils.complex_mul(
                fft.ifft2(
                    k_dc,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                utils.complex_conj(sensitivity_maps),
            ),
            dim=(-5),
        )
        return self.beta * x + (1 - self.beta) * x_dc

    def set_learnable(self, flag: bool):
        """
        Set the learnable flag of the parameters.

        Parameters
        ----------
        flag : bool
            If True, the parameters are learnable.
        """
        self.learnable = flag
        self.alpha.requires_grad = self.learnable
        self.beta.requires_grad = self.learnable


class DCLayer(torch.nn.Module):
    """
    Data Consistency layer from DC-CNN, apply for single coil mainly.

    Parameters
    ----------
    lambda_init : float
        Initial value of data term weight lambda.
    learnable : bool
        If True, the data term weight lambda is learnable. Default is ``True``.
    fft_centered : bool
        If True, the FFT is centered. Default is ``False``.
    fft_normalization : str
        If "ortho", the FFT is normalized. Default is ``"backward"``.
    spatial_dims : tuple of int
        If not None, the spatial dimensions of the FFT. Default is ``None``.
    """

    def __init__(
        self,
        lambda_init: float,
        learnable: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        super(DCLayer, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.Tensor(1))
        self.lambda_.data = torch.tensor(lambda_init, dtype=self.lambda_.dtype)

        self.learnable = learnable
        self.set_learnable(learnable)

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the data layer.

        Parameters
        ----------
        x : torch.Tensor
            Prediction. Shape [batch_size, num_coils, height, width, 2].
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, num_coils, height, width, 2].
        mask : torch.Tensor
            Sampling mask. Shape [batch_size, 1, num_channels, height, width, 1].

        Returns
        -------
        torch.Tensor
            Data loss term.
        """
        A_x = fft.fft2(
            x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )
        k_dc = (1 - mask) * A_x + mask * (self.lambda_ * A_x + (1 - self.lambda_) * y)
        return fft.ifft2(
            k_dc, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )

    def set_learnable(self, flag: bool):
        """
        Set the learnable flag of the parameters.

        Parameters
        ----------
        flag : bool
            If True, the parameters are learnable.
        """
        self.learnable = flag
        self.lambda_.requires_grad = self.learnable
