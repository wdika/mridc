# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/khammernik/sigmanet/blob/master/reconstruction/common/mytorch/models/datalayer.py

from typing import Optional, Tuple

import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import complex_abs, complex_conj, complex_mul


class DataIDLayer(torch.nn.Module):
    """Placeholder for the data layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class DataGDLayer(torch.nn.Module):
    """DataLayer computing the gradient on the L2 dataterm."""

    def __init__(
        self,
        lambda_init,
        learnable=True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        lambda_init: Init value of data term weight lambda.
        learnable: If True, the data term weight lambda is learnable.
        fft_centered: If True, the FFT is centered.
        fft_normalization: If "ortho", the FFT is normalized.
        spatial_dims: If not None, the spatial dimensions of the FFT.
        """
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

    def forward(self, x, y, smaps, mask):
        """

        Parameters
        ----------
        x: Input image.
        y: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.

        Returns
        -------
        data_loss: Data term loss.
        """
        A_x_y = (
            torch.sum(
                fft2(
                    complex_mul(x.unsqueeze(-5).expand_as(smaps), smaps),
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
            complex_mul(
                ifft2c(
                    A_x_y * mask,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                complex_conj(smaps),
            ),
            dim=(-5),
        )
        return x - self.data_weight * gradD_x


class DataProxCGLayer(torch.nn.Module):
    """Solving the prox wrt. dataterm using Conjugate Gradient as proposed by Aggarwal et al."""

    def __init__(
        self,
        lambda_init,
        tol=1e-6,
        iter=10,
        learnable=True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        super(DataProxCGLayer, self).__init__()

        self.lambdaa = torch.nn.Parameter(torch.Tensor(1))
        self.lambdaa.data = torch.tensor(lambda_init)
        self.lambdaa_init = lambda_init
        self.lambdaa.requires_grad = learnable

        self.tol = tol
        self.iter = iter

        self.op = ConjugateGradient

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(self, x, f, smaps, mask):
        """

        Parameters
        ----------
        x: Input image.
        f: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.

        Returns
        -------
        data_loss: Data term loss.
        """
        return self.op.apply(
            x,
            self.lambdaa,
            f,
            smaps,
            mask,
            self.tol,
            self.iter,
            self.fft_centered,
            self.fft_normalization,
            self.spatial_dims,
        )

    def set_learnable(self, flag):
        self.lambdaa.requires_grad = flag


class ConjugateGradient(torch.autograd.Function):
    """Conjugate Gradient solver for the prox of the data term."""

    @staticmethod
    def complexDot(data1, data2):
        """Complex dot product of two tensors."""
        nBatch = data1.shape[0]
        mult = complex_mul(data1, complex_conj(data2))
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re.view(nBatch, -1), dim=-1), torch.sum(im.view(nBatch, -1), dim=-1)], -1)

    @staticmethod
    def solve(x0, M, tol, max_iter):
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
            alpha = torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], -1) / complex_abs(data2) ** 2

            x += complex_mul(alpha.reshape(nBatch, 1, 1, 1, -1), p.clone())
            r -= complex_mul(alpha.reshape(nBatch, 1, 1, 1, -1), q.clone())
            rr_new = torch.stack([(r.pow(2)).view(nBatch, -1).sum(-1), torch.zeros(nBatch).to(x0.device)], dim=-1)
            beta = torch.stack([rr_new[..., 0] / rr[..., 0], torch.zeros(nBatch).to(x0.device)], dim=-1)
            p = r.clone() + complex_mul(beta.reshape(nBatch, 1, 1, 1, -1), p)
            rr = rr_new.clone()
        return x

    @staticmethod
    def forward(ctx, z, lambdaa, y, smaps, mask, tol, max_iter, fft_centered, fft_normalization, spatial_dims):
        """
        Forward pass of the conjugate gradient solver.

        Parameters
        ----------
        ctx: Context object.
        z: Input image.
        lambdaa: Regularization parameter.
        y: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.
        tol: Tolerance for the stopping criterion.
        max_iter: Maximum number of iterations.
        fft_centered: Boolean flag for centering the FFT.
        fft_normalization: Boolean flag for normalizing the FFT.
        spatial_dims: Spatial dimensions.

        Returns
        -------
        z: Output image.
        """
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.fft_centered = fft_centered
        ctx.fft_normalization = fft_normalization
        ctx.spatial_dims = spatial_dims

        def A(x):
            x = (
                fft2(
                    complex_mul(x.expand_as(smaps), smaps),
                    centered=fft_centered,
                    normalization=fft_normalization,
                    spatial_dims=spatial_dims,
                )
                * mask
            )
            return torch.sum(x, dim=-4, keepdim=True)

        def AT(x):
            return torch.sum(
                complex_mul(
                    ifft2(x * mask, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims),
                    complex_conj(smaps),
                ),
                dim=(-5),
            )

        def M(p):
            return lambdaa * AT(A(p)) + p

        x0 = lambdaa * AT(y) + z
        ctx.save_for_backward(AT(y), x0, smaps, mask, lambdaa)

        return ConjugateGradient.solve(x0, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass of the conjugate gradient solver.

        Parameters
        ----------
        ctx: Context object.
        grad_x: Gradient of the output image.

        Returns
        -------
        grad_z: Gradient of the input image.
        """
        ATy, rhs, smaps, mask, lambdaa = ctx.saved_tensors

        def A(x):
            x = (
                fft2(
                    complex_mul(x.expand_as(smaps), smaps),
                    centered=ctx.fft_centered,
                    normalization=ctx.fft_normalization,
                    spatial_dims=ctx.spatial_dims,
                )
                * mask
            )
            return torch.sum(x, dim=-4, keepdim=True)

        def AT(x):
            return torch.sum(
                complex_mul(
                    ifft2(
                        x * mask,
                        centered=ctx.fft_centered,
                        normalization=ctx.fft_normalization,
                        spatial_dims=ctx.spatial_dimso,
                    ),
                    complex_conj(smaps),
                ),
                dim=(-5),
            )

        def M(p):
            return lambdaa * AT(A(p)) + p

        Qe = ConjugateGradient.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = ConjugateGradient.solve(Qe, M, ctx.tol, ctx.max_iter)

        grad_z = Qe

        grad_lambdaa = (
            complex_mul(
                ifft2(
                    Qe, centered=ctx.fft_centered, normalization=ctx.fft_normalization, spatial_dims=ctx.spatial_dims
                ),
                complex_conj(ATy),
            ).sum()
            - complex_mul(
                ifft2(
                    QQe, centered=ctx.fft_centered, normalization=ctx.fft_normalization, spatial_dims=ctx.spatial_dims
                ),
                complex_conj(rhs),
            ).sum()
        )

        return grad_z, grad_lambdaa, None, None, None, None, None, None


class DataVSLayer(torch.nn.Module):
    """
    DataLayer using variable splitting formulation
    """

    def __init__(
        self,
        alpha_init,
        beta_init,
        learnable=True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        alpha_init: Init value of data consistency block (DCB)
        beta_init: Init value of weighted averaging block (WAB)
        learnable: If True, the parameters of the model are learnable
        fft_centered: If True, the FFT is centered
        fft_normalization: If "ortho", the FFT is normalized to be orthogonal
        spatial_dims: If not None, the spatial dimensions of the FFT
        """
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

    def forward(self, x, y, smaps, mask):
        """
        Forward pass of the data-consistency block.

        Parameters
        ----------
        x: Input image.
        y: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.

        Returns
        -------
        Output image.
        """
        A_x = torch.sum(
            fft2(
                complex_mul(x.unsqueeze(-5).expand_as(smaps), smaps),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ),
            -4,
            keepdim=True,
        )
        k_dc = (1 - mask) * A_x + mask * (self.alpha * A_x + (1 - self.alpha) * y)
        x_dc = torch.sum(
            complex_mul(
                ifft2(
                    k_dc,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                complex_conj(smaps),
            ),
            dim=(-5),
        )
        return self.beta * x + (1 - self.beta) * x_dc

    def set_learnable(self, flag):
        """
        Set the learnable flag of the parameters.

        Parameters
        ----------
        flag: If True, the parameters of the model are learnable.
        """
        self.learnable = flag
        self.alpha.requires_grad = self.learnable
        self.beta.requires_grad = self.learnable


class DCLayer(torch.nn.Module):
    """
    Data Consistency layer from DC-CNN, apply for single coil mainly
    """

    def __init__(
        self,
        lambda_init=0.0,
        learnable=True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        lambda_init: Init value of data consistency block (DCB)
        learnable: If True, the parameters of the model are learnable
        fft_centered: If True, the FFT is centered
        fft_normalization: If "ortho", the FFT is normalized to be orthogonal
        spatial_dims: If not None, the spatial dimensions of the FFT
        """
        super(DCLayer, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.Tensor(1))
        self.lambda_.data = torch.tensor(lambda_init, dtype=self.lambda_.dtype)

        self.learnable = learnable
        self.set_learnable(learnable)

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

    def forward(self, x, y, mask):
        """
        Forward pass of the data-consistency block.

        Parameters
        ----------
        x: Input image.
        y: Subsampled k-space data.
        mask: Sampling mask.

        Returns
        -------
        Output image.
        """
        A_x = fft2(x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims)
        k_dc = (1 - mask) * A_x + mask * (self.lambda_ * A_x + (1 - self.lambda_) * y)
        return ifft2(
            k_dc, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )

    def set_learnable(self, flag):
        """
        Set the learnable flag of the parameters.

        Parameters
        ----------
        flag: If True, the parameters of the model are learnable.
        """
        self.learnable = flag
        self.lambda_.requires_grad = self.learnable
