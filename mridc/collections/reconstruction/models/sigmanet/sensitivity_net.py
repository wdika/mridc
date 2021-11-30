# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/khammernik/sigmanet/blob/master/reconstruction/common/mytorch/models/sn.py
import numpy as np
import torch


def matrix_invert(xx, xy, yx, yy):
    """Invert a 2x2 matrix."""
    det = xx * yy - xy * yx
    return yy.div(det), -xy.div(det), -yx.div(det), xx.div(det)


class ComplexInstanceNorm(torch.nn.Module):
    """Motivated by 'Deep Complex Networks' (https://arxiv.org/pdf/1705.09792.pdf)"""

    def __init__(self):
        super(ComplexInstanceNorm, self).__init__()
        self.mean = 0
        self.cov_xx_half = 1 / np.sqrt(2)
        self.cov_xy_half = 0
        self.cov_yx_half = 0
        self.cov_yy_half = 1 / np.sqrt(2)

    def complex_instance_norm(self, x, eps=1e-5):
        """Operates on images x of size [nBatch, nSmaps, nFE, nPE, 2]"""
        x_combined = torch.sum(x, dim=1, keepdim=True)
        mean = x_combined.mean(dim=(1, 2, 3), keepdim=True)
        x_m = x - mean
        self.mean = mean
        self.complex_pseudocovariance(x_m)

    def complex_pseudocovariance(self, data):
        """Data variable hast to be already mean-free! Operates on images x of size [nBatch, nSmaps, nFE, nPE, 2]"""
        if data.size(-1) != 2:
            raise AssertionError
        shape = data.shape

        # compute number of elements
        N = shape[2] * shape[3]

        # separate real/imaginary channel
        re, im = torch.unbind(data, dim=-1)

        # dimensions is now length of original shape - 1 (because channels are seperated)
        dim = list(range(1, len(shape) - 1))

        # compute covariance entries. cxy = cyx
        cxx = (re * re).sum(dim=dim, keepdim=True) / (N - 1)
        cyy = (im * im).sum(dim=dim, keepdim=True) / (N - 1)
        cxy = (re * im).sum(dim=dim, keepdim=True) / (N - 1)

        # Eigenvalue decomposition C = V*S*inv(V)
        # compute eigenvalues
        s1 = (cxx + cyy) / 2 - torch.sqrt((cxx + cyy) ** 2 / 4 - cxx * cyy + cxy**2)
        s2 = (cxx + cyy) / 2 + torch.sqrt((cxx + cyy) ** 2 / 4 - cxx * cyy + cxy**2)

        # compute eigenvectors
        v1x = s1 - cyy
        v1y = cxy
        v2x = s2 - cyy
        v2y = cxy

        # normalize eigenvectors
        norm1 = torch.sqrt(torch.sum(v1x * v1x + v1y * v1y, dim=dim, keepdim=True))
        norm2 = torch.sqrt(torch.sum(v2x * v2x + v2y * v2y, dim=dim, keepdim=True))

        v1x = v1x.div(norm1)
        v1y = v1y.div(norm1)

        v2x = v2x.div(norm2)
        v2y = v2y.div(norm2)

        # now we need the sqrt of the covariance matrix.
        # C^{-0.5} = V * sqrt(S) * inv(V)
        det = v1x * v2y - v2x * v1y
        s1 = torch.sqrt(s1).div(det)
        s2 = torch.sqrt(s2).div(det)

        self.cov_xx_half = v1x * v2y * s1 - v1y * v2x * s2
        self.cov_yy_half = v1x * v2y * s2 - v1y * v2x * s1
        self.cov_xy_half = v1x * v2x * (s2 - s1)
        self.cov_yx_half = v1y * v2y * (s1 - s2)

    def forward(self, input):
        """Operates on images x of size [nBatch, nSmaps, nFE, nPE, 2]"""
        return self.normalize(input)

    def set_normalization(self, input):
        """Set the normalization parameters for a given input."""
        mean = torch.tensor([torch.mean(input).item()]).to(input)
        self.complex_pseudocovariance(input - mean)
        self.mean = mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.cov_xx_half = self.cov_xx_half.view(-1, 1, 1, 1)
        self.cov_xy_half = self.cov_xy_half.view(-1, 1, 1, 1)
        self.cov_yx_half = self.cov_yx_half.view(-1, 1, 1, 1)
        self.cov_yy_half = self.cov_yy_half.view(-1, 1, 1, 1)

    def normalize(self, x):
        """Normalize the input x."""
        x_m = x - self.mean
        re, im = torch.unbind(x_m, dim=-1)

        cov_xx_half_inv, cov_xy_half_inv, cov_yx_half_inv, cov_yy_half_inv = matrix_invert(
            self.cov_xx_half, self.cov_xy_half, self.cov_yx_half, self.cov_yy_half
        )
        x_norm_re = cov_xx_half_inv * re + cov_xy_half_inv * im
        x_norm_im = cov_yx_half_inv * re + cov_yy_half_inv * im
        img = torch.stack([x_norm_re, x_norm_im], dim=-1)
        img = img.clamp(-6, 6)
        return img

    def unnormalize(self, x):
        """Unnormalize the input x."""
        re, im = torch.unbind(x, dim=-1)
        x_unnorm_re = self.cov_xx_half * re + self.cov_xy_half * im
        x_unnorm_im = self.cov_yx_half * re + self.cov_yy_half * im
        return torch.stack([x_unnorm_re, x_unnorm_im], dim=-1) + self.mean


class ComplexNormWrapper(torch.nn.Module):
    """Wrapper for complex normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.complex_instance_norm = ComplexInstanceNorm()

    def forward(self, input):
        # compute complex instance norm on sample of size [nBatch, nSmaps, nFE, nPE, 2]
        self.complex_instance_norm.set_normalization(input)
        output = self.complex_instance_norm.normalize(input)

        # re-shape data from [nBatch, nSmaps, nFE, nPE, 2] to [nBatch*nSmaps, 2, nFE, nPE]
        shp = output.shape
        output = output.view(shp[0] * shp[1], *shp[2:]).permute(0, 3, 1, 2)

        # apply denoising
        output = self.model(output)

        # re-shape data from [nBatch*nSmaps, 2, nFE, nPE]
        # to [nBatch, nSmaps, nFE, nPE, 2]
        output = output.permute(0, 2, 3, 1).view(*shp)
        # unnormalize
        output = self.complex_instance_norm.unnormalize(output)
        return output


class SensitivityNetwork(torch.nn.Module):
    """Sensitivity network with data term based on forward and adjoint containing the sensitivity maps"""

    def __init__(
        self,
        num_iter,
        model,
        datalayer,
        shared_params=True,
        save_space=False,
        reset_cache=False,
    ):
        """

        Parameters
        ----------
        num_iter: Number of iterations.
        model: Model to be used for the forward and adjoint.
        datalayer: Data layer to be used for the forward and adjoint.
        shared_params: If True, the parameters of the model are shared between the forward and adjoint.
        save_space: If True, the adjoint is computed in the forward pass.
        reset_cache: If True, the adjoint is computed in the forward pass.
        """
        super().__init__()

        self.shared_params = shared_params

        self.num_iter = 1 if self.shared_params else num_iter
        self.num_iter_total = num_iter

        self.is_trainable = [True] * num_iter

        # setup the modules
        self.gradR = torch.nn.ModuleList([ComplexNormWrapper(model) for _ in range(self.num_iter)])

        self.gradD = torch.nn.ModuleList([datalayer for _ in range(self.num_iter)])

        self.save_space = save_space
        if self.save_space:
            self.forward = self.forward_save_space
        self.reset_cache = reset_cache

    def forward(self, x, y, smaps, mask):
        """

        Parameters
        ----------
        x: Input data.
        y: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.

        Returns
        -------
        Output data.
        """
        x_all = [x]
        x_half_all = []
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)

        for i in range(num_iter):
            x_thalf = x - self.gradR[i % self.num_iter](x)
            x = self.gradD[i % self.num_iter](x_thalf, y, smaps, mask)
            x_all.append(x)
            x_half_all.append(x_thalf)

        return x_all[-1]

    def forward_save_space(self, x, y, smaps, mask):
        """

        Parameters
        ----------
        x: Input data.
        y: Subsampled k-space data.
        smaps: Coil sensitivity maps.
        mask: Sampling mask.

        Returns
        -------
        Output data.
        """
        if self.shared_params:
            num_iter = self.num_iter_total
        else:
            num_iter = min(np.where(self.is_trainable)[0][-1] + 1, self.num_iter)

        for i in range(num_iter):
            x_thalf = x - self.gradR[i % self.num_iter](x)
            x = self.gradD[i % self.num_iter](x_thalf, y, smaps, mask)

            # would run out of memory at test time
            # if this is False for some cases
            if self.reset_cache:
                torch.cuda.empty_cache()
                torch.backends.cuda.cufft_plan_cache.clear()

        return x

    def freeze(self, i):
        """freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = False
        self.is_trainable[i] = False

    def unfreeze(self, i):
        """freeze parameter of cascade i"""
        for param in self.gradR[i].parameters():
            param.require_grad_ = True
        self.is_trainable[i] = True

    def freeze_all(self):
        """freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.freeze(i)

    def unfreeze_all(self):
        """freeze parameter of cascade i"""
        for i in range(self.num_iter):
            self.unfreeze(i)

    def copy_params(self, src_i, trg_j):
        """copy i-th cascade net parameters to j-th cascade net parameters"""
        src_params = self.gradR[src_i].parameters()
        trg_params = self.gradR[trg_j].parameters()

        for trg_param, src_param in zip(trg_params, src_params):
            trg_param.data.copy_(src_param.data)

    def stage_training_init(self):
        """set stage training flag to True"""
        self.freeze_all()
        self.unfreeze(0)
        print(self.is_trainable)

    def stage_training_transition_i(self, copy=False):
        """set stage training flag to True"""
        if self.shared_params:
            return

        # if all unlocked, don't do anything
        if not np.all(self.is_trainable):
            for i in range(self.num_iter):

                # if last cascade is reached, unlock all
                if i == self.num_iter - 1:
                    self.unfreeze_all()
                    break

                # freeze current i, unlock next. copy parameter if specified
                if self.is_trainable[i]:
                    self.freeze(i)
                    self.unfreeze(i + 1)
                    if copy:
                        self.copy_params(i, i + 1)
                    break
