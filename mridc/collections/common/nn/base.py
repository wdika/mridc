# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Dict, Optional, Sequence, Tuple

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.nn.unet_base.unet_block as unet_block
import mridc.core.classes.modelPT as modelPT
import mridc.utils.model_utils as model_utils
from mridc.collections.common.parts.fft import ifft2

wandb.require("service")

__all__ = ["DistributedMetricSum", "BaseMRIModel", "BaseSensitivityModel"]


class DistributedMetricSum(Metric):
    """
    A metric that sums the values of a metric across all workers.
    Taken from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/mri_module.py

    Parameters
    ----------
    dist_sync_on_step: bool
        Synchronize metric state across processes at each ``forward()`` before returning the value at the step.

    Returns
    -------
    metric: torch.FloatTensor, shape [1]
        The metric value.

    Examples
    --------
    >>> metric = DistributedMetricSum()
    >>> metric(torch.tensor(1.0))
    >>> metric(torch.tensor(2.0))
    >>> metric.compute()
    tensor(3.)
    """

    full_state_update: bool = True

    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        """Update the metric with a batch of data."""
        self.quantity += batch

    def compute(self):
        """Compute the metric value."""
        return self.quantity


class BaseMRIModel(modelPT.ModelPT, ABC):  # type: ignore
    """
    Base class for (any task performed on) MRI models.

    Parameters
    ----------
    cfg : DictConfig
        The configuration file.
    trainer : Trainer
        The PyTorch Lightning trainer.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.log_images = cfg_dict.get("log_images", True)

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a training step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the loss and the log.
        """
        raise NotImplementedError

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the loss and the log.
        """
        raise NotImplementedError

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """
        Performs a test step.

        Parameters
        ----------
        batch : Dict[float, torch.Tensor]
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        Tuple[str, int, torch.Tensor]
            Tuple with the filename, the slice index and the prediction.
        """
        raise NotImplementedError

    def log_image(self, name, image):
        """
        Logs an image.

        Parameters
        ----------
        name : str
            Name of the image.
        image : torch.Tensor
            Image to log.
        """
        if image.dim() > 3:
            image = image[0, 0, :, :].unsqueeze(0)
        elif image.shape[0] != 1:
            image = image[0].unsqueeze(0)

        if "wandb" in self.logger.__module__.lower():
            if image.is_cuda:
                image = image.detach().cpu()
            self.logger.experiment.log({name: wandb.Image(image.numpy())})
        elif "tensorboard" in self.logger.__module__.lower():
            self.logger.experiment.add_image(name, image, global_step=self.global_step)
        else:
            raise NotImplementedError(f"Logging images is not implemented for {self.logger.__module__}.")

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation epoch to aggregate outputs.

        Parameters
        ----------
        outputs : List[Dict[str, torch.Tensor]]
            List of outputs from validation steps.

        Returns
        -------
        metrics : Dict[str, torch.Tensor]
            Dictionary with the aggregated metrics.
        """
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        """
        Called at the end of test epoch to aggregate outputs and save predictions.

        Parameters
        ----------
        outputs : List[Dict[str, torch.Tensor]]
            List of outputs from validation steps.

        Returns
        -------
        metrics : Dict[str, torch.Tensor]
            Dictionary with the aggregated metrics.
        """
        raise NotImplementedError

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        """
        Setups the training data.

        Parameters
        ----------
        train_data_config: Training data configuration.
            dict

        Returns
        -------
        train_data: Training data.
            torch.utils.data.DataLoader
        """
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        """
        Setups the validation data.

        Parameters
        ----------
        val_data_config: Validation data configuration.
            dict

        Returns
        -------
        val_data: Validation data.
            torch.utils.data.DataLoader
        """
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        """
        Setups the test data.

        Parameters
        ----------
        test_data_config: Test data configuration.
            dict

        Returns
        -------
        test_data: Test data.
            torch.utils.data.DataLoader
        """
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    @staticmethod
    def _setup_dataloader_from_config(cfg: DictConfig) -> DataLoader:
        """
        Setups the dataloader from the configuration (yaml) file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration file.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            Dataloader.
        """
        raise NotImplementedError


class BaseSensitivityModel(nn.Module, ABC):
    """
    Model for learning sensitivity estimation from k-space data [1]. This model applies an IFFT to multichannel
    k-space data and then a U-Net to the coil images to estimate coil sensitivities.

    References
    ----------
    .. [1] Sriram A, Zbontar J, Murrell T, Defazio A, Zitnick CL, Yakubova N, Knoll F, Johnson P. End-to-end
        variational networks for accelerated MRI reconstruction. InInternational Conference on Medical Image Computing
        and Computer-Assisted Intervention 2020 Oct 4 (pp. 64-73). Springer, Cham.

    Parameters
    ----------
    chans : int
        Number of channels in the input k-space data. Default is ``8``.
    num_pools : int
        Number of U-Net downsampling/upsampling operations. Default is ``4``.
    in_chans : int
        Number of input channels to the U-Net. Default is ``2``.
    out_chans : int
        Number of output channels to the U-Net. Default is ``2``.
    drop_prob : float
        Dropout probability. Default is ``0.0``.
    padding_size : int
        Padding size for the U-Net. Default is ``15``.
    mask_type : str
        If kspace is undersampled, the undersampling mask type must be specified. Default is ``"2D"``.
    fft_centered : bool
        If ``True``, the input data is assumed to be centered. Default is ``False``.
    fft_normalization : str
        Normalization for the IFFT. Default is ``"backward"``.
    spatial_dims : Sequence[int]
        Spatial dimensions of the input data. Default is ``None``.
    coil_dim : int
        Dimension of the coils. Default is ``1``.
    normalize : bool
        If ``True``, the input data is normalized. Default is ``True``.

    Returns
    -------
    torch.Tensor
        Estimated coil sensitivity maps.

    Examples
    --------
    >>> from mridc.collections.common.models.base import BaseSensitivityModel
    >>> import torch
    >>> model = BaseSensitivityModel()
    >>> kspace = torch.randn([1, 8, 320, 320, 2], dtype=torch.float32)
    >>> coil_sensitivity_maps = model(kspace)
    >>> coil_sensitivity_maps.shape
    torch.Size([1, 8, 320, 320, 2])
    """

    def __init__(
        self,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        mask_type: str = "2D",
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 1,
        normalize: bool = True,
        mask_center: bool = True,
    ):
        super().__init__()

        if mask_type != "2D":
            raise ValueError("Currently only 2D masks are supported for coil sensitivity estimation.")
        self.mask_type = mask_type  # TODO: make this generalizable

        self.norm_unet = unet_block.NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            padding_size=padding_size,
            normalize=normalize,
        )

        self.mask_center = mask_center
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
        self.normalize = normalize

    @staticmethod
    def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Combines the channel dimension with the batch dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to convert.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Tuple of the converted tensor and the batch size.
        """
        batch_size, coils, height, width, complex_dim = x.shape
        return x.view(batch_size * coils, 1, height, width, complex_dim), batch_size

    @staticmethod
    def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Splits the batch and channel dimensions into the channel dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to convert.
        batch_size : int
            Batch size.

        Returns
        -------
        torch.Tensor
            Converted tensor.
        """
        batch_size_coils, _, height, width, complex_dim = x.shape
        coils = torch.div(batch_size_coils, batch_size, rounding_mode="trunc")
        return x.view(batch_size, coils, height, width, complex_dim)

    @staticmethod
    def divide_root_sum_of_squares(x: torch.Tensor, coil_dim: int) -> torch.Tensor:
        """
        Divide the input by the root of the sum of squares of the magnitude of each complex number.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to divide.
        coil_dim : int
            Dimension of the coils.

        Returns
        -------
        torch.Tensor
            Normalized tensor by the root sum of squares.
        """
        return x / utils.rss_complex(x, dim=coil_dim).unsqueeze(-1).unsqueeze(coil_dim)

    @staticmethod
    def get_pad_and_num_low_freqs(
        mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the padding to apply to the input to make it square and the number of low frequencies to keep.

        Parameters
        ----------
        mask : torch.Tensor
            Mask to use to determine the padding and number of low frequencies.
        num_low_frequencies : Optional[int]
            Number of low frequencies to keep. Default is ``None``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the padding to apply to the input and the number of low frequencies to keep.
        """
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = torch.div(squeezed_mask.shape[1], 2, rounding_mode="trunc")
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = torch.div(mask.shape[-2] - num_low_frequencies_tensor + 1, 2, rounding_mode="trunc")

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        masked_kspace : torch.Tensor
            Subsampled k-space data of shape [batch_size, n_coils, n_x, n_y, 2].
        mask : torch.Tensor
            Sampling mask of shape [batch_size, 1, n_x, n_y, 1].
        num_low_frequencies : Optional[int]
            Number of low frequencies to keep. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Estimated coil sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2].
        """
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = utils.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs, mask_type=self.mask_type
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(
            ifft2(
                masked_kspace,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        )

        # estimate sensitivities
        images = self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        if self.normalize:
            images = self.divide_root_sum_of_squares(images, self.coil_dim)
        return images
