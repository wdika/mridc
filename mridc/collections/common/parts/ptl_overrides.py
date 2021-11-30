# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/parts/ptl_overrides.py

import torch
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin


class MRIDCNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    """Native Mixed Precision Plugin for MRIDC."""

    def __init__(self, init_scale: float = 2**32, growth_interval: int = 1000) -> None:
        super().__init__(precision=16, device=self.device)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale, growth_interval=growth_interval)
