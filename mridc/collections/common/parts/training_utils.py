# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/parts/training_utils.py

from contextlib import nullcontext

import torch

__all__ = ["avoid_bfloat16_autocast_context", "avoid_float16_autocast_context"]


def avoid_bfloat16_autocast_context():
    """If the current autocast context is bfloat16, cast it to float32."""
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return nullcontext()


def avoid_float16_autocast_context():
    """If the current autocast context is float16, cast it to bfloat16 if available or float32."""
    if not torch.is_autocast_enabled() or torch.get_autocast_gpu_dtype() != torch.float16:
        return nullcontext()
    if torch.cuda.is_bf16_supported():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return torch.cuda.amp.autocast(dtype=torch.float32)
