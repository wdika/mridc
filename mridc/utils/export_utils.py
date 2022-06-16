# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/export_utils.py
import os
from enum import Enum
from typing import Callable, Dict, Optional, Type

import onnx
import torch
import torch.nn as nn

from mridc.utils import logging

try:
    import onnxruntime

    ort_available = True
except ImportError:
    ort_available = False


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {".pt": ExportFormat.TORCHSCRIPT, ".ts": ExportFormat.TORCHSCRIPT, ".onnx": ExportFormat.ONNX}


class CastToFloat(nn.Module):
    """Cast input to float"""

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        """Forward pass"""
        return self.mod.forward(x.to(torch.float).to(x.dtype)) if torch.is_autocast_enabled() else self.mod.forward(x)


def get_export_format(filename: str):
    """Get export format from filename"""
    _, ext = os.path.splitext(filename)
    try:
        return _EXT_DICT[ext]
    except KeyError as e:
        raise ValueError(f"Export file {filename} extension does not correspond to any export format!") from e


def augment_filename(output: str, prepend: str):
    """Augment output filename with prepend"""
    if prepend == "self":
        return output
    path, filename = os.path.split(output)
    filename = f"{prepend}-{filename}"
    return os.path.join(path, filename)


def forward_method(self):
    """Forward method for export"""
    if hasattr(self, "forward_for_export"):
        return self.forward_for_export
    return self.forward


def wrap_forward_method(self):
    """Wraps the forward method of the module with a function that returns the output of the forward method"""
    tp = type(self)
    old_forward_method = None
    if hasattr(tp, "forward_for_export"):
        forward_method = tp.forward_for_export
        old_forward_method = tp.forward
        tp.forward = forward_method
    else:
        forward_method = None
    return forward_method, old_forward_method


def parse_input_example(input_example):
    """Parse input example to onnxrt input format"""
    input_list = list(input_example)
    input_dict = {}
    # process possible kwargs
    if isinstance(input_list[-1], dict):
        input_dict = input_list[-1]
        input_list = input_list[:-1]
    return input_list, input_dict


def to_onnxrt_input(input_names, input_dict, input_list):
    """Transforms input to onnxrt input format"""
    return {
        k: input_dict[k].cpu().numpy() if k in input_dict else input_list.pop().cpu().numpy()
        for k in reversed(input_names)
    }


def verify_runtime(
    output,
    input_list,
    input_dict,
    input_names,
    output_names,
    output_example,
    check_tolerance=0.01,
):
    """
    Verify runtime output with onnxrt.

    Parameters
    ----------
    output: The output of the module.
    input_list: The input list of the module.
    input_dict: The input dict of the module.
    input_names: The input names of the module.
    output_names: The output names of the module.
    output_example: The output example of the module.
    check_tolerance: The tolerance for the check.

    Returns
    -------
    The runtime output.
    """
    # Verify the model can be read, and is valid
    onnx_model = onnx.load(output)
    input_names = [node.name for node in onnx_model.graph.input]
    # skipcq: PYL-W0622
    global ort_available
    if not ort_available:
        logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime_gpu package.\n")
        onnx.checker.check_model(onnx_model, full_check=True)
        return

    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), sess_options=onnx_session_opt, providers=["CUDAExecutionProvider"]
    )
    ort_out = sess.run(output_names, to_onnxrt_input(input_names, input_dict, input_list))
    all_good = True

    for i, out in enumerate(ort_out[0]):
        expected = output_example[i]
        if torch.is_tensor(expected):
            tout = torch.from_numpy(out)
            if not torch.allclose(tout, expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance):
                all_good = False
                logging.info(f"onnxruntime results mismatch! PyTorch(expected):\n{expected}\nONNXruntime:\n{tout}")
    status = "SUCCESS" if all_good else "FAIL"
    logging.info(f"ONNX generated at {output} verified with onnxruntime : {status}")
    return all_good


def simple_replace(BaseT: Type[nn.Module], DestT: Type[nn.Module]) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT. BaseT and DestT should have same attributes.
    No weights are copied.

    Parameters
    ----------
    BaseT: The base type of the module.
    DestT: The destination type of the module.

    Returns
    -------
    A function to replace BaseT with DestT.
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        """Swap function to replace BaseT module with DestT"""
        if not isinstance(mod, BaseT):
            return None
        args = [getattr(mod, name, None) for name in mod.__constants__]
        return DestT(*args)

    return expansion_fn


def wrap_module(BaseT: Type[nn.Module], DestT: Type[nn.Module]) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    Generic function generator to replace BaseT module with DestT. BaseT and DestT should have same attributes.
    No weights are copied.

    Parameters
    ----------
    BaseT: The base type of the module.
    DestT: The destination type of the module.

    Returns
    -------
    A function to replace BaseT with DestT.
    """

    def expansion_fn(mod: nn.Module) -> Optional[nn.Module]:
        """Expansion function to replace BaseT module with DestT"""
        return DestT(mod)

    return expansion_fn


def swap_modules(model: nn.Module, mapping: Dict[str, nn.Module]):
    """
    This function swaps nested modules as specified by "dot paths" in mod with a desired replacement. This allows
    for swapping nested modules through arbitrary levels if children
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.
    """
    for path, new_mod in mapping.items():
        expanded_path = path.split(".")
        parent_mod = model
        for sub_path in expanded_path[:-1]:
            parent_mod = parent_mod._modules[sub_path]  # noqa
        parent_mod._modules[expanded_path[-1]] = new_mod  # noqa

    return model


def replace_modules(
    model: nn.Module, expansions: Dict[str, Callable[[nn.Module], Optional[nn.Module]]] = None
) -> nn.Module:
    """
    Top-level function to replace modules in model, specified by class name with a desired replacement.
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    Parameters
    ----------
    model: Top-level model to replace modules in.
    expansions: A dictionary of module class names to functions to replace them with.

    Returns
    -------
    The model with replaced modules.
    """
    mapping: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        m_type = type(m).__name__
        if m_type in expansions:  # type: ignore
            if swapped := expansions[m_type](m):  # type: ignore
                mapping[name] = swapped
    logging.warning(f"Swapped {len(mapping)} modules")
    swap_modules(model, mapping)
    return model


default_replacements = {
    "BatchNorm1d": wrap_module(nn.BatchNorm1d, CastToFloat),
    "BatchNorm2d": wrap_module(nn.BatchNorm2d, CastToFloat),
    "LayerNorm": wrap_module(nn.LayerNorm, CastToFloat),
}


def replace_for_export(model: nn.Module) -> nn.Module:
    """
    Top-level function to replace default set of modules in model
    NOTE: This occurs in place, if you want to preserve model then make sure to copy it first.

    Parameters
    ----------
    model: Top-level model to replace modules in.

    Returns
    -------
    The model with replaced modules.
    """
    replace_modules(model, default_replacements)
