# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/debug_hook.py

import os

import torch


def get_forward_hook(name, trainer, rank, logger, dump_to_file=False):
    """
    A forward hook to dump all the module input and output norms. It is called at every time after forward() has
    computed an output. Only float type input/output tensor norms are computed.

    For more details about the forward hook, check:
    https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html

    Parameters
    ----------
    name : str
        tensor name
    trainer : PTL trainer
        PTL trainer
    rank : int
        worker rank
    logger : PTL log function
        PTL log function
    dump_to_file : bool, optional
        wether dump the csv file to the disk, by default False

    Returns
    -------
    forward_hook
    """
    if dump_to_file:
        os.makedirs("debug_info", exist_ok=True)
        fp = open(f"debug_info/forward_{name}_rank{rank}.txt", "w")
        header = False

    def forward_hook(module, inputs, outputs):
        """Forward hook to dump all of the module input and output norms. It is called at every time after forward()
        has computed an output. Only float type input/output tensor norms are computed."""
        nonlocal header
        nonlocal fp
        if trainer.training:
            values = []
            headers = []
            for n, i in enumerate(inputs):
                if isinstance(i, torch.Tensor) and i.dtype in [torch.float, torch.half, torch.bfloat16]:
                    if not header:
                        headers.append("input")
                    input_norm = i.data.norm()
                    values.append(f"{input_norm}")
                    logger(f"debug_info_forward/{name}_rank{rank}_input{n}", input_norm)
            if isinstance(outputs, tuple):
                for n, i in enumerate(outputs):
                    if isinstance(i, torch.Tensor) and i.dtype in [torch.float, torch.half, torch.bfloat16]:
                        if not header:
                            headers.append("output")
                        output_norm = i.data.norm()
                        values.append(f"{output_norm}")
                        logger(f"debug_info_forward/{name}_rank{rank}_output{n}", output_norm)
            else:
                headers.append("output")
                values.append(f"{outputs.data.norm()}")
            values.append(f"{trainer.global_step}")
            if not header:
                headers.append("step")
                fp.write(",".join(headers) + "\n")
                header = True
            fp.write(",".join(values) + "\n")
        fp.flush()

    return forward_hook


def get_backward_hook(name, trainer, rank, logger, dump_to_file=False):
    """
    A backward hook to dump all the module input and output grad norms. The hook will be called every time the \
    gradients with respect to module inputs are computed. Only float type input/output grad tensor norms are computed.

    For more details about the backward hook, check:
    https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html

    Parameters
    ----------
    name : str
        tensor name
    trainer : PTL trainer
        PTL trainer
    rank : int
        worker rank
    logger : PTL log function
        PTL log function
    dump_to_file : bool, optional
        wether dump the csv file to the disk, by default False

    Returns
    -------
    backward_hook
    """
    if dump_to_file:
        os.makedirs("debug_info", exist_ok=True)
        fp = open(f"debug_info/backward_{name}_rank{rank}.txt", "w")
        header = False

    def backward_hook(module, inputs, outputs):
        """Backward hook to dump all the module input and output grad norms. The hook will be called every time the \
        has computed an output. Only float type input/output tensor norms are computed."""
        nonlocal header
        nonlocal fp
        if trainer.training:
            values = []
            headers = []
            for n, i in enumerate(inputs):
                if isinstance(i, torch.Tensor) and i.dtype in [torch.float, torch.half, torch.bfloat16]:
                    if not header:
                        headers.append("input")
                    input_norm = i.data.norm()
                    values.append(f"{input_norm}")
                    logger(f"debug_info_backward/{name}_rank{rank}_input{n}", input_norm)
            if isinstance(outputs, tuple):
                for n, i in enumerate(outputs):
                    if isinstance(i, torch.Tensor) and i.dtype in [torch.float, torch.half, torch.bfloat16]:
                        if not header:
                            headers.append("output")
                        output_norm = i.data.norm()
                        values.append(f"{output_norm}")
                        logger(f"debug_info_backward/{name}_rank{rank}_output{n}", output_norm)
            else:
                headers.append("output")
                values.append(f"{outputs.data.norm()}")
            values.append(f"{trainer.global_step}")
            if not header:
                headers.append("step")
                fp.write(",".join(headers) + "\n")
                header = True
            fp.write(",".join(values) + "\n")
        fp.flush()

    return backward_hook


def get_tensor_hook(module, name, trainer, rank, logger, dump_to_file=False):
    """
    A tensor hook to dump all of the tensor weight norms and grad norms at the end of each of the backward steps.

    For more details about the tensor hook, check:
    https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html

    Parameters
    ----------
    module : torch.nn.Module
        module to register the hook
    name : str
        tensor name
    trainer : PTL trainer
        PTL trainer
    rank : int
        worker rank
    logger : PTL log function
        PTL log function
    dump_to_file : bool, optional
        wether dump the csv file to the disk, by default False

    Returns
    -------
    tensor_hook
    """
    if dump_to_file:
        os.makedirs("debug_info", exist_ok=True)
        fp = open(f"debug_info/tensor_{name}_rank{rank}.csv", "w")
        header = False

    def tensor_hook(grad):
        """Tensor hook to dump all the tensor weight norms and grad norms at the end of each of the backward steps."""
        nonlocal header
        nonlocal fp
        values = []
        headers = []

        weight = module.get_parameter(name)
        weight_norm = weight.data.norm()
        grad_norm = grad.data.norm()
        logger(f"debug_info_tensors/{name}_rank{rank}_grad_norm", grad_norm)
        logger(f"debug_info_tensors/{name}_rank{rank}_weight_norm", weight_norm)
        values.append(f"{weight_norm}")
        values.append(f"{grad_norm}")
        values.append(f"{trainer.global_step}")
        if dump_to_file:
            if not header:
                headers.append("weight")
                headers.append("grad")
                headers.append("step")
                fp.write(",".join(headers) + "\n")
                header = True
            fp.write(",".join(values) + "\n")
            fp.flush()
        return grad

    return tensor_hook


def register_debug_hooks(module, trainer, logger, dump_to_file=False):
    """
    Register debug hooks. It can
    1. track the module forward step input/output norm
    2. track the module backward step input/output grad norm
    3. track the parameter weight norm and grad norm.
    """
    # default rank 0
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    for name, tensor in module.named_parameters():
        if name != "":
            tensor.register_hook(get_tensor_hook(module, name, trainer, rank, logger, dump_to_file))
    for name, layer in module.named_modules():
        if name != "":
            layer.register_forward_hook(get_forward_hook(name, trainer, rank, logger, dump_to_file))
            layer.register_full_backward_hook(get_backward_hook(name, trainer, rank, logger, dump_to_file))
