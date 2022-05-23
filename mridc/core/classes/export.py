# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/exportable.py

import os
from abc import ABC

import torch
from torch.onnx import TrainingMode

from mridc.core.classes.common import typecheck
from mridc.core.utils.neural_type_utils import get_dynamic_axes, get_io_names
from mridc.utils import logging
from mridc.utils.export_utils import (
    ExportFormat,
    get_export_format,
    parse_input_example,
    replace_for_export,
    verify_runtime,
    wrap_forward_method,
)

__all__ = ["ExportFormat", "Exportable"]


class Exportable(ABC):
    """
    This Interface should be implemented by particular classes derived from mridc.core.NeuralModule or
    mridc.core.ModelPT. It gives these entities ability to be exported for deployment to formats such as ONNX.
    """

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        onnx_opset_version=None,
        try_script: bool = False,
        training=TrainingMode.EVAL,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):
        """
        Export the module to a file.

        Parameters
        ----------
        output: The output file path.
        input_example: A dictionary of input names and values.
        verbose: If True, print out the export process.
        export_params: If True, export the parameters of the module.
        do_constant_folding: If True, do constant folding.
        onnx_opset_version: The ONNX opset version to use.
        try_script: If True, try to export as TorchScript.
        training: Training mode for the export.
        check_trace: If True, check the trace of the exported model.
        use_dynamic_axes: If True, use dynamic axes for the export.
        dynamic_axes: A dictionary of input names and dynamic axes.
        check_tolerance: The tolerance for the check_trace.
        """
        my_args = locals().copy()
        my_args.pop("self")

        exportables = []
        for m in self.modules():  # type: ignore
            if isinstance(m, Exportable):
                exportables.append(m)

        qual_name = self.__module__ + "." + self.__class__.__qualname__
        format = get_export_format(output)
        output_descr = f"{qual_name} exported to {format}"

        # Pytorch's default for None is too low, can't pass None through
        if onnx_opset_version is None:
            onnx_opset_version = 13

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Allow user to completely override forward method to export
            forward_method, old_forward_method = wrap_forward_method(self)

            # Set module mode
            with torch.onnx.select_model_mode_for_export(
                self, training
            ), torch.inference_mode(), torch.jit.optimized_execution(True):

                if input_example is None:
                    input_example = self.input_module.input_example()

                # Remove i/o examples from args we propagate to enclosed Exportables
                my_args.pop("output")
                my_args.pop("input_example")

                # Run (possibly overridden) prepare methods before calling forward()
                for ex in exportables:
                    ex._prepare_for_export(**my_args, noreplace=True)
                self._prepare_for_export(output=output, input_example=input_example, **my_args)

                input_list, input_dict = parse_input_example(input_example)
                input_names = self.input_names
                output_names = self.output_names
                output_example = tuple(self.forward(*input_list, **input_dict))  # type: ignore

                jitted_model = None
                if try_script:
                    try:
                        jitted_model = torch.jit.script(self)
                    except Exception as e:
                        logging.error(f"jit.script() failed!\n{e}")

                if format == ExportFormat.TORCHSCRIPT:
                    if jitted_model is None:
                        jitted_model = torch.jit.trace_module(
                            self,
                            {"forward": tuple(input_list) + tuple(input_dict.values())},
                            strict=True,
                            check_trace=check_trace,
                            check_tolerance=check_tolerance,
                        )
                    if not self.training:  # type: ignore
                        jitted_model = torch.jit.optimize_for_inference(jitted_model)
                    if verbose:
                        logging.info(f"JIT code:\n{jitted_model.code}")
                    jitted_model.save(output)
                elif format == ExportFormat.ONNX:
                    if jitted_model is None:
                        jitted_model = self

                    # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
                    if dynamic_axes is None and use_dynamic_axes:
                        dynamic_axes = get_dynamic_axes(self.input_module.input_types, input_names)
                        dynamic_axes.update(get_dynamic_axes(self.output_module.output_types, output_names))

                    torch.onnx.export(
                        jitted_model,
                        input_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        export_params=export_params,
                        do_constant_folding=do_constant_folding,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                    )

                    if check_trace:
                        verify_runtime(output, input_list, input_dict, input_names, output_names, output_example)

                else:
                    raise ValueError(f"Encountered unknown export format {format}.")
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            if forward_method:
                type(self).forward = old_forward_method  # type: ignore
            self._export_teardown()
        return [output], [output_descr]

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set()

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set()

    @property
    def supported_export_formats(self):
        """Implement this method to return a set of export formats supported. Default is all types."""
        return {ExportFormat.ONNX, ExportFormat.TORCHSCRIPT}

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        if "noreplace" not in kwargs:
            replace_for_export(self)

    def _export_teardown(self):
        """
        Override this method for any teardown code after export.
        """

    @property
    def input_names(self):
        """Implement this method to return a list of input names"""
        return get_io_names(self.input_module.input_types, self.disabled_deployment_input_names)

    @property
    def output_names(self):
        """Override this method to return a set of output names disabled for export"""
        return get_io_names(self.output_module.output_types, self.disabled_deployment_output_names)
