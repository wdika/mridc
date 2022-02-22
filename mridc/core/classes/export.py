# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/exportable.py

import os
from abc import ABC
from collections import defaultdict
from enum import Enum

import onnx
import torch

__all__ = ["ExportFormat", "Exportable"]

from mridc.core.classes.common import typecheck
from mridc.core.neural_types.axes import AxisKind
from mridc.core.neural_types.neural_type import NeuralType
from mridc.utils import logging
from mridc.utils.export_utils import replace_for_export


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {".pt": ExportFormat.TORCHSCRIPT, ".ts": ExportFormat.TORCHSCRIPT, ".onnx": ExportFormat.ONNX}


def get_input_names(self):
    """Returns a list of input names for the Neural Module"""
    if not hasattr(self, "input_types"):
        raise NotImplementedError("For export to work you must define input_types")
    return list(self.input_types.keys())


def get_output_names(self):
    """Returns a list of output names for the Neural Module"""
    if not hasattr(self, "output_types"):
        raise NotImplementedError("For export to work you must define output_types")
    return list(self.output_types.keys())


def get_input_dynamic_axes(self, input_names):
    """Returns a dictionary of dynamic axes for the Neural Module"""
    dynamic_axes = defaultdict(list)
    for name in input_names:
        if name in self.input_types:
            dynamic_axes = {**dynamic_axes, **Exportable._extract_dynamic_axes(name, self.input_types[name])}
    return dynamic_axes


def get_output_dynamic_axes(self, output_names):
    """Returns a dictionary of dynamic axes for the Neural Module"""
    dynamic_axes = defaultdict(list)
    for name in output_names:
        if name in self.output_types:
            dynamic_axes = {**dynamic_axes, **Exportable._extract_dynamic_axes(name, self.output_types[name])}
    return dynamic_axes


def to_onnxrt_input(input_names, input_list, input_dict):
    """Converts a list of inputs to a list of ONNX inputs"""
    odict = {k: v.cpu().numpy() for k, v in input_dict.items()}
    for i, input in enumerate(input_list):
        if type(input) in (list, tuple):
            odict[input_names[i]] = tuple(ip.cpu().numpy() for ip in input)
        else:
            odict[input_names[i]] = input.cpu().numpy()
    return odict


def unpack_nested_neural_type(neural_type):
    """Unpacks a nested NeuralType into a list of NeuralTypes"""
    if type(neural_type) in (list, tuple):
        return unpack_nested_neural_type(neural_type[0])
    return neural_type


class Exportable(ABC):
    """It gives these entities ability to be exported for deployment to formats such as ONNX."""

    @staticmethod
    def get_format(filename: str):
        """Returns the format of the file"""
        _, ext = os.path.splitext(filename)
        try:
            return _EXT_DICT[ext]
        except KeyError:
            raise ValueError(f"Export file {filename} extension does not correspond to any export format!")

    @property
    def input_module(self):
        """Returns the input module"""
        return self

    @property
    def output_module(self):
        """Returns the output module"""
        return self

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 13,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):
        """Exports the Neural Module to a file."""
        my_args = locals()
        del my_args["self"]

        qual_name = f"{self.__module__}.{self.__class__.__qualname__}"
        format = self.get_format(output)
        output_descr = f"{qual_name} exported to {format}"

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Set module to eval mode
            self._set_eval(set_eval)

            if input_example is None:
                input_example = self._get_input_example()

            my_args["input_example"] = input_example

            # Run (possibly overridden) prepare method before calling forward()
            self._prepare_for_export(**my_args)

            input_list, input_dict = self._setup_input_example(input_example)

            input_names = self._process_input_names()
            output_names = self._process_output_names()

            output_example = tuple(self.forward(*input_list, **input_dict))

            with torch.jit.optimized_execution(True), torch.no_grad():
                jitted_model = self._try_jit_compile_model(self, try_script)

                if format == ExportFormat.TORCHSCRIPT:
                    self._export_torchscript(
                        jitted_model, output, input_dict, input_list, check_trace, check_tolerance, verbose
                    )

                elif format == ExportFormat.ONNX:
                    self._export_onnx(
                        jitted_model,
                        input_example,
                        output_example,
                        input_names,
                        output_names,
                        use_dynamic_axes,
                        do_constant_folding,
                        dynamic_axes,
                        output,
                        export_params,
                        keep_initializers_as_inputs,
                        onnx_opset_version,
                        verbose,
                    )

                    # Verify the model can be read, and is valid
                    self._verify_onnx_export(
                        output,
                        output_example,
                        input_list,
                        input_dict,
                        input_names,
                        output_names,
                        check_tolerance,
                        check_trace,
                    )
                else:
                    raise ValueError(f"Encountered unknown export format {format}.")
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            self._export_teardown()
        return [output], [output_descr]

    def _verify_onnx_export(
        self, output, output_example, input_list, input_dict, input_names, output_names, check_tolerance, check_trace
    ):
        """Verifies the exported model can be read and is valid."""
        onnx_model = onnx.load(output)
        onnx.checker.check_model(onnx_model, full_check=True)
        test_runtime = check_trace

        if test_runtime:
            logging.info(f"Graph ips: {[x.name for x in onnx_model.graph.input]}")
            logging.info(f"Graph ops: {[x.name for x in onnx_model.graph.output]}")

        if test_runtime:
            self._verify_runtime(
                onnx_model, input_list, input_dict, input_names, output_names, output_example, output, check_tolerance
            )

    @staticmethod
    def _verify_runtime(
        onnx_model, input_list, input_dict, input_names, output_names, output_example, output, check_tolerance
    ):
        """Verifies the exported model can be run on a test input."""
        try:
            import onnxruntime
        except ImportError:
            logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime.\n")
            return

        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_out = sess.run(output_names, to_onnxrt_input(input_names, input_list, input_dict))
        all_good = True

        for i, out in enumerate(ort_out[0]):
            expected = output_example[i]
            if torch.is_tensor(expected) and not torch.allclose(
                torch.from_numpy(out), expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance
            ):
                all_good = False
                logging.info(f"onnxruntime results mismatch! PyTorch(expected):\n{expected}\nONNX runtime:\n{out}")
        status = "SUCCESS" if all_good else "FAIL"
        logging.info(f"ONNX generated at {output} verified with onnxruntime : " + status)

    def _export_onnx(
        self,
        jitted_model,
        input_example,
        output_example,
        input_names,
        output_names,
        use_dynamic_axes,
        do_constant_folding,
        dynamic_axes,
        output,
        export_params,
        keep_initializers_as_inputs,
        onnx_opset_version,
        verbose,
    ):
        """Exports the model to ONNX."""
        if jitted_model is None:
            jitted_model = self

        dynamic_axes = self._get_dynamic_axes(dynamic_axes, input_names, output_names, use_dynamic_axes)

        torch.onnx.export(
            jitted_model,
            input_example,
            output,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
            export_params=export_params,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            dynamic_axes=dynamic_axes,
            opset_version=onnx_opset_version,
            example_outputs=output_example,
        )

    def _get_dynamic_axes(self, dynamic_axes, input_names, output_names, use_dynamic_axes):
        """Dynamic axis is a mapping from input/output_name => list of "dynamic" indices"""
        if dynamic_axes is None and use_dynamic_axes:
            dynamic_axes = get_input_dynamic_axes(self.input_module, input_names)  # type: ignore
            dynamic_axes = {
                **dynamic_axes,
                **get_output_dynamic_axes(self.output_module, output_names),
            }  # type: ignore
        return dynamic_axes

    def _export_torchscript(self, jitted_model, output, input_dict, input_list, check_trace, check_tolerance, verbose):
        """Exports the model to TorchScript."""
        if jitted_model is None:
            jitted_model = torch.jit.trace_module(
                self,  # type: ignore
                {"forward": tuple(input_list) + tuple(input_dict.values())},
                strict=False,
                optimize=True,
                check_trace=check_trace,
                check_tolerance=check_tolerance,
            )
        if verbose:
            logging.info(f"JIT code:\n{jitted_model.code}")
        jitted_model.save(output)
        if not os.path.exists(output):
            raise AssertionError

    @staticmethod
    def _try_jit_compile_model(module, try_script):
        """Attempts to compile the model."""
        jitted_model = None
        if try_script:
            try:
                jitted_model = torch.jit.script(module)
            except Exception as e:
                logging.error(f"jit.script() failed!{e}")
        return jitted_model

    def _set_eval(self, set_eval):
        """Sets the model to eval mode."""
        if set_eval:
            self.freeze()
            self.input_module.freeze()
            self.output_module.freeze()

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

    @staticmethod
    def _extract_dynamic_axes(name: str, ntype: NeuralType):
        """
        Implement this method to provide dynamic axes id for ONNX export. By default, this method will extract BATCH
        and TIME dimension ids from each provided input/output name argument. For example, if module/model accepts
        argument named "input_signal" with type corresponding to [Batch, Time, Dim] shape, then the returned result
        should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes as they can change from call
        to call during inference.
        Args:
            name: Name of input or output parameter
            ntype: Corresponding Neural Type
        Returns:
        """
        dynamic_axes = defaultdict(list)
        if type(ntype) in (list, tuple):
            ntype = unpack_nested_neural_type(ntype)

        if ntype.axes:
            for ind, axis in enumerate(ntype.axes):
                if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                    dynamic_axes[name].append(ind)
        return dynamic_axes

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        replace_1D_2D = kwargs.get("replace_1D_2D", False)
        replace_for_export(self, replace_1D_2D)  # type: ignore

    def _export_teardown(self):
        """Override this method for any teardown code after export."""
        raise NotImplementedError()

    def _wrap_forward_method(self):
        """Wraps the forward method to handle dynamic axes."""
        old_forward_method = None

        if hasattr(type(self), "forward_for_export"):
            forward_method = type(self).forward_for_export
            old_forward_method = type(self).forward
            type(self).forward = forward_method
        else:
            forward_method = None

        return forward_method, old_forward_method

    @staticmethod
    def _setup_input_example(input_example):
        """Sets up input example for export."""
        input_list = list(input_example)
        input_dict = {}
        # process possible kwargs
        if isinstance(input_list[-1], dict):
            input_dict = input_list[-1]
            input_list = input_list[:-1]
        return input_list, input_dict

    def _get_input_example(self):
        """Gets input example for export."""
        return self.input_module.input_example()

    def _process_input_names(self):
        """Processes input names for export."""
        input_names = get_input_names(self.input_module)  # type: ignore
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_input_names:
            if name in input_names:
                input_names.remove(name)
        return input_names

    def _process_output_names(self):
        """Processes output names for export."""
        output_names = get_output_names(self.output_module)  # type: ignore
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_output_names:
            if name in output_names:
                output_names.remove(name)
        return output_names

    @staticmethod
    def _augment_output_filename(output, prepend: str):
        """Augments output filename with prepend string."""
        path, filename = os.path.split(output)
        filename = f"{prepend}-{filename}"
        return os.path.join(path, filename)

    def forward(self, *inputs, **kwargs):
        """Override this method to implement forward pass."""
        raise NotImplementedError
