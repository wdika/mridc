# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/optimizer_with_master_params.py

from contextlib import contextmanager

import torch

from mridc.utils import logging

try:
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.transformer.parallel_state import get_data_parallel_world_size, get_data_parallel_group
    from apex.transformer.tensor_parallel import copy_tensor_model_parallel_attributes
    import amp_C

    HAVE_APEX = True

except ImportError:

    HAVE_APEX = False


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters. Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf):
    """
    Use multi-tensor-applier to copy values from one list to another. We don't have a blfoat16 implementation so for
    now if the overflow_buf is not provided, we default back to simple loop copy to be compatible with bfloat16.
    """
    if overflow_buf:
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        # FIXME: use multi-tensor applier for bf16
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class GradBucket:
    """Persistent buffer for main gradients that remains allocated between training iterations."""

    def __init__(self, numel, chunk_size_mb):
        if not HAVE_APEX:
            raise ImportError("Apex was not found. Using model parallel models will error out.")

        self.numel = numel
        self.data = torch.zeros(self.numel, dtype=torch.float, device=torch.cuda.current_device(), requires_grad=False)

        self.chunk_size_mb = chunk_size_mb
        if self.chunk_size_mb > 0:
            chunk_size_bytes = chunk_size_mb * 1024 * 1024
            self.chunk_size_numel = chunk_size_bytes // 4
            self.num_chunks = self.numel // self.chunk_size_numel
            self.numel_per_chunk = [self.chunk_size_numel] * self.num_chunks
            if self.numel % self.chunk_size_numel != 0:
                self.num_chunks += 1
                self.numel_per_chunk.append(self.numel % self.chunk_size_numel)

            self.start_index_per_chunk = torch.cumsum(torch.tensor([0] + self.numel_per_chunk[:-1]), dim=0)
            self.current_chunk = 0
            self.computed_numel_per_chunk = [0] * self.num_chunks

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()

    def allreduce_buffer(self):
        """Synchronous buffer data allreduce"""
        self.data.div_(get_data_parallel_world_size())
        torch.distributed.all_reduce(self.data, group=get_data_parallel_group())  # type: ignore

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the 1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        if end_index > self.numel:
            raise AssertionError("requested tensor is out of the buffer range.")
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)

        grad_chunk_info = None
        if self.chunk_size_mb > 0:
            chunk = start_index // self.chunk_size_numel
            chunk_start_index = self.start_index_per_chunk[chunk]
            chunk_end_index = chunk_start_index + self.numel_per_chunk[chunk]
            grad_chunk_info = {chunk: min(chunk_end_index, end_index) - start_index}
            while chunk_end_index < end_index:
                chunk += 1
                chunk_start_index = self.start_index_per_chunk[chunk]
                chunk_end_index = chunk_start_index + self.numel_per_chunk[chunk]
                grad_chunk_info[chunk] = min(chunk_end_index, end_index) - chunk_start_index

        return buffer_tensor, grad_chunk_info

    def update_chunk_info(self, grad_chunk_info):
        """Update the chunk info with the grad_chunk_info."""
        for chunk in grad_chunk_info.keys():
            self.computed_numel_per_chunk[chunk] += grad_chunk_info[chunk]

    def get_allreduce_tensor(self):
        """Get a tensor that can be used for allreduce."""
        if self.computed_numel_per_chunk[self.current_chunk] != self.numel_per_chunk[self.current_chunk]:
            return None

        chunk_start_index = self.start_index_per_chunk[self.current_chunk]
        chunk_end_index = chunk_start_index + self.numel_per_chunk[self.current_chunk]
        self.computed_numel_per_chunk[self.current_chunk] = 0
        self.current_chunk += 1
        if self.current_chunk == self.num_chunks:
            self.current_chunk = 0

        return self.data[chunk_start_index:chunk_end_index]


class MainParamsOptimizerWrapper(torch.optim.Optimizer):
    """
    Float16 optimizer wrapper for half precision (fp16 and bf16) data types.
    This optimizer wrapper holds main parameters and gradients in fp32 to support
    stable convergence.

    Parameters
    ----------
    optimizer: base optimizer such as Adam or SGD.
    fp32_grad_accum: to enable the use of fp32 in gradient accumulation and allreduce.
    contiguous_grad_bucket: to enable allocating the master gradients in the contiguous memory space to reduce memory
    fragmentation.
    async_grad_allreduce: enable asynchronous gradient allreduce that is executed along with the training step back \
    prop.
    """

    def __init__(
        self,
        optimizer,
        fp32_grad_accum=False,
        contiguous_grad_bucket=False,
        async_grad_allreduce=False,
        grad_div_ar_fusion=True,
        grad_allreduce_chunk_size_mb=0,
    ):
        super().__init__(optimizer.param_groups)
        if not HAVE_APEX:
            raise ImportError("Apex was not found. Using model parallel models will error out.")

        self.optimizer = optimizer
        if not self.optimizer:
            raise AssertionError("no optimizer is provided.")
        if contiguous_grad_bucket and not fp32_grad_accum:
            raise AssertionError("contiguous gradient buffer assumes using fp32 grad.")
        if async_grad_allreduce:
            if not fp32_grad_accum:
                raise AssertionError(
                    "async allreduce applies to master gradients only, "
                    "which is supposed to be accumulated after grad op."
                )
            if not contiguous_grad_bucket:
                raise AssertionError(
                    "currently async_grad_allreduce is supported only " "with contiguous_grad_bucket."
                )

        self._fp32_grad_accum = fp32_grad_accum
        self._contiguous_grad_bucket = contiguous_grad_bucket
        self._async_grad_allreduce = async_grad_allreduce
        self._grad_divisor = 1 / get_data_parallel_world_size()

        if self._async_grad_allreduce:
            # use @no_sync to disable backward grad sync during gradient accumulation
            self._require_backward_grad_sync = True
            self._grad_div_ar_fusion = grad_div_ar_fusion
            self._grad_allreduce_chunk_size_mb = grad_allreduce_chunk_size_mb
        else:
            self._require_backward_grad_sync = False
            self._grad_div_ar_fusion = False
            self._grad_allreduce_chunk_size_mb = 0

        # Dummy tensor needed for apex multi-apply tensor.
        self._dummy_overflow_buf = None

        # Create persistent buffers for main gradients in contiguous memory space
        # - Chunked element-wise and allreduce ops without creating a temporary buffer for merged operation
        # - Low memory fragmentation
        self._main_grad_buffers = None
        if self._contiguous_grad_bucket:
            self._main_grad_buffers = {}
            # get the size of buffers
            num_elements = {}
            for i, param_group in enumerate(self.optimizer.param_groups):
                for param in param_group["params"]:
                    if param.requires_grad:
                        num_elements[i] = num_elements.get(i, 0) + param.data.nelement()

                # Allocate gradient memory buffers for each data type
                self._main_grad_buffers[i] = GradBucket(num_elements[i], self._grad_allreduce_chunk_size_mb)

        # Three groups of parameters:
        self.float16_groups = []  # original float16 parameters
        self.fp32_from_float16_groups = []  # fp32 copy of float16 parameters
        self.fp32_from_fp32_groups = []  # original fp32 parameters

        # gradient function hooks
        if self._fp32_grad_accum:
            self.grad_accs = []

        # For all the groups in the original optimizer:
        for i, param_group in enumerate(self.optimizer.param_groups):
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for j, param in enumerate(param_group["params"]):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ["torch.cuda.HalfTensor", "torch.cuda.BFloat16Tensor"]:
                        float16_params_this_group.append(param)

                        # Allocate the main parameter
                        main_param = param.detach().clone().float()

                        # Copy tensor model parallel attributes.
                        copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, "shared"):
                            main_param.shared = param.shared

                        # Assign the grad buffer offset to main parameters
                        if self._contiguous_grad_bucket:
                            num_elements[i] -= param.data.nelement()
                            main_param.grad, grad_chunk_info = self._main_grad_buffers[i].get(
                                param.data.shape, num_elements[i]
                            )
                            param.main_grad = main_param.grad

                        # Replace the optimizer params with the new fp32 copy.
                        param_group["params"][j] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                    elif param.type() == "torch.cuda.FloatTensor":
                        fp32_params_this_group.append(param)
                        param_group["params"][j] = param

                    else:
                        raise TypeError(
                            "Wrapped parameters must be one of torch.cuda.FloatTensor,  torch.cuda.HalfTensor, "
                            f"or torch.cuda.BFloat16Tensor. Received {param.type()}"
                        )

                # Add gradient accumulation hook for fp32 grad accumulation
                if self._fp32_grad_accum:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param, main_param, i, grad_chunk_info))
                    self.grad_accs.append(grad_acc)

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def _make_param_hook(self, param, main_param, i, grad_chunk_info):
        """Create the grad accumulation and all-reduce hook for back prop."""

        def param_hook(*unused):
            """Gradient accumulation and all-reduce."""
            if param.grad.data is None:
                return
            if main_param.grad is None:
                main_param.grad = param.grad.float()
            else:
                main_param.grad.add_(param.grad.data)
            # Deallocate grad memory.
            param.grad = None

            # Asynchronous gradients allreduce across data_parallel ranks
            if self._require_backward_grad_sync:
                if self._grad_allreduce_chunk_size_mb > 0:
                    self._main_grad_buffers[i].update_chunk_info(grad_chunk_info)
                    while True:
                        allreduce_tensor = self._main_grad_buffers[i].get_allreduce_tensor()
                        if allreduce_tensor is None:
                            break
                        if self._grad_div_ar_fusion:
                            torch.distributed.all_reduce(
                                allreduce_tensor,
                                group=get_data_parallel_group(),
                                async_op=True,
                                op=torch.distributed.make_nccl_premul_sum(self._grad_divisor),
                            )
                        else:
                            allreduce_tensor.div_(get_data_parallel_world_size())
                            torch.distributed.all_reduce(
                                allreduce_tensor,
                                group=get_data_parallel_group(),
                                async_op=True,
                            )
                else:
                    if self._grad_div_ar_fusion:
                        torch.distributed.all_reduce(
                            main_param.grad,
                            group=get_data_parallel_group(),
                            async_op=True,
                            op=torch.distributed.make_nccl_premul_sum(self._grad_divisor),
                        )
                    else:
                        main_param.grad.div_(get_data_parallel_world_size())
                        torch.distributed.all_reduce(
                            main_param.grad,
                            group=get_data_parallel_group(),
                            async_op=True,
                        )

        return param_hook

    def zero_grad(self, set_to_none=True):
        """
        We only need to zero the model related parameters, i.e., float16_groups & fp32_from_fp32_groups. We
        additionally zero fp32_from_float16_groups as a memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated at this point.
        """
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        if self._contiguous_grad_bucket:
            for i in self._main_grad_buffers:
                self._main_grad_buffers[i].zero()
        else:
            for group in self.fp32_from_float16_groups:
                _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def copy_model_grads_to_main_grads(self):
        """Copy model grads to main grads."""
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if model_param.grad is not None:
                    main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

    def _get_model_and_main_params_data_float16(self):
        """Get model and main params data in float16."""
        model_data = []
        main_data = []
        half_dtype = None
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if half_dtype is None:
                    half_dtype = model_param.data.dtype
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data, half_dtype

    def _set_overflow_buffer(self, half_dtype):
        """Set overflow buffer."""
        if half_dtype == torch.float16:
            if self._dummy_overflow_buf is None:
                self._dummy_overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
            else:
                self._dummy_overflow_buf.fill_(0)

    def _copy_main_params_to_model_params(self):
        """Copy main params to model params."""
        # Only needed for the float16 params.
        model_data, main_data, half_dtype = self._get_model_and_main_params_data_float16()
        self._set_overflow_buffer(half_dtype)
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf)

    def _copy_model_params_to_main_params(self):
        """Copy model params to main params."""
        # Only needed for the float16 params.
        model_data, main_data, half_dtype = self._get_model_and_main_params_data_float16()
        self._set_overflow_buffer(half_dtype)
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf)

    def reload_model_params(self):
        """Reload model params."""
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self, **kwargs):
        """Step the optimizer."""
        self.optimizer.step(closure=None, **kwargs)
        # Update params from main params.
        with torch.no_grad():
            self._copy_main_params_to_model_params()
        # Successful update.
        return True

    def state_dict(self):
        """Return the state of the optimizer."""
        return {"optimizer": self.optimizer.state_dict(), "fp32_from_fp16_params": self.fp32_from_float16_groups}

    def load_state_dict(self, state_dict):
        """Load the state of the optimizer."""
        # Optimizer.
        optimizer_key = "optimizer"
        if optimizer_key not in state_dict:
            optimizer_key = "optimizer_state_dict"
            logging.info("***WARNING*** loading optimizer from " "an old checkpoint ...")
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Copy data for the main params.
        fp32_from_float16_params_key = "fp32_from_fp16_params"
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = "fp32_from_fp16"
        for current_group, saved_group in zip(self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)

    def allreduce_main_grads(self):
        """All reduce main grads."""
        for i in self._main_grad_buffers:
            self._main_grad_buffers[i].allreduce_buffer()

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronizations across data-parallel ranks."""
        old_require_backward_grad_sync = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = old_require_backward_grad_sync

    @property
    def async_master_grads_allreduce(self):
        """Return whether to use async allreduce for master grads."""
        return self._async_grad_allreduce

    @property
    def fp32_grad_accumulation(self):
        """Return whether to accumulate gradients in fp32."""
        return self._fp32_grad_accum

    def get_parameters(self):
        """Return the parameters of the optimizer."""
        params = []
        for param_group in self.optimizer.param_groups:
            params.extend(iter(param_group["params"]))
        return params

    def _get_state(self):
        """Promote state, so it can be retrieved or set via "optimizer_instance.state."""
        return self.optimizer.state

    def _set_state(self, value):
        """Promote state, so it can be retrieved or set via "optimizer_instance.state."""
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        """
        Promote param_groups, so it can be retrieved or set via "optimizer_instance.param_groups.
        (for example, to adjust the learning rate)
        """
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        """Set param_groups."""
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)
