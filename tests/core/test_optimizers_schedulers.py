# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/blob/main/tests/core/test_optimizers_schedulers.py

import math
import os
import random
import shutil
from abc import ABC

import numpy as np
import omegaconf
import pytest
import pytorch_lightning as pl
import torch
import torch.optim

from mridc.core import optim
from mridc.core.conf import optimizers
from mridc.core.conf.optimizers import NovogradParams, SGDParams
from mridc.core.conf.schedulers import CosineAnnealingParams
from mridc.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS, SquareRootAnnealing
from mridc.core.optim.novograd import Novograd
from mridc.core.optim.optimizers import AVAILABLE_OPTIMIZERS, get_optimizer, parse_optimizer_args, register_optimizer
from mridc.utils import logging


class TempModel(torch.nn.Module):
    """Create a dummy model for testing."""

    def __init__(self):
        super(TempModel, self).__init__()
        self.layer = torch.nn.Linear(5, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.layer(x)
        return x


class OptCounter(torch.optim.SGD):
    """A simple optimizer that counts the number of calls to step()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            group.setdefault("count", 0)

    def step(self, closure=None):
        """Performs a single optimization step."""
        for group in self.param_groups:
            group["count"] += 1
        super().step(closure)


class RandomDataset(torch.utils.data.Dataset):
    """A dataset that returns random tensors."""

    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.randn(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(pl.LightningModule, ABC):
    """A dummy model for testing."""

    def __init__(self, batch_size, dataset_len, drop_last, max_steps):
        super().__init__()
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.drop_last = drop_last
        self.max_steps = max_steps

        self.my_opt = None

    def train_dataloader(self):
        """Return a training data loader."""
        dataset = RandomDataset(self.dataset_len)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=self.drop_last)

    def training_step(self, batch, batch_idx):
        """Set training step."""
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.ones(output.size()).to(output.device))
        return {"loss": output}

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        self.my_opt = OptCounter(self.parameters(), lr=0.02)
        return self.my_opt


class Callback(pl.callbacks.Callback):
    """A dummy callback for testing."""

    @pl.utilities.distributed.rank_zero_only
    def on_train_end(self, trainer, module):
        """On train end, check that the number of steps is correct"""
        count = module.my_opt.param_groups[0]["count"]
        if trainer.global_step != count or trainer.global_step != module.max_steps:
            logging.debug(f"max_epochs: {trainer.max_epochs}")
            logging.debug(f"accumulate_grad_batches: {trainer.accumulate_grad_batches}")
            logging.debug(f"limit_train_batches: {trainer.limit_train_batches}")
            logging.debug(f"num_processes: {trainer.num_processes}")
            logging.debug(f"batch_size: {module.batch_size}")
            logging.debug(f"dataset_len: {module.dataset_len}")
            logging.debug(f"drop_last: {module.drop_last}")
            logging.debug(f"{len(trainer.train_dataloader)}")
            logging.debug(f"{trainer.num_training_batches}")

        self.assert_counts(trainer, module, count)

    @staticmethod
    def assert_counts(trainer, module, count):
        """Assert that the number of steps is correct"""
        if trainer.global_step != count:
            raise AssertionError(f"{trainer.global_step} != {count} != {module.max_steps}")
        if trainer.global_step != module.max_steps:
            raise AssertionError(f"{trainer.global_step} != {count} != {module.max_steps}")


class SchedulerNoOpCallback(Callback):
    """A dummy callback for testing."""

    @staticmethod
    def on_train_batch_end(trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
        """On each training batch end"""
        # pl_module.max_steps is "original" max steps without trainer extra steps.
        if (trainer.global_step + 1) % 3 == 0 and (trainer.global_step + 1) < pl_module.max_steps:
            schedulers = trainer.lr_schedulers

            for scheduler in schedulers:
                # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                # as well as update the optimizer lr in all param groups
                scheduler["scheduler"].last_epoch -= 2
                scheduler["scheduler"].step()

            # Increase the max step count by 1
            trainer.fit_loop.max_steps = trainer.fit_loop.max_steps + 1

    def assert_counts(self, trainer, module, count):
        """This is a no-op callback, so the counts should not change"""
        num_skips = torch.div(module.max_steps, 3, rounding_mode="trunc")
        extra_steps = module.max_steps + num_skips
        if trainer.global_step != count:
            raise AssertionError(f"{trainer.global_step} != {count} != {extra_steps}")
        if trainer.global_step != extra_steps:
            raise AssertionError(f"{trainer.global_step} != {count} != {extra_steps}")


class TestOptimizersSchedulers:
    """Test the optimizers and schedulers."""

    INITIAL_LR = 0.1
    MIN_LR = 1e-3
    MAX_STEPS = 10

    # fused_adam is looking for CUDA and this test is being run on CPU only tests
    @pytest.mark.unit
    def test_get_optimizer(self):
        """Test that the optimizer is correctly created"""
        model = TempModel()

        for opt_name in AVAILABLE_OPTIMIZERS:
            if opt_name == "fused_adam" and not torch.cuda.is_available():
                continue
            opt_cls = get_optimizer(opt_name)
            if opt_name == "adafactor":
                # Adafactor's default mode uses relative_step without any lr.
                opt = opt_cls(model.parameters())
            else:
                opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

            if not isinstance(opt, AVAILABLE_OPTIMIZERS[opt_name]):
                raise AssertionError

    @pytest.mark.unit
    def test_register_optimizer(self):
        """Test that we can register a new optimizer"""

        class TempOpt(torch.optim.SGD):
            """A dummy optimizer"""

        class TempOptParams(optimizers.SGDParams):
            """A dummy optimizer params"""

        register_optimizer("TempOpt", TempOpt, TempOptParams)

        model = TempModel()
        opt_cls = get_optimizer("TempOpt")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        if not isinstance(opt, TempOpt):
            raise AssertionError

    @pytest.mark.unit
    def test_optim_config_parse_bypass(self):
        """Test that the optimizer config is parsed correctly when the optimizer is not registered."""
        basic_optim_config = {"weight_decay": 0.001, "betas": [0.8, 0.5]}
        parsed_params = parse_optimizer_args("novograd", basic_optim_config)
        if parsed_params["weight_decay"] != basic_optim_config["weight_decay"]:
            raise AssertionError
        if parsed_params["betas"][0] != basic_optim_config["betas"][0]:
            raise AssertionError
        if parsed_params["betas"][1] != basic_optim_config["betas"][1]:
            raise AssertionError

        dict_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = parse_optimizer_args("novograd", dict_config)
        if parsed_params["weight_decay"] != dict_config["weight_decay"]:
            raise AssertionError
        if parsed_params["betas"][0] != dict_config["betas"][0]:
            raise AssertionError
        if parsed_params["betas"][1] != dict_config["betas"][1]:
            raise AssertionError

    @pytest.mark.unit
    def test_optim_config_parse_arg_by_target(self):
        """Test that the optimizer config is parsed correctly by target."""
        basic_optim_config = {
            "_target_": "mridc.core.conf.optimizers.NovogradParams",
            "params": {"weight_decay": 0.001, "betas": [0.8, 0.5]},
        }
        basic_optim_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = parse_optimizer_args("novograd", basic_optim_config)
        if parsed_params["weight_decay"] != basic_optim_config["params"]["weight_decay"]:
            raise AssertionError
        if parsed_params["betas"][0] != basic_optim_config["params"]["betas"][0]:
            raise AssertionError
        if parsed_params["betas"][1] != basic_optim_config["params"]["betas"][1]:
            raise AssertionError

        dict_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = parse_optimizer_args("novograd", dict_config)
        if parsed_params["weight_decay"] != dict_config["params"]["weight_decay"]:
            raise AssertionError
        if parsed_params["betas"][0] != dict_config["params"]["betas"][0]:
            raise AssertionError
        if parsed_params["betas"][1] != dict_config["params"]["betas"][1]:
            raise AssertionError

        # Names are ignored when passing class path
        # This will be captured during optimizer instantiation
        output_config = parse_optimizer_args("sgd", dict_config)
        sgd_config = vars(SGDParams())
        novograd_config = vars(NovogradParams())

        if set(output_config.keys()) == set(sgd_config.keys()):
            raise AssertionError
        if set(output_config.keys()) != set(novograd_config):
            raise AssertionError

    @pytest.mark.unit
    def test_get_scheduler(self):
        """Test that get_scheduler returns the correct scheduler class."""
        model = TempModel()
        optimizer = Novograd(model.parameters(), lr=self.INITIAL_LR)

        for sched_name in AVAILABLE_SCHEDULERS:
            sched_cls = optim.lr_scheduler.get_scheduler(sched_name)

            try:
                sched = sched_cls(optimizer)
                if not isinstance(sched, AVAILABLE_SCHEDULERS[sched_name]):
                    raise AssertionError
                continue
            except Exception:
                pass

            try:
                sched = sched_cls(optimizer, max_steps=self.MAX_STEPS)
                if not isinstance(sched, AVAILABLE_SCHEDULERS[sched_name]):
                    raise AssertionError
                continue
            except Exception:
                pass

    @pytest.mark.unit
    def test_register_scheduler(self):
        """Test registering a new scheduler"""

        class TempSched(optim.lr_scheduler.CosineAnnealing):
            """Temporary scheduler class."""

        class TempSchedParams(CosineAnnealingParams):
            """Temporary scheduler class."""

        optim.lr_scheduler.register_scheduler("TempSched", TempSched, TempSchedParams)

        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)
        sched_cls = optim.lr_scheduler.get_scheduler("TempSched")
        sched = sched_cls(opt, max_steps=self.MAX_STEPS)

        if not isinstance(sched, TempSched):
            raise AssertionError

    @pytest.mark.unit
    def test_sched_config_parse_simple(self):
        """Test that scheduler config is parsed correctly"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        basic_sched_config = {"name": "CosineAnnealing", "max_steps": 10}
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, basic_sched_config)
        if not isinstance(scheduler_setup["scheduler"], optim.lr_scheduler.CosineAnnealing):
            raise AssertionError

        dict_config = omegaconf.OmegaConf.create(basic_sched_config)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, dict_config)
        if not isinstance(scheduler_setup["scheduler"], optim.lr_scheduler.CosineAnnealing):
            raise AssertionError

    @pytest.mark.unit
    def test_sched_config_parse_from_cls(self):
        """Test that we can parse a scheduler from a class"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        basic_sched_config = {
            "_target_": "mridc.core.conf.schedulers.CosineAnnealingParams",
            "params": {"min_lr": 0.1},
            "max_steps": self.MAX_STEPS,
        }
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, basic_sched_config)
        if not isinstance(scheduler_setup["scheduler"], optim.lr_scheduler.CosineAnnealing):
            raise AssertionError

        dict_config = omegaconf.OmegaConf.create(basic_sched_config)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, dict_config)
        if not isinstance(scheduler_setup["scheduler"], optim.lr_scheduler.CosineAnnealing):
            raise AssertionError

    @pytest.mark.unit
    def test_WarmupPolicy(self):
        """Test WarmupPolicy"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupPolicy(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupPolicy(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 4:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_WarmupHoldPolicy(self):
        """Test WarmupHoldPolicy"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupHoldPolicy(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr <= self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 4:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr <= self.MIN_LR:
            raise AssertionError

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 4:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr < self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_WarmupAnnealing(self):
        """Test that the warmup annealing policy works as expected."""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr < self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 4:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] != self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr < self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_SquareAnnealing(self):
        """Test SquareAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.SquareAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.SquareAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_SquareRootAnnealing(self):
        """Test SquareRootAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = SquareRootAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.SquareRootAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_CosineAnnealing(self):
        """Test CosineAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.CosineAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.CosineAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup + Constant steps available
        policy = optim.lr_scheduler.CosineAnnealing(
            opt, warmup_steps=3, constant_steps=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 3:
                if policy.get_last_lr()[0] > self.INITIAL_LR + 1e-5:
                    raise AssertionError
            elif 3 < i <= 8:
                if policy.get_last_lr()[0] != policy._get_lr(i)[0]:
                    raise AssertionError
            elif policy.get_last_lr()[0] != self.MIN_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_PolynomialDecayAnnealing(self):
        """Test PolynomialDecayAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.PolynomialDecayAnnealing(
            opt, power=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.PolynomialDecayAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_PolynomialHoldDecayAnnealing(self):
        """Test PolynomialHoldDecayAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, power=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr <= self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, power=2, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr < self.MIN_LR:
            raise AssertionError

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR, power=2
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 4:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif i <= 8:
                if policy.get_last_lr()[0] < self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr < self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_InverseSquareRootAnnealing(self):
        """Test InverseSquareRootAnnealing"""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.InverseSquareRootAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        for _ in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

        # Warmup steps available
        policy = optim.lr_scheduler.InverseSquareRootAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        if initial_lr >= self.INITIAL_LR:
            raise AssertionError

        for i in range(self.MAX_STEPS):
            if i <= 5:
                if policy.get_last_lr()[0] > self.INITIAL_LR:
                    raise AssertionError
            elif policy.get_last_lr()[0] >= self.INITIAL_LR:
                raise AssertionError

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        if final_lr != self.MIN_LR:
            raise AssertionError

    @pytest.mark.unit
    def test_CosineAnnealing_with_noop_steps(self):
        """Test CosineAnnealing with noop steps."""
        model = TempModel()
        opt_cls = get_optimizer("novograd")
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.CosineAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        if initial_lr != self.INITIAL_LR:
            raise AssertionError

        update_steps = 0
        for i in range(self.MAX_STEPS):
            if policy.get_last_lr()[0] > self.INITIAL_LR:
                raise AssertionError
            opt.step()
            policy.step()

            # Perform a No-Op for scheduler every 2 steps
            if i % 2 == 0:
                policy.last_epoch -= 1
            else:
                update_steps += 1

        policy.step()
        update_steps += 1

        if update_steps >= self.MAX_STEPS:
            raise AssertionError

        final_lr = policy.get_last_lr()[0]
        if final_lr <= self.MIN_LR:
            raise AssertionError

        # update step = true number of updates performed after some number of skipped steps
        true_end_lr = policy._get_lr(step=update_steps)[0]
        if final_lr != true_end_lr:
            raise AssertionError

    @pytest.mark.unit
    @pytest.mark.run_only_on("CPU")
    def test_max_step_computation(self):
        """Test max step computation."""

        def train(
            max_epochs, accumulate_grad_batches, limit_train_batches, devices, batch_size, dataset_len, drop_last
        ):
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                strategy="ddp_spawn",
                accelerator="cpu",
                devices=devices,
                accumulate_grad_batches=accumulate_grad_batches,
                limit_train_batches=limit_train_batches,
                enable_checkpointing=False,
                progress_bar_refresh_rate=0,
                weights_summary=None,
            )
            max_steps = optim.lr_scheduler.compute_max_steps(
                max_epochs,
                accumulate_grad_batches,
                limit_train_batches,
                devices,
                dataset_len,
                batch_size,
                drop_last,
            )
            model = ExampleModel(batch_size, dataset_len, drop_last, max_steps)
            trainer.callbacks.append(Callback())
            trainer.fit(model)

        # This test will break once we and lightning upgrade to pytorch 1.7.0 due to a bug fix in pytorch 1.7.0
        train(
            31,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            devices=9,
            batch_size=60,
            dataset_len=1613,
            drop_last=True,
        )
        train(
            5,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            devices=4,
            batch_size=97,
            dataset_len=498,
            drop_last=False,
        )
        train(
            5,
            accumulate_grad_batches=8,
            limit_train_batches=1.0,
            devices=4,
            batch_size=54,
            dataset_len=629,
            drop_last=True,
        )
        train(
            5,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            devices=1,
            batch_size=68,
            dataset_len=488,
            drop_last=False,
        )
        for _ in range(5):
            drop_last = bool(random.randint(0, 1))
            accumulate_grad_batches = random.randint(1, 10)

            limit_train_batches_int = random.randint(1, 10)
            limit_train_batches_float = 1.0
            limit_train_batches = random.choice([limit_train_batches_int, limit_train_batches_float])
            max_epochs = random.randint(4, 20)
            devices = random.randint(1, 5)
            dataset_len = random.randint(20, devices * 500)
            batch_size = random.randint(math.ceil(5.0 / devices), min(dataset_len // devices, 128))
            train(
                max_epochs,
                accumulate_grad_batches,
                limit_train_batches,
                devices,
                batch_size,
                dataset_len,
                drop_last,
            )

    @pytest.mark.unit
    @pytest.mark.run_only_on("CPU")
    def test_max_step_computation_with_sched_no_ops(self):
        """Test that max_step is computed correctly when scheduler has no_ops"""

        def train(
            max_steps, accumulate_grad_batches, limit_train_batches, num_processes, batch_size, dataset_len, drop_last
        ):
            """Set up trainer and model"""
            trainer = pl.Trainer(
                max_steps=max_steps,
                strategy="ddp_spawn",
                accelerator="cpu",
                num_processes=num_processes,
                accumulate_grad_batches=accumulate_grad_batches,
                limit_train_batches=limit_train_batches,
                enable_checkpointing=False,
                progress_bar_refresh_rate=0,
                weights_summary=None,
            )
            model = ExampleModel(batch_size, dataset_len, drop_last, max_steps)
            trainer.callbacks.append(SchedulerNoOpCallback())
            trainer.fit(model)

        # This test will break once we and lightning upgrade to pytorch 1.7.0 due to a bug fix in pytorch 1.7.0
        train(
            max_steps=20,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            num_processes=4,
            batch_size=60,
            dataset_len=2000,
            drop_last=True,
        )

    @staticmethod
    def test_remove_logs_left():
        """Remove logs left by the trainer."""
        if os.path.exists(os.path.join(os.getcwd(), "lightning_logs")):
            shutil.rmtree(os.path.join(os.getcwd(), "lightning_logs"))
