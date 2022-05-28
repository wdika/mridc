# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/core/test_exp_manager.py

import math
import re
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from mridc.constants import MRIDC_ENV_VARNAME_VERSION
from mridc.core.classes.modelPT import ModelPT
from mridc.utils.exp_manager import (
    CheckpointMisconfigurationError,
    LoggerMisconfigurationError,
    NotFoundError,
    exp_manager,
)


class MyTestOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if self._step == 0:
                    p.data = 0.1 * torch.ones(p.shape)
                elif self._step == 1:
                    p.data = 0.0 * torch.ones(p.shape)
                else:
                    p.data = 0.01 * torch.ones(p.shape)
        self._step += 1
        return loss


class DoNothingOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        return loss


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        pl.seed_everything(1234)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(2)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def forward(self, batch):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))
        return output

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return MyTestOptimizer(self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def validation_epoch_end(self, loss):
        self.log("val_loss", torch.stack(loss).mean())


class DoNothingModel(ExampleModel):
    def configure_optimizers(self):
        return DoNothingOptimizer(self.parameters())


class TestExpManager:
    @pytest.mark.unit
    def test_omegaconf(self):
        """Ensure omegaconf raises an error when an unexcepted argument is passed"""
        with pytest.raises(OmegaConfBaseException):
            exp_manager(
                pl.Trainer(
                    accelerator="cpu",
                    max_epochs=1,
                    devices=1,
                ),
                {"unused": 1},
            )

    @pytest.mark.unit
    def test_mridc_checkpoint_restore_model(self, tmp_path):
        """Ensure that the model is restored correctly when a checkpoint is provided"""
        test_trainer = pl.Trainer(accelerator="cpu", enable_checkpointing=False, logger=False, max_epochs=4)
        model = ExampleModel()
        test_trainer.fit(model)

        test_trainer = pl.Trainer(accelerator="cpu", enable_checkpointing=False, logger=False, max_epochs=5)
        model = DoNothingModel()
        model.l1.weight = torch.nn.Parameter(torch.tensor((0.0, 0.0)).unsqueeze(0))
        model.l1.bias = torch.nn.Parameter(torch.tensor(1.0))
        test_trainer.fit(model)
