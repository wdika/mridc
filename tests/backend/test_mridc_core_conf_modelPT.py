# coding=utf-8
# Automatically generated by Pynguin.
import pytest

import mridc.core.classes.dataset as module_1
import mridc.core.conf.modelPT as module_0
import mridc.core.conf.optimizers as module_2


def test_case_0():
    none_type_0 = None
    float_0 = 0.99
    sched_config_0 = module_0.SchedConfig(float_0)
    assert sched_config_0.name == pytest.approx(0.99, abs=0.01, rel=0.01)
    assert sched_config_0.min_lr == pytest.approx(0.0, abs=0.01, rel=0.01)
    assert sched_config_0.last_epoch == -1
    assert module_0.SchedConfig.name == "???"
    assert module_0.SchedConfig.min_lr == pytest.approx(0.0, abs=0.01, rel=0.01)
    assert module_0.SchedConfig.last_epoch == -1
    optim_config_0 = module_0.OptimConfig()
    assert optim_config_0.name == "???"
    assert optim_config_0.sched is None
    assert module_0.OptimConfig.name == "???"
    assert module_0.OptimConfig.sched is None
    model_config_0 = module_0.ModelConfig(optim_config_0)
    assert model_config_0.validation_ds is None
    assert model_config_0.test_ds is None
    assert model_config_0.optim is None
    assert module_0.ModelConfig.train_ds is None
    assert module_0.ModelConfig.validation_ds is None
    assert module_0.ModelConfig.test_ds is None
    assert module_0.ModelConfig.optim is None
    model_config_builder_0 = module_0.ModelConfigBuilder(model_config_0)
    assert model_config_builder_0.train_ds_cfg is None
    assert model_config_builder_0.validation_ds_cfg is None
    assert model_config_builder_0.test_ds_cfg is None
    assert model_config_builder_0.optim_cfg is None
    var_0 = model_config_builder_0.set_test_ds()
    model_config_1 = module_0.ModelConfig()
    assert model_config_1.train_ds is None
    assert model_config_1.validation_ds is None
    assert model_config_1.test_ds is None
    assert model_config_1.optim is None
    dict_0 = {}
    hydra_config_0 = module_0.HydraConfig(dict_0)
    assert hydra_config_0.run == {}
    assert hydra_config_0.job_logging == {"root": {"handlers": None}}
    model_config_builder_1 = module_0.ModelConfigBuilder(model_config_1)
    assert model_config_builder_1.train_ds_cfg is None
    assert model_config_builder_1.validation_ds_cfg is None
    assert model_config_builder_1.test_ds_cfg is None
    assert model_config_builder_1.optim_cfg is None
    var_1 = model_config_builder_1.set_test_ds()
    model_config_builder_2 = module_0.ModelConfigBuilder(model_config_1)
    assert model_config_builder_2.train_ds_cfg is None
    assert model_config_builder_2.validation_ds_cfg is None
    assert model_config_builder_2.test_ds_cfg is None
    assert model_config_builder_2.optim_cfg is None
    model_config_builder_3 = module_0.ModelConfigBuilder(model_config_1)
    assert model_config_builder_3.train_ds_cfg is None
    assert model_config_builder_3.validation_ds_cfg is None
    assert model_config_builder_3.test_ds_cfg is None
    assert model_config_builder_3.optim_cfg is None
    var_2 = model_config_builder_2.set_validation_ds()
    var_3 = model_config_builder_2.set_test_ds(none_type_0)


def test_case_1():
    bool_0 = True
    dataset_config_0 = module_1.DatasetConfig(bool_0)
    model_config_0 = module_0.ModelConfig(dataset_config_0)
    assert model_config_0.validation_ds is None
    assert model_config_0.test_ds is None
    assert model_config_0.optim is None
    assert module_0.ModelConfig.train_ds is None
    assert module_0.ModelConfig.validation_ds is None
    assert module_0.ModelConfig.test_ds is None
    assert module_0.ModelConfig.optim is None
    model_config_builder_0 = module_0.ModelConfigBuilder(model_config_0)
    assert model_config_builder_0.train_ds_cfg is None
    assert model_config_builder_0.validation_ds_cfg is None
    assert model_config_builder_0.test_ds_cfg is None
    assert model_config_builder_0.optim_cfg is None
    var_0 = model_config_builder_0.set_train_ds()
    assert model_config_0.train_ds is None
    int_0 = -3346
    dataset_config_1 = module_1.DatasetConfig(int_0, bool_0, bool_0)
    model_config_1 = module_0.ModelConfig(dataset_config_1)
    assert model_config_1.validation_ds is None
    assert model_config_1.test_ds is None
    assert model_config_1.optim is None
    dataset_config_2 = module_1.DatasetConfig(bool_0)
    optimizer_params_0 = module_2.OptimizerParams()
    model_config_2 = module_0.ModelConfig(optimizer_params_0)
    assert model_config_2.validation_ds is None
    assert model_config_2.test_ds is None
    assert model_config_2.optim is None
    model_config_builder_1 = module_0.ModelConfigBuilder(model_config_2)
    assert model_config_builder_1.train_ds_cfg is None
    assert model_config_builder_1.validation_ds_cfg is None
    assert model_config_builder_1.test_ds_cfg is None
    assert model_config_builder_1.optim_cfg is None
    var_1 = model_config_builder_1.set_validation_ds(dataset_config_2)


def test_case_2():
    optim_config_0 = module_0.OptimConfig()
    assert optim_config_0.name == "???"
    assert optim_config_0.sched is None
    assert module_0.OptimConfig.name == "???"
    assert module_0.OptimConfig.sched is None
    model_config_0 = module_0.ModelConfig(optim_config_0)
    assert model_config_0.validation_ds is None
    assert model_config_0.test_ds is None
    assert model_config_0.optim is None
    assert module_0.ModelConfig.train_ds is None
    assert module_0.ModelConfig.validation_ds is None
    assert module_0.ModelConfig.test_ds is None
    assert module_0.ModelConfig.optim is None
    model_config_builder_0 = module_0.ModelConfigBuilder(model_config_0)
    assert model_config_builder_0.train_ds_cfg is None
    assert model_config_builder_0.validation_ds_cfg is None
    assert model_config_builder_0.test_ds_cfg is None
    assert model_config_builder_0.optim_cfg is None
    var_0 = model_config_builder_0.set_train_ds()
    assert model_config_0.train_ds is None
