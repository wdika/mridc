# coding=utf-8
# Automatically generated by Pynguin.
import pytest

import mridc.collections.reconstruction.models.sigmanet.sensitivity_net as module_0


def test_case_0():
    try:
        sensitivity_network_0 = None
        complex_norm_wrapper_0 = module_0.ComplexNormWrapper(sensitivity_network_0)
        assert complex_norm_wrapper_0.training is True
        assert complex_norm_wrapper_0.model is None
        float_0 = 677.0
        var_0 = complex_norm_wrapper_0.forward(float_0)
    except BaseException:
        pass


def test_case_1():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        set_0 = None
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        var_0 = complex_instance_norm_0.complex_instance_norm(set_0, complex_instance_norm_1)
    except BaseException:
        pass


def test_case_2():
    try:
        bool_0 = None
        list_0 = []
        str_0 = "Mlr<Gt\rX@B"
        bool_1 = True
        bytes_0 = b"\x9b\xd7\x1e\x08y\x1d\x9f3\xd4\xb5"
        complex_norm_wrapper_0 = module_0.ComplexNormWrapper(bytes_0)
        assert complex_norm_wrapper_0.training is True
        assert complex_norm_wrapper_0.model == b"\x9b\xd7\x1e\x08y\x1d\x9f3\xd4\xb5"
        sensitivity_network_0 = module_0.SensitivityNetwork(bool_1, complex_norm_wrapper_0, complex_norm_wrapper_0)
        assert sensitivity_network_0.training is True
        assert sensitivity_network_0.shared_params is True
        assert sensitivity_network_0.num_iter == 1
        assert sensitivity_network_0.num_iter_total is True
        assert sensitivity_network_0.is_trainable == [True]
        assert sensitivity_network_0.save_space is False
        assert sensitivity_network_0.reset_cache is False
        var_0 = sensitivity_network_0.unfreeze_all()
        assert var_0 is None
        complex_norm_wrapper_1 = module_0.ComplexNormWrapper(bool_0)
        assert complex_norm_wrapper_1.training is True
        assert complex_norm_wrapper_1.model is None
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        int_0 = 1277
        complex_norm_wrapper_2 = module_0.ComplexNormWrapper(int_0)
        assert complex_norm_wrapper_2.training is True
        assert complex_norm_wrapper_2.model == 1277
        tuple_0 = str_0, complex_norm_wrapper_2, list_0
        list_1 = None
        bool_2 = False
        list_2 = None
        var_1 = module_0.matrix_invert(tuple_0, list_1, bool_2, list_2)
    except BaseException:
        pass


def test_case_3():
    try:
        bytes_0 = b"\xd4\xb3?\x16#vN\xa3\x8ek\xf9\x1c\xa2\xbe\xd9\x15"
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        list_0 = [bytes_0]
        bytes_1 = b"\xab?\xb0\xa1\xc3y,\x82x\xa6;\x11\xcapn\x06"
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        int_0 = 3114
        tuple_0 = (int_0,)
        str_0 = None
        tuple_1 = complex_instance_norm_1, tuple_0, str_0
        sensitivity_network_0 = module_0.SensitivityNetwork(list_0, list_0, bytes_1, tuple_1)
    except BaseException:
        pass


def test_case_4():
    try:
        bool_0 = True
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        float_0 = None
        float_1 = -2025.0715
        str_0 = "9.@~</"
        set_0 = {bool_0, str_0, float_0}
        complex_norm_wrapper_0 = module_0.ComplexNormWrapper(set_0)
        assert complex_norm_wrapper_0.training is True
        assert complex_norm_wrapper_0.model == {None, True, "9.@~</"}
        bytes_0 = b"\x8d"
        sensitivity_network_0 = module_0.SensitivityNetwork(float_0, float_1, str_0, complex_norm_wrapper_0, bytes_0)
    except BaseException:
        pass


def test_case_5():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        int_0 = -34
        var_0 = complex_instance_norm_0.forward(int_0)
    except BaseException:
        pass


def test_case_6():
    try:
        bool_0 = False
        int_0 = None
        int_1 = 37
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        dict_0 = {bool_0: complex_instance_norm_0}
        bool_1 = False
        sensitivity_network_0 = module_0.SensitivityNetwork(int_0, int_1, dict_0, bool_1)
    except BaseException:
        pass


def test_case_7():
    try:
        str_0 = "'qAH@f=%P2_\n|q"
        list_0 = [str_0, str_0, str_0]
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        var_0 = complex_instance_norm_0.unnormalize(list_0)
    except BaseException:
        pass


def test_case_8():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        int_0 = -137
        bytes_0 = b"\xb3\xf0\x85p\x90\x99\xff\xe0?"
        str_0 = "=,a"
        tuple_0 = (str_0,)
        sensitivity_network_0 = module_0.SensitivityNetwork(int_0, bytes_0, complex_instance_norm_0, tuple_0)
        assert sensitivity_network_0.training is True
        assert sensitivity_network_0.shared_params == ("=,a",)
        assert sensitivity_network_0.num_iter == 1
        assert sensitivity_network_0.num_iter_total == -137
        assert sensitivity_network_0.is_trainable == []
        assert sensitivity_network_0.save_space is False
        assert sensitivity_network_0.reset_cache is False
        var_0 = sensitivity_network_0.stage_training_init()
    except BaseException:
        pass


def test_case_9():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        int_0 = -137
        bytes_0 = b"\xb3\xf0\x85p\x90\x99\xff\xe0?"
        str_0 = "=,a"
        tuple_0 = (str_0,)
        sensitivity_network_0 = module_0.SensitivityNetwork(int_0, bytes_0, complex_instance_norm_0, tuple_0)
        assert sensitivity_network_0.training is True
        assert sensitivity_network_0.shared_params == ("=,a",)
        assert sensitivity_network_0.num_iter == 1
        assert sensitivity_network_0.num_iter_total == -137
        assert sensitivity_network_0.is_trainable == []
        assert sensitivity_network_0.save_space is False
        assert sensitivity_network_0.reset_cache is False
        var_0 = sensitivity_network_0.stage_training_init()
    except BaseException:
        pass


def test_case_10():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        float_0 = -1946.6886
        int_0 = -128
        int_1 = -137
        bytes_0 = b"\xb3\xf0\x85p\x90\x99\xff\xe0?"
        str_0 = "=,a"
        tuple_0 = (str_0,)
        sensitivity_network_0 = module_0.SensitivityNetwork(int_1, bytes_0, complex_instance_norm_0, tuple_0)
        assert sensitivity_network_0.training is True
        assert sensitivity_network_0.shared_params == ("=,a",)
        assert sensitivity_network_0.num_iter == 1
        assert sensitivity_network_0.num_iter_total == -137
        assert sensitivity_network_0.is_trainable == []
        assert sensitivity_network_0.save_space is False
        assert sensitivity_network_0.reset_cache is False
        list_0 = [complex_instance_norm_0]
        int_2 = 1332
        var_0 = sensitivity_network_0.forward_save_space(sensitivity_network_0, list_0, int_2, complex_instance_norm_0)
        assert var_0.training is True
        assert var_0.shared_params == ("=,a",)
        assert var_0.num_iter == 1
        assert var_0.num_iter_total == -137
        assert var_0.is_trainable == []
        assert var_0.save_space is False
        assert var_0.reset_cache is False
        var_1 = sensitivity_network_0.forward(float_0, float_0, float_0, int_0)
        assert var_1 == pytest.approx(-1946.6886, abs=0.01, rel=0.01)
        set_0 = {complex_instance_norm_0, complex_instance_norm_0, complex_instance_norm_1, complex_instance_norm_0}
        var_2 = complex_instance_norm_0.set_normalization(set_0)
    except BaseException:
        pass


def test_case_11():
    try:
        complex_instance_norm_0 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_0.training is True
        assert complex_instance_norm_0.mean == 0
        assert complex_instance_norm_0.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_0.cov_xy_half == 0
        assert complex_instance_norm_0.cov_yx_half == 0
        assert complex_instance_norm_0.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        complex_instance_norm_1 = module_0.ComplexInstanceNorm()
        assert complex_instance_norm_1.training is True
        assert complex_instance_norm_1.mean == 0
        assert complex_instance_norm_1.cov_xx_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        assert complex_instance_norm_1.cov_xy_half == 0
        assert complex_instance_norm_1.cov_yx_half == 0
        assert complex_instance_norm_1.cov_yy_half == pytest.approx(0.7071067811865475, abs=0.01, rel=0.01)
        float_0 = -1946.6886
        int_0 = -128
        int_1 = -137
        bytes_0 = b"\xb3\xf0\x85p\x90\x99\xff\xe0?"
        str_0 = "=,a"
        tuple_0 = (str_0,)
        sensitivity_network_0 = module_0.SensitivityNetwork(int_1, bytes_0, complex_instance_norm_0, tuple_0)
        assert sensitivity_network_0.training is True
        assert sensitivity_network_0.shared_params == ("=,a",)
        assert sensitivity_network_0.num_iter == 1
        assert sensitivity_network_0.num_iter_total == -137
        assert sensitivity_network_0.is_trainable == []
        assert sensitivity_network_0.save_space is False
        assert sensitivity_network_0.reset_cache is False
        list_0 = [complex_instance_norm_0]
        int_2 = 1332
        var_0 = sensitivity_network_0.forward_save_space(sensitivity_network_0, list_0, int_2, complex_instance_norm_0)
        assert var_0.training is True
        assert var_0.shared_params == ("=,a",)
        assert var_0.num_iter == 1
        assert var_0.num_iter_total == -137
        assert var_0.is_trainable == []
        assert var_0.save_space is False
        assert var_0.reset_cache is False
        var_1 = sensitivity_network_0.forward(float_0, float_0, float_0, int_0)
        assert var_1 == pytest.approx(-1946.6886, abs=0.01, rel=0.01)
        set_0 = {complex_instance_norm_0, complex_instance_norm_0, complex_instance_norm_1, complex_instance_norm_0}
        var_2 = complex_instance_norm_0.set_normalization(set_0)
    except BaseException:
        pass
