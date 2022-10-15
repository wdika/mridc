# coding=utf-8
# Automatically generated by Pynguin.
import pytest

import mridc.collections.reconstruction.models.rim.conv_layers as module_0


def test_case_0():
    try:
        bytes_0 = None
        str_0 = "rCJ"
        float_0 = 3.0
        set_0 = {bytes_0}
        bool_0 = True
        conv_nonlinear_0 = None
        int_0 = 107
        conv_nonlinear_1 = module_0.ConvNonlinear(
            str_0, conv_nonlinear_0, float_0, bool_0, set_0, int_0)
    except BaseException:
        pass


def test_case_1():
    try:
        bytes_0 = b"=<"
        set_0 = {bytes_0, bytes_0, bytes_0}
        float_0 = 10.02
        int_0 = None
        str_0 = "5F"
        conv_r_n_n_stack_0 = module_0.ConvRNNStack(int_0, str_0)
        assert conv_r_n_n_stack_0.training is True
        assert conv_r_n_n_stack_0.convs is None
        assert conv_r_n_n_stack_0.rnn == "5F"
        str_1 = None
        conv_r_n_n_stack_1 = module_0.ConvRNNStack(float_0, set_0)
        assert conv_r_n_n_stack_1.training is True
        assert conv_r_n_n_stack_1.convs == pytest.approx(
            10.02, abs=0.01, rel=0.01)
        assert conv_r_n_n_stack_1.rnn == {b"=<"}
        conv_r_n_n_stack_2 = module_0.ConvRNNStack(str_1, conv_r_n_n_stack_1)
        assert conv_r_n_n_stack_2.training is True
        assert conv_r_n_n_stack_2.convs is None
        dict_0 = {}
        bool_0 = True
        conv_nonlinear_0 = module_0.ConvNonlinear(
            conv_r_n_n_stack_2, conv_r_n_n_stack_1, bytes_0, dict_0, bool_0, conv_r_n_n_stack_1
        )
    except BaseException:
        pass


def test_case_2():
    try:
        bool_0 = False
        bool_1 = True
        set_0 = set()
        list_0 = [set_0, set_0, bool_1]
        conv_nonlinear_0 = None
        conv_r_n_n_stack_0 = module_0.ConvRNNStack(list_0, conv_nonlinear_0)
        assert conv_r_n_n_stack_0.training is True
        assert conv_r_n_n_stack_0.convs == [
            {1}.__class__(), {1}.__class__(), True]
        assert conv_r_n_n_stack_0.rnn is None
        list_1 = [conv_r_n_n_stack_0, list_0, conv_r_n_n_stack_0, bool_0]
        str_0 = "v3of'g9\\S9:aD)-R$7_"
        bool_2 = True
        list_2 = [bool_2, conv_r_n_n_stack_0, bool_0, set_0]
        str_1 = "'2rv"
        str_2 = "NK>Kd=8sN?v3A'p"
        conv_nonlinear_1 = module_0.ConvNonlinear(
            list_1, str_0, bool_2, list_2, set_0, str_1, str_2)
    except BaseException:
        pass


def test_case_3():
    try:
        bool_0 = False
        bool_1 = True
        set_0 = set()
        list_0 = [set_0, set_0, bool_1]
        conv_nonlinear_0 = None
        conv_r_n_n_stack_0 = module_0.ConvRNNStack(list_0, conv_nonlinear_0)
        assert conv_r_n_n_stack_0.training is True
        assert conv_r_n_n_stack_0.convs == [
            {1}.__class__(), {1}.__class__(), True]
        assert conv_r_n_n_stack_0.rnn is None
        list_1 = [conv_nonlinear_0, conv_r_n_n_stack_0,
                  list_0, set_0, conv_r_n_n_stack_0, bool_0]
        str_0 = "v3of'g9\\S9:aD)-R$7_"
        bool_2 = True
        list_2 = [bool_2, conv_r_n_n_stack_0, bool_0, set_0]
        str_1 = "'2rv"
        str_2 = "LossType"
        conv_nonlinear_1 = module_0.ConvNonlinear(
            list_1, str_0, bool_2, list_2, set_0, str_1, str_2)
    except BaseException:
        pass


def test_case_5():
    try:
        int_0 = 49
        list_0 = [int_0, int_0]
        str_0 = "06#,PVOad>F\\"
        str_1 = "NDM}3oK6JnF&(,}."
        bytes_0 = None
        set_0 = {str_1}
        conv_r_n_n_stack_0 = module_0.ConvRNNStack(set_0, str_0)
        assert conv_r_n_n_stack_0.training is True
        assert conv_r_n_n_stack_0.convs == {"NDM}3oK6JnF&(,}."}
        assert conv_r_n_n_stack_0.rnn == "06#,PVOad>F\\"
        conv_nonlinear_0 = module_0.ConvNonlinear(
            str_0, str_1, list_0, bytes_0, list_0, conv_r_n_n_stack_0, set_0)
    except BaseException:
        pass


def test_case_6():
    try:
        str_0 = "A1?AE8}mJfp\rbw"
        str_1 = "G_[H"
        bytes_0 = b"\xa984"
        bool_0 = True
        tuple_0 = bytes_0, bytes_0, bool_0
        conv_r_n_n_stack_0 = module_0.ConvRNNStack(str_1, tuple_0)
        assert conv_r_n_n_stack_0.training is True
        assert conv_r_n_n_stack_0.convs == "G_[H"
        assert conv_r_n_n_stack_0.rnn == (b"\xa984", b"\xa984", True)
        var_0 = conv_r_n_n_stack_0.forward(str_0, str_0)
    except BaseException:
        pass


def test_case_7():
    try:
        str_0 = "mx\x0c%%JA"
        str_1 = "e@O-+Gd^Y\x0c.$<\n"
        set_0 = {str_0, str_1, str_0}
        bool_0 = True
        bytes_0 = b"\xc2G4!\xc8\x18X\x82\xfan\x8d?\x83\xe38\x173"
        tuple_0 = bool_0, bytes_0
        dict_0 = {str_1: set_0, str_0: set_0, str_0: str_0, tuple_0: str_0}
        float_0 = -1199.0
        str_2 = "\x0cJY##nWxlb0F"
        int_0 = 1918
        conv_nonlinear_0 = module_0.ConvNonlinear(
            set_0, dict_0, dict_0, float_0, str_2, int_0)
    except BaseException:
        pass
