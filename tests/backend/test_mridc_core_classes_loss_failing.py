# coding=utf-8
# Automatically generated by Pynguin.
import mridc.core.classes.loss as module_0


def test_case_0():
    try:
        str_0 = ")k(JmhPX#DA7l\nI%B5M"
        loss_0 = module_0.Loss(str_0)
        assert loss_0.training is True
        assert loss_0.reduction == "mean"
        loss_1 = module_0.Loss()
        assert loss_1.training is True
        assert loss_1.reduction == "mean"
        set_0 = set()
        float_0 = 4598.59851
        loss_2 = module_0.Loss()
        assert loss_2.training is True
        assert loss_2.reduction == "mean"
        str_1 = "S,pO'`rW"
        loss_3 = module_0.Loss(set_0, str_1)
        assert loss_3.training is True
        assert loss_3.reduction == "sum"
        loss_4 = module_0.Loss()
        assert loss_4.training is True
        assert loss_4.reduction == "mean"
        loss_5 = module_0.Loss(loss_1)
        assert loss_5.training is True
        assert loss_5.reduction == "mean"
        loss_6 = module_0.Loss(float_0)
        assert loss_6.training is True
        assert loss_6.reduction == "mean"
        loss_7 = module_0.Loss()
        assert loss_7.training is True
        assert loss_7.reduction == "mean"
        str_2 = "('&>XN'VFgG!C"
        loss_8 = module_0.Loss()
        assert loss_8.training is True
        assert loss_8.reduction == "mean"
        str_3 = "HrR\x0b(&g.&Z\x0b+"
        loss_9 = module_0.Loss(float_0, str_3)
        assert loss_9.training is True
        assert loss_9.reduction == "mean"
        loss_10 = module_0.Loss()
        assert loss_10.training is True
        assert loss_10.reduction == "mean"
        loss_11 = module_0.Loss(str_2)
        assert loss_11.training is True
        assert loss_11.reduction == "mean"
        loss_12 = module_0.Loss()
        assert loss_12.training is True
        assert loss_12.reduction == "mean"
        loss_13 = module_0.Loss(set_0, str_2)
        assert loss_13.training is True
        assert loss_13.reduction == "sum"
        int_0 = 1007
        loss_14 = module_0.Loss(int_0)
        assert loss_14.training is True
        assert loss_14.reduction == "mean"
        int_1 = 2174
        bytes_0 = b"\x88b\xd5>\xc6\x90\xd5]\x1d\x86\xa3{7"
        loss_15 = module_0.Loss()
        assert loss_15.training is True
        assert loss_15.reduction == "mean"
        float_1 = -19.925
        loss_16 = module_0.Loss(float_1)
        assert loss_16.training is True
        assert loss_16.reduction == "mean"
        loss_17 = module_0.Loss(bytes_0)
        assert loss_17.training is True
        assert loss_17.reduction == "mean"
        loss_18 = module_0.Loss(int_1)
        assert loss_18.training is True
        assert loss_18.reduction == "mean"
    except BaseException:
        pass
