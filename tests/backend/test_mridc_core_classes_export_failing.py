# coding=utf-8
# Automatically generated by Pynguin.
import mridc.core.classes.export as module_0


def test_case_0():
    try:
        exportable_0 = module_0.Exportable()
        str_0 = "numba.cuda.cudadrv.driver"
        str_1 = None
        var_0 = exportable_0.list_export_subnets()
        assert var_0 == ["self"]
        dict_0 = {str_0: str_0, str_1: str_1, str_0: str_0}
        exportable_1 = module_0.Exportable(**dict_0)
    except BaseException:
        pass


def test_case_1():
    try:
        str_0 = ""
        exportable_0 = module_0.Exportable()
        int_0 = 3228
        list_0 = []
        exportable_1 = module_0.Exportable(*list_0)
        var_0 = exportable_1.get_export_subnet()
        assert exportable_0 is not None
        assert exportable_1 is not None
        assert var_0 is not None
        assert module_0.logging.once_logged == {1}.__class__()
        assert module_0.logging.rank == 0
        bool_0 = True
        exportable_2 = module_0.Exportable()
        assert exportable_2 is not None
        var_1 = exportable_2.get_export_subnet()
        assert var_1 is not None
        var_2 = exportable_2.export(str_0, int_0, bool_0)
    except BaseException:
        pass


def test_case_2():
    try:
        exportable_0 = module_0.Exportable()
        str_0 = "vO`^la `MfU]]R{XU-Sq"
        float_0 = 3394.6123
        int_0 = -1855
        var_0 = exportable_0.export(str_0, float_0, int_0)
    except BaseException:
        pass


def test_case_3():
    try:
        int_0 = -1104
        list_0 = [int_0, int_0, int_0, int_0]
        str_0 = "w[A~s:XCU"
        exportable_0 = module_0.Exportable()
        var_0 = exportable_0.list_export_subnets()
        assert var_0 == ["self"]
        dict_0 = {str_0: list_0, str_0: int_0}
        exportable_1 = module_0.Exportable()
        list_1 = [dict_0]
        var_1 = exportable_0.get_export_subnet(list_1)
    except BaseException:
        pass


def test_case_5():
    try:
        str_0 = "Removing ignored arguments - "
        int_0 = 8
        bool_0 = True
        list_0 = [int_0, bool_0]
        bytes_0 = b"B\xd79o\x8e\x93k;\xe0\x1cON["
        list_1 = [int_0]
        exportable_0 = module_0.Exportable()
        var_0 = exportable_0.list_export_subnets()
        assert var_0 == ["self"]
        str_1 = "RoR>{_*x4e&["
        set_0 = set()
        list_2 = [str_0, bytes_0, list_0, set_0]
        dict_0 = {str_1: str_1, str_1: bytes_0, str_1: bytes_0, str_0: list_2}
        exportable_1 = module_0.Exportable(*list_1, **dict_0)
    except BaseException:
        pass


def test_case_6():
    try:
        str_0 = "val_"
        str_1 = "Vj/Hi\t>bv6}gE~U"
        tuple_0 = None
        exportable_0 = module_0.Exportable()
        tuple_1 = ()
        tuple_2 = tuple_0, exportable_0, tuple_1
        bool_0 = False
        list_0 = [str_1, exportable_0, bool_0, tuple_1]
        float_0 = -2934.59
        exportable_1 = module_0.Exportable()
        var_0 = exportable_1.export(str_0, tuple_2, list_0, float_0)
    except BaseException:
        pass


def test_case_7():
    try:
        list_0 = None
        exportable_0 = module_0.Exportable(*list_0)
    except BaseException:
        pass
