# coding=utf-8
# Automatically generated by Pynguin.
import mridc.collections.common.parts.rnn_utils as module_0


def test_case_0():
    bytes_0 = b"\x8c%\x04t\xf7\x94\xd5\xcc\x1eu"
    str_0 = "<roI*\tA\tRh:D"
    int_0 = -3281
    tuple_0 = str_0, int_0
    set_0 = {bytes_0, int_0, str_0}
    var_0 = module_0.rnn_weights_init(set_0)
    assert var_0 is None
    var_1 = module_0.rnn_weights_init(set_0)
    assert var_1 is None
    str_1 = "9+z>f0'KUIOCb>!,0G'"
    str_2 = "F)+Re&9Uk"
    dict_0 = {var_1: set_0}
    float_0 = -591.12
    var_2 = module_0.rnn_weights_init(dict_0, float_0)
    assert var_2 is None
    var_3 = module_0.rnn_weights_init(int_0, dict_0, str_1)
    assert var_3 is None
    dict_1 = {str_1: str_1, str_2: str_1}
    bool_0 = False
    str_3 = 'Y@"=h}C)IF@_%6O1~\x0c:_'
    var_4 = module_0.rnn_weights_init(bool_0, str_3)
    assert var_4 is None
    bool_1 = True
    var_5 = module_0.rnn_weights_init(dict_1, bool_1)
    assert var_5 is None
    tuple_1 = bytes_0, str_0, tuple_0, str_1
    str_4 = "HMv]\rL4#vN\r{K14f"
    var_6 = module_0.rnn_weights_init(tuple_1, str_4)
    assert var_6 is None
    str_5 = "d=}7`||Vp"
    dict_2 = {str_5: str_5}
    str_6 = "\x0ci`_=."
    bytes_1 = b"\xb2\xf6\xe5t\xcd\xdbU"
    var_7 = module_0.rnn_weights_init(bytes_1)
    assert var_7 is None
    tuple_2 = dict_2, str_6
    var_8 = module_0.rnn_weights_init(str_5, tuple_2)
    assert var_8 is None
    str_7 = ""
    var_9 = module_0.rnn_weights_init(str_7)
    assert var_9 is None
    var_10 = module_0.rnn_weights_init(bool_1, dict_1)
    assert var_10 is None


def test_case_1():
    str_0 = "E"
    str_1 = 's"x.\r<;c]_{L\r'
    var_0 = module_0.rnn_weights_init(str_1, str_1)
    assert var_0 is None
    bytes_0 = b"b"
    bool_0 = True
    var_1 = module_0.rnn_weights_init(bytes_0, bool_0)
    assert var_1 is None
    bytes_1 = b"L"
    tuple_0 = (bytes_1,)
    bool_1 = True
    int_0 = 3649
    float_0 = 84.0
    var_2 = module_0.rnn_weights_init(int_0, float_0)
    assert var_2 is None
    int_1 = 15
    var_3 = module_0.rnn_weights_init(bool_1, int_1)
    assert var_3 is None
    str_2 = """
    Extends existing argparse with default reconstruction args.

    Parameters
    ----------
    parent_parser: Custom CLI parser that will be extended.
        ArgumentParser

    Returns
    -------
    Parser extended by Reconstruction arguments.
        ArgumentParser
    """
    var_4 = module_0.rnn_weights_init(str_2, str_2)
    assert var_4 is None
    bool_2 = False
    dict_0 = None
    int_2 = 2947
    str_3 = "I1I'd>a5P(\x0bn|B#|\t"
    list_0 = [bytes_0, bytes_1, str_3]
    int_3 = 992
    var_5 = module_0.rnn_weights_init(list_0, int_3)
    assert var_5 is None
    str_4 = "MultiDomainNet"
    list_1 = []
    var_6 = module_0.rnn_weights_init(int_2, str_4, list_1)
    assert var_6 is None
    var_7 = module_0.rnn_weights_init(bool_2, dict_0)
    assert var_7 is None
    str_5 = 'r"n9Iv'
    var_8 = module_0.rnn_weights_init(str_5, tuple_0)
    assert var_8 is None
    var_9 = module_0.rnn_weights_init(tuple_0)
    assert var_9 is None
    var_10 = module_0.rnn_weights_init(tuple_0)
    assert var_10 is None
    var_11 = module_0.rnn_weights_init(str_0)
    assert var_11 is None


def test_case_2():
    int_0 = -4184
    dict_0 = {}
    var_0 = module_0.rnn_weights_init(int_0, dict_0)
    assert var_0 is None


def test_case_3():
    str_0 = "~#<i\ttjT="
    str_1 = ")9(na8Z#SM*f^(HWZ"
    int_0 = 1024
    var_0 = module_0.rnn_weights_init(str_1, int_0)
    assert var_0 is None
    int_1 = 3031
    var_1 = module_0.rnn_weights_init(str_1, str_1)
    assert var_1 is None
    var_2 = module_0.rnn_weights_init(int_1)
    assert var_2 is None
    dict_0 = {int_1: var_0, int_1: int_1}
    var_3 = module_0.rnn_weights_init(dict_0, dict_0)
    assert var_3 is None
    list_0 = [str_0, str_0]
    complex_0 = None
    bool_0 = True
    list_1 = [dict_0, var_0, var_1, int_0]
    bytes_0 = b"\x92\x1c\x11\x9b\x87"
    var_4 = module_0.rnn_weights_init(bool_0, list_1, bytes_0)
    assert var_4 is None
    str_2 = "~TS'G@6Rx*z99B"
    var_5 = module_0.rnn_weights_init(str_1, str_2)
    assert var_5 is None
    float_0 = -2904.0
    dict_1 = {str_2: var_5, var_5: str_1}
    int_2 = 36
    var_6 = module_0.rnn_weights_init(float_0, dict_1, int_2)
    assert var_6 is None
    var_7 = module_0.rnn_weights_init(list_0, complex_0)
    assert var_7 is None
    str_3 = "T^N;Obs~]yF~ .ksV"
    set_0 = {var_7}
    var_8 = module_0.rnn_weights_init(set_0)
    assert var_8 is None
    var_9 = module_0.rnn_weights_init(str_0, str_3)
    assert var_9 is None
