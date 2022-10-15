# coding=utf-8
# Automatically generated by Pynguin.
import os as module_1
import pathlib as module_2

import mridc.collections.reconstruction.data.mri_data as module_0


def test_case_0():
    try:
        str_0 = "x?nU"
        set_0 = {str_0, str_0}
        str_1 = "flowgroup"
        bool_0 = False
        bool_1 = True
        int_0 = 1658
        tuple_0 = (int_0,)
        fast_m_r_i_combined_slice_dataset_0 = module_0.FastMRICombinedSliceDataset(
            set_0, str_1, bool_0, bool_1, tuple_0
        )
    except BaseException:
        pass


def test_case_1():
    try:
        float_0 = -2973.3277
        str_0 = "singlecoil"
        str_1 = "singlecoil"
        bool_0 = False
        int_0 = 667
        fast_m_r_i_slice_dataset_0 = module_0.FastMRISliceDataset(str_0, str_1, str_1, float_0, bool_0, int_0)
    except BaseException:
        pass


def test_case_2():
    try:
        float_0 = -2973.3277
        str_0 = "singlecoil"
        str_1 = "singlecoil"
        bool_0 = False
        int_0 = 667
        fast_m_r_i_slice_dataset_0 = module_0.FastMRISliceDataset(str_0, str_1, str_1, float_0, bool_0, int_0)
    except BaseException:
        pass


def test_case_3():
    try:
        float_0 = None
        str_0 = "{hVe@A5v-!#<T"
        list_0 = [str_0, float_0]
        str_1 = "ljD0r'e\x0bO$5(5)0"
        str_2 = None
        dict_0 = {str_1: str_0, str_2: float_0, str_0: str_0}
        path_like_0 = module_1.PathLike(*list_0, **dict_0)
    except BaseException:
        pass


def test_case_4():
    try:
        path_like_0 = module_1.PathLike()
    except BaseException:
        pass


def test_case_5():
    try:
        float_0 = 1123.465167
        str_0 = ".Q(:M<u4e_7z"
        fast_m_r_i_slice_dataset_0 = module_0.FastMRISliceDataset(str_0, float_0)
    except BaseException:
        pass


def test_case_6():
    try:
        str_0 = "tCXy_QZ\r,gke]kt"
        path_0 = module_2.Path()
        str_1 = "SLURM_PROCID"
        dict_0 = {str_1: path_0, path_0: str_0}
        bool_0 = True
        fast_m_r_i_combined_slice_dataset_0 = module_0.FastMRICombinedSliceDataset(str_1, dict_0, dict_0, bool_0)
    except BaseException:
        pass


def test_case_7():
    try:
        sequence_0 = None
        tuple_0 = ()
        int_0 = -772
        path_0 = module_2.Path()
        bool_0 = False
        int_1 = 1958
        tuple_1 = (int_1,)
        fast_m_r_i_combined_slice_dataset_0 = module_0.FastMRICombinedSliceDataset(
            tuple_0, sequence_0, int_0, path_0, bool_0, path_0, tuple_1
        )
    except BaseException:
        pass


def test_case_8():
    try:
        float_0 = -646.7015
        str_0 = "iEiLA"
        list_0 = [str_0, float_0, float_0]
        fast_m_r_i_combined_slice_dataset_0 = module_0.FastMRICombinedSliceDataset(str_0, list_0)
    except BaseException:
        pass


def test_case_9():
    try:
        float_0 = -646.7015
        str_0 = "iEiLA"
        list_0 = [str_0, float_0, float_0]
        fast_m_r_i_combined_slice_dataset_0 = module_0.FastMRICombinedSliceDataset(str_0, list_0)
    except BaseException:
        pass


def test_case_10():
    try:
        str_0 = "Target"
        bytes_0 = b"8\xa2\xd9\xed\xed\x9a\x07Y\xf8\xca\xbe \xaap\x00[D\x0f\x03"
        str_1 = module_0.et_query(str_0, bytes_0)
    except BaseException:
        pass


def test_case_11():
    try:
        list_0 = []
        path_0 = module_2.Path(*list_0)
        fast_m_r_i_slice_dataset_0 = module_0.FastMRISliceDataset(path_0)
    except BaseException:
        pass
