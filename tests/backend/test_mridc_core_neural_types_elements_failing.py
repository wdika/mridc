# coding=utf-8
# Automatically generated by Pynguin.
import abc as module_2

import mridc.core.neural_types.comparison as module_1
import mridc.core.neural_types.elements as module_0


def test_case_0():
    try:
        element_type_0 = module_0.ElementType()
        var_0 = element_type_0.__repr__()
        assert var_0 == "ElementType"
        m_r_i_signal_0 = module_0.MRISignal()
        assert element_type_0 is not None
        assert m_r_i_signal_0 is not None
        loss_type_0 = module_0.LossType()
        assert loss_type_0 is not None
        length_0 = module_0.Length()
        assert length_0 is not None
        log_determinant_type_0 = module_0.LogDeterminantType()
        assert log_determinant_type_0 is not None
        list_0 = []
        element_type_1 = module_0.ElementType(*list_0)
        assert element_type_1 is not None
        var_1 = element_type_1.__repr__()
        assert var_1 == "ElementType"
        string_label_0 = module_0.StringLabel()
        assert string_label_0 is not None
        element_type_2 = module_0.ElementType()
        assert element_type_2 is not None
        dict_0 = {}
        index_0 = module_0.Index()
        assert index_0 is not None
        probability_distribution_samples_type_0 = module_0.ProbabilityDistributionSamplesType(
            **dict_0)
        assert probability_distribution_samples_type_0 is not None
        normal_distribution_mean_type_0 = module_0.NormalDistributionMeanType()
        assert normal_distribution_mean_type_0 is not None
        list_1 = []
        probs_type_0 = module_0.ProbsType()
        assert probs_type_0 is not None
        bool_type_0 = module_0.BoolType(*list_1)
        assert bool_type_0 is not None
        labels_type_0 = module_0.LabelsType()
        assert labels_type_0 is not None
        string_type_0 = module_0.StringType()
        assert string_type_0 is not None
        list_2 = []
        bool_type_1 = module_0.BoolType(*list_1)
        assert bool_type_1 is not None
        labels_type_1 = module_0.LabelsType()
        assert labels_type_1 is not None
        mask_type_0 = module_0.MaskType()
        assert mask_type_0 is not None
        length_1 = module_0.Length(*list_2)
        assert length_1 is not None
        neural_type_comparison_result_0 = element_type_2.compare(bool_type_0)
        assert neural_type_comparison_result_0 == module_1.NeuralTypeComparisonResult.GREATER
        assert module_1.NeuralTypeComparisonResult.SAME == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.LESS == module_1.NeuralTypeComparisonResult.LESS
        assert module_1.NeuralTypeComparisonResult.GREATER == module_1.NeuralTypeComparisonResult.GREATER
        assert (
            module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            == module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        )
        assert module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME == module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME
        assert (
            module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
            == module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
        )
        assert module_1.NeuralTypeComparisonResult.INCOMPATIBLE == module_1.NeuralTypeComparisonResult.INCOMPATIBLE
        assert (
            module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            == module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        )
        assert module_1.NeuralTypeComparisonResult.UNCHECKED == module_1.NeuralTypeComparisonResult.UNCHECKED
        string_type_1 = module_0.StringType()
        assert string_type_1 is not None
        mask_type_1 = module_0.MaskType(**dict_0)
        assert mask_type_1 is not None
        bool_type_2 = module_0.BoolType()
        assert bool_type_2 is not None
        list_3 = [mask_type_1, mask_type_1, list_2]
        channel_type_0 = module_0.ChannelType(*list_3, **dict_0)
    except BaseException:
        pass


def test_case_1():
    try:
        a_b_c_meta_0 = module_2.ABCMeta()
    except BaseException:
        pass


def test_case_2():
    try:
        str_0 = None
        float_type_0 = module_0.FloatType()
        list_0 = [str_0, str_0]
        length_0 = module_0.Length(*list_0)
    except BaseException:
        pass


def test_case_3():
    try:
        void_type_0 = module_0.VoidType()
        normalized_image_value_0 = module_0.NormalizedImageValue()
        list_0 = []
        normal_distribution_mean_type_0 = module_0.NormalDistributionMeanType()
        element_type_0 = module_0.ElementType()
        neural_type_comparison_result_0 = element_type_0.compare(
            normal_distribution_mean_type_0)
        assert neural_type_comparison_result_0 == module_1.NeuralTypeComparisonResult.GREATER
        categorical_values_type_0 = module_0.CategoricalValuesType(*list_0)
        str_0 = "`G%4O/W"
        neural_type_comparison_result_1 = element_type_0.compare(str_0)
        assert void_type_0 is not None
        assert normalized_image_value_0 is not None
        assert normal_distribution_mean_type_0 is not None
        assert element_type_0 is not None
        assert categorical_values_type_0 is not None
        assert neural_type_comparison_result_1 == module_1.NeuralTypeComparisonResult.INCOMPATIBLE
        assert module_1.NeuralTypeComparisonResult.SAME == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.LESS == module_1.NeuralTypeComparisonResult.LESS
        assert module_1.NeuralTypeComparisonResult.GREATER == module_1.NeuralTypeComparisonResult.GREATER
        assert (
            module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            == module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        )
        assert module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME == module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME
        assert (
            module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
            == module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
        )
        assert module_1.NeuralTypeComparisonResult.INCOMPATIBLE == module_1.NeuralTypeComparisonResult.INCOMPATIBLE
        assert (
            module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            == module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        )
        assert module_1.NeuralTypeComparisonResult.UNCHECKED == module_1.NeuralTypeComparisonResult.UNCHECKED
        a_b_c_meta_0 = module_2.ABCMeta()
    except BaseException:
        pass


def test_case_4():
    try:
        predictions_type_0 = module_0.PredictionsType()
        str_0 = "_IsyBSMp)H9/X/"
        list_0 = []
        sequence_to_sequence_alignment_type_0 = module_0.SequenceToSequenceAlignmentType(
            *list_0)
        element_type_0 = module_0.ElementType()
        neural_type_comparison_result_0 = element_type_0.compare(
            sequence_to_sequence_alignment_type_0)
        assert neural_type_comparison_result_0 == module_1.NeuralTypeComparisonResult.GREATER
        list_1 = [predictions_type_0, str_0, predictions_type_0]
        recurrents_type_0 = module_0.RecurrentsType(*list_1)
    except BaseException:
        pass


def test_case_5():
    try:
        probability_distribution_samples_type_0 = module_0.ProbabilityDistributionSamplesType()
        list_0 = None
        string_label_0 = module_0.StringLabel()
        labels_type_0 = module_0.LabelsType(*list_0)
    except BaseException:
        pass


def test_case_6():
    try:
        str_0 = None
        str_1 = """
    A base class for a Convolutional Minimal Gated Unit cell.
    # TODO: add paper reference
    """
        str_2 = "'fk"
        dict_0 = {str_1: str_0, str_2: str_2, str_2: str_2}
        bool_type_0 = module_0.BoolType(**dict_0)
    except BaseException:
        pass


def test_case_7():
    try:
        loss_type_0 = module_0.LossType()
        normal_distribution_samples_type_0 = module_0.NormalDistributionSamplesType()
        list_0 = []
        logprobs_type_0 = module_0.LogprobsType(*list_0)
        string_type_0 = module_0.StringType()
        probs_type_0 = module_0.ProbsType()
        predictions_type_0 = module_0.PredictionsType()
        element_type_0 = module_0.ElementType()
        var_0 = element_type_0.__repr__()
        assert var_0 == "ElementType"
        normal_distribution_log_variance_type_0 = module_0.NormalDistributionLogVarianceType()
        list_1 = [normal_distribution_samples_type_0,
                  normal_distribution_samples_type_0, probs_type_0]
        normal_distribution_samples_type_1 = module_0.NormalDistributionSamplesType(
            *list_0)
        element_type_1 = module_0.ElementType()
        var_1 = element_type_1.__str__()
        assert (
            var_1
            == "Abstract class defining semantics of the tensor elements. We are relying on Python for inheritance checking"
        )
        neural_type_comparison_result_0 = element_type_0.compare(
            element_type_0)
        assert loss_type_0 is not None
        assert normal_distribution_samples_type_0 is not None
        assert logprobs_type_0 is not None
        assert string_type_0 is not None
        assert probs_type_0 is not None
        assert predictions_type_0 is not None
        assert element_type_0 is not None
        assert normal_distribution_log_variance_type_0 is not None
        assert normal_distribution_samples_type_1 is not None
        assert element_type_1 is not None
        assert neural_type_comparison_result_0 == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.SAME == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.LESS == module_1.NeuralTypeComparisonResult.LESS
        assert module_1.NeuralTypeComparisonResult.GREATER == module_1.NeuralTypeComparisonResult.GREATER
        assert (
            module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            == module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        )
        assert module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME == module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME
        assert (
            module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
            == module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
        )
        assert module_1.NeuralTypeComparisonResult.INCOMPATIBLE == module_1.NeuralTypeComparisonResult.INCOMPATIBLE
        assert (
            module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            == module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        )
        assert module_1.NeuralTypeComparisonResult.UNCHECKED == module_1.NeuralTypeComparisonResult.UNCHECKED
        probability_distribution_samples_type_0 = module_0.ProbabilityDistributionSamplesType()
        assert probability_distribution_samples_type_0 is not None
        neural_type_comparison_result_1 = element_type_1.compare(
            probability_distribution_samples_type_0)
        assert neural_type_comparison_result_1 == module_1.NeuralTypeComparisonResult.GREATER
        var_2 = element_type_0.__repr__()
        assert var_2 == "ElementType"
        sequence_to_sequence_alignment_type_0 = module_0.SequenceToSequenceAlignmentType(
            *list_1)
    except BaseException:
        pass


def test_case_8():
    try:
        element_type_0 = module_0.ElementType()
        var_0 = element_type_0.__repr__()
        assert var_0 == "ElementType"
        element_type_1 = module_0.ElementType()
        neural_type_comparison_result_0 = element_type_1.compare(
            element_type_0)
        assert element_type_0 is not None
        assert element_type_1 is not None
        assert neural_type_comparison_result_0 == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.SAME == module_1.NeuralTypeComparisonResult.SAME
        assert module_1.NeuralTypeComparisonResult.LESS == module_1.NeuralTypeComparisonResult.LESS
        assert module_1.NeuralTypeComparisonResult.GREATER == module_1.NeuralTypeComparisonResult.GREATER
        assert (
            module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            == module_1.NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        )
        assert module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME == module_1.NeuralTypeComparisonResult.TRANSPOSE_SAME
        assert (
            module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
            == module_1.NeuralTypeComparisonResult.CONTAINER_SIZE_MISMATCH
        )
        assert module_1.NeuralTypeComparisonResult.INCOMPATIBLE == module_1.NeuralTypeComparisonResult.INCOMPATIBLE
        assert (
            module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            == module_1.NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        )
        assert module_1.NeuralTypeComparisonResult.UNCHECKED == module_1.NeuralTypeComparisonResult.UNCHECKED
        string_type_0 = module_0.StringType()
        assert string_type_0 is not None
        str_0 = ">C%$^5Il:t_\r"
        dict_0 = {str_0: string_type_0}
        probability_distribution_samples_type_0 = module_0.ProbabilityDistributionSamplesType(
            **dict_0)
    except BaseException:
        pass
