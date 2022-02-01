# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/blob/main/tests/core/test_typecheck.py

import pytest
import torch

from mridc.core.classes.common import Typing, typecheck
from mridc.core.neural_types.comparison import NeuralTypeComparisonResult
from mridc.core.neural_types.elements import CategoricalValuesType, ChannelType, ElementType
from mridc.core.neural_types.neural_type import NeuralType


def recursive_assert_shape(x, shape):
    """Perform recursive shape assert"""
    if isinstance(x, (list, tuple)):
        for xi in x:
            recursive_assert_shape(xi, shape)
        return

    if x.shape != shape:
        raise AssertionError


def recursive_assert_homogeneous_type(x, type_val):
    """Perform recursive type homogeneous assert"""
    if isinstance(x, (list, tuple)):
        for xi in x:
            recursive_assert_homogeneous_type(xi, type_val)
        return

    if x.neural_type.compare(type_val) != NeuralTypeComparisonResult.SAME:
        raise AssertionError


class TestNeuralTypeCheckSystem:
    """Test the typecheck system"""

    @pytest.mark.unit
    def test_no_types_passthrough(self):
        """Test that no types are checked when passing through."""

        class NoTypes(Typing):
            """No types"""

            @typecheck()
            def __call__(self, x):
                return torch.tensor(1.0)

        obj = NoTypes()
        result = obj(torch.tensor(1.0))

        if result != torch.tensor(1.0):
            raise AssertionError
        if hasattr(result, "neural_type"):
            raise AssertionError

    @pytest.mark.unit
    def test_input_output_types(self):
        """Test that input and output types are correctly checked."""

        class InputOutputTypes(Typing):
            """Set input and output types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputOutputTypes()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if result.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        # Test passing wrong key for input
        with pytest.raises(TypeError):
            _ = obj(a=torch.zeros(10))

        # Test using positional args
        with pytest.raises(TypeError):
            _ = obj(torch.zeros(10))

    @pytest.mark.unit
    def test_input_types_only(self):
        """Test that input types are correctly checked."""

        class InputTypes(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputTypes()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if hasattr(result, "neural_type") is not False:
            raise AssertionError

    @pytest.mark.unit
    def test_multiple_input_types_only(self):
        """Test that multiple input types are correctly checked."""

        class InputTypes(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType()), "y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x, y):
                x += y
                return x

        obj = InputTypes()
        result = obj(x=torch.zeros(10), y=torch.ones(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if hasattr(result, "neural_type") is not False:
            raise AssertionError

    @pytest.mark.unit
    def test_output_types_only(self):
        """Test that output types are correctly inferred"""

        class OutputTypes(Typing):
            """Set output types"""

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = OutputTypes()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if result.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        # Test passing positional args
        # Positional args allowed if input types is not set !
        result = obj(torch.zeros(10))
        if result.sum() != torch.tensor(10.0):
            raise AssertionError

    @pytest.mark.unit
    def test_multiple_output_types_only(self):
        """Test multiple output types only"""

        class MultipleOutputTypes(Typing):
            """Set output types"""

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType()), "z": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return y, z

        obj = MultipleOutputTypes()
        result_y, result_z = obj(x=torch.zeros(10))

        if result_y.sum() != torch.tensor(10.0):
            raise AssertionError
        if result_y.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        if result_z.sum() != torch.tensor(20.0):
            raise AssertionError
        if result_z.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only(self):
        """Test that multiple output types can be mixed"""

        class MultipleMixedOutputTypes(Typing):
            """Set output types"""

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType()), "z": [NeuralType(("B",), ElementType())]}

            @typecheck()
            def __call__(self, x):
                y = x + 1
                z = x + 2
                return y, [z, z]

        obj = MultipleMixedOutputTypes()
        result_y, result_z = obj(x=torch.zeros(10))

        if result_y.sum() != torch.tensor(10.0):
            raise AssertionError
        if result_y.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        if result_z[0].sum() != torch.tensor(20.0):
            raise AssertionError
        if result_z[0].neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        if result_z[1].sum() != torch.tensor(20.0):
            raise AssertionError
        if result_z[1].neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

    @pytest.mark.unit
    def test_multiple_mixed_output_types_only_mismatched(self):
        """Test multiple output types with mismatched types"""

        class MultipleMixedOutputTypes(Typing):
            """Set output types"""

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType()), "z": [NeuralType(("B",), ElementType())]}

            @typecheck()
            def __call__(self, x):
                # Use list of y, single z, contrary to signature
                y = x + 1
                z = x + 2
                return [y, y], z

        obj = MultipleMixedOutputTypes()
        with pytest.raises(TypeError):
            _, _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_incorrect_inheritance(self):
        """Test incorrect inheritance"""

        class IncorrectInheritance(object):
            """Inherit from object"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = IncorrectInheritance()

        with pytest.raises(RuntimeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_port_definition_rejection(self):
        """Test port definition rejection"""

        class InputPortDefinitionRejection(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {"w": NeuralType(("B",), ElementType()), "u": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x, y):
                x += 1
                y -= 1
                return x, y

        # Test input port mismatch
        obj = InputPortDefinitionRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10), y=torch.zeros(10))

        class OutputPortDefinitionRejection(Typing):
            """Set output types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {
                    "w": NeuralType(("B",), ElementType()),
                }

            @typecheck()
            def __call__(self, x):
                return x + 1, x - 1

        obj = OutputPortDefinitionRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_port_shape_rejection(self):
        """Test port shape rejection"""

        class InputPortShapeRejection(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B", "T"), ElementType())}  # expect rank 2 matrix

            @property
            def output_types(self):
                """Set output types"""
                return {"w": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        # Test input port mismatch
        obj = InputPortShapeRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

        class OutputPortShapeRejection(Typing):
            """Set output types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {
                    "w": NeuralType(("B", "T", "D"), ElementType()),  # expect rank 3 matrix
                }

            @typecheck()
            def __call__(self, x):
                return x + 1

        obj = OutputPortShapeRejection()

        with pytest.raises(TypeError):
            _ = obj(x=torch.zeros(10))

    @pytest.mark.unit
    def test_positional_args(self):
        """Test positional check on input type."""

        class InputPositional(Typing):
            """Swt input types"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = InputPositional()

        with pytest.raises(TypeError):
            _ = obj(torch.zeros(10))

        class OutputPositionalPassthrough(Typing):
            """
            Test positional pass-through for only output ports defined. NOTE: This is required behaviour to support
            type checking of MRIDC Dataset class during collate_fn() call.
            """

            @property
            def output_types(self):
                """Output port definitions"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

        obj = OutputPositionalPassthrough()
        result = obj(torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError

    @pytest.mark.unit
    def test_optional_types(self):
        """Test optional types"""

        class InputOptionalTypes(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Input port definitions"""
                return {"x": NeuralType(("B",), ElementType()), "y": NeuralType(("B",), ElementType(), optional=True)}

            @typecheck()
            def __call__(self, x, y=None):
                if y is None:
                    x += 1
                else:
                    x += y
                return x

        obj = InputOptionalTypes()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if hasattr(result, "neural_type") is not False:
            raise AssertionError

        result2 = obj(x=torch.zeros(10), y=torch.full([10], fill_value=5, dtype=torch.int32))

        if result2.sum() != torch.tensor(10 * 5):
            raise AssertionError
        if hasattr(result, "neural_type") is not False:
            raise AssertionError

    @pytest.mark.unit
    def test_multi_forward_type(self):
        """Test multi-forward type"""

        class AdaptiveTypeCheck(Typing):
            """Set input types"""

            @property
            def input_types(self):
                """Input port definitions"""
                if self.mode == "train":
                    return {"x": NeuralType(("B",), ElementType())}

                if self.mode == "infer":
                    return {"y": NeuralType(("B",), ChannelType())}

                if self.mode == "eval":
                    return {"x": NeuralType(("B",), ElementType()), "y": NeuralType(("B",), ChannelType())}
                raise ValueError("Wrong mode of operation")

            @property
            def output_types(self):
                """Output port definitions"""
                if self.mode == "train":
                    return {"u": NeuralType(("B",), ElementType())}

                if self.mode == "infer":
                    return {"v": NeuralType(("B",), ChannelType())}

                if self.mode == "eval":
                    return {"u": NeuralType(("B",), ElementType()), "v": NeuralType(("B",), ChannelType())}
                raise ValueError("Wrong mode of operation")

            def __init__(self):
                self.mode = "train"

            def __call__(self, **kwargs):
                # Call should call and forward appropriate method in its own mode
                if self.mode == "train":
                    return self.train_forward(x=kwargs["x"])

                if self.mode == "eval":
                    return self.eval_forward(x=kwargs["x"], y=kwargs["y"])

                if self.mode == "infer":
                    return self.infer_forward(y=kwargs["y"])

            @typecheck()
            def train_forward(self, x):
                """Train forward"""
                return x + 10

            @typecheck()
            def eval_forward(self, x, y):
                """Eval forward"""
                return x - 1, y - 1

            @typecheck()
            def infer_forward(self, y):
                """Infer forward"""
                return y - 10

            @property
            def mode(self):
                """Set mode"""
                return self._mode

            @mode.setter
            def mode(self, val):
                """Set mode"""
                if val not in ["train", "infer", "eval"]:
                    raise ValueError("mode must be either train infer or eval")
                self._mode = val

        obj = AdaptiveTypeCheck()

        x = torch.zeros(10)
        y = torch.full([10], fill_value=5, dtype=torch.int32)

        obj.mode = "train"
        x = obj(x=x)

        if not torch.all(x == 10):
            raise AssertionError
        if x.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        obj.mode = "eval"
        x, y = obj(x=x, y=y)

        if not torch.all(x == 9):
            raise AssertionError
        if not torch.all(y == 4):
            raise AssertionError
        if x.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError
        if y.neural_type.compare(NeuralType(("B",), ChannelType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        obj.mode = "infer"
        y = obj(y=y)

        if not torch.all(y == -6):
            raise AssertionError
        if y.neural_type.compare(NeuralType(("B",), ChannelType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        # Now perform assertions of wrong mode with wrong input combinations
        obj.mode = "train"

        # In train mode, call infer
        with pytest.raises(TypeError):
            _ = obj.eval_forward(x=x, y=y)

        with pytest.raises(TypeError):
            # wrong input + wrong mode
            _ = obj.infer_forward(y=x)

    @pytest.mark.unit
    def test_input_type_override(self):
        """Test that input type override works"""

        class InputTypesOverride(Typing):
            """Test class"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

            @typecheck(input_types={"y": NeuralType(("B",), CategoricalValuesType())})
            def forward(self, y):
                """Forward"""
                y -= 1
                return y

        obj = InputTypesOverride()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if hasattr(result, "neural_type") is not False:
            raise AssertionError

        # Test override
        result2 = obj.forward(y=torch.zeros(10))

        if result2.sum() != torch.tensor(-10.0):
            raise AssertionError
        if hasattr(result2, "neural_type") is not False:
            raise AssertionError

    @pytest.mark.unit
    def test_output_type_override(self):
        """Test overriding output type"""

        class OutputTypes(Typing):
            """Test class"""

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x):
                x += 1
                return x

            @typecheck(output_types={"z": NeuralType(("B",), CategoricalValuesType())})
            def forward(self, z):
                """Forward"""
                z -= 1
                return z

        obj = OutputTypes()
        result = obj(x=torch.zeros(10))

        if result.sum() != torch.tensor(10.0):
            raise AssertionError
        if result.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        # Test passing positional args
        # Positional args allowed if input types is not set !
        result = obj(torch.zeros(10))
        if result.sum() != torch.tensor(10.0):
            raise AssertionError

        # Test override
        result2 = obj.forward(z=torch.zeros(10))

        if result2.sum() != torch.tensor(-10.0):
            raise AssertionError
        if not hasattr(result2, "neural_type"):
            raise AssertionError
        if result2.neural_type.compare(NeuralType(("B",), CategoricalValuesType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

    @pytest.mark.unit
    def test_multi_type_override(self):
        """Test overriding multiple types"""

        class AdaptiveTypeCheck(Typing):
            """Test class"""

            @property
            def input_types(self):
                """__call__ assumed to be for inference only, therefore infer types checked at class scope"""
                return {"y": NeuralType(("B",), ChannelType())}

            @property
            def output_types(self):
                """__call__ assumed to be for inference only, therefore infer types checked at class scope"""
                return {"v": NeuralType(("B",), ChannelType())}

            def __call__(self, **kwargs):
                # Call should call and forward appropriate method in its own mode
                # Let default "forward" call be the infer mode (this is upto developer)
                # Therefore default class level types == infer types
                return self.infer_forward(y=kwargs["y"])

            @typecheck(
                input_types={"x": NeuralType(("B",), ElementType())},
                output_types={"u": NeuralType(("B",), ElementType())},
            )
            def train_forward(self, x):
                """Train forward"""
                return x + 10

            @typecheck(
                input_types={"x": NeuralType(("B",), ElementType()), "y": NeuralType(("B",), ChannelType())},
                output_types={"u": NeuralType(("B",), ElementType()), "v": NeuralType(("B",), ChannelType())},
            )
            def eval_forward(self, x, y):
                """Eval forward"""
                return x - 1, y - 1

            @typecheck(
                input_types={"y": NeuralType(("B",), ChannelType())},
                output_types={"v": NeuralType(("B",), ChannelType())},
            )
            def infer_forward(self, y):
                """Infers output types from input types"""
                return y - 10

        obj = AdaptiveTypeCheck()

        x = torch.zeros(10)
        y = torch.full([10], fill_value=5, dtype=torch.int32)

        # infer mode
        y = obj(y=y)

        if not torch.all(y == -5):
            raise AssertionError
        if y.neural_type.compare(NeuralType(("B",), ChannelType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        x, y = obj.eval_forward(x=x, y=y)

        if not torch.all(x == -1):
            raise AssertionError
        if not torch.all(y == -6):
            raise AssertionError
        if x.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError
        if y.neural_type.compare(NeuralType(("B",), ChannelType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        x = obj.train_forward(x=x)

        if not torch.all(x == 9):
            raise AssertionError
        if x.neural_type.compare(NeuralType(("B",), ElementType())) != NeuralTypeComparisonResult.SAME:
            raise AssertionError

        # In train func, call eval signature
        with pytest.raises(TypeError):
            _ = obj.train_forward(x=x, y=y)

        with pytest.raises(TypeError):
            # wrong input + wrong mode
            _ = obj.infer_forward(x=x)

    @pytest.mark.unit
    def test_disable_typecheck(self):
        """Test disabling typecheck"""

        class InputOutputTypes(Typing):
            """Test class"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": NeuralType(("B",), ElementType())}

            @property
            def output_types(self):
                """Set output types"""
                return {"y": NeuralType(("B",), ElementType())}

            @typecheck()
            def __call__(self, x, **kwargs):
                x += 1
                return x

        # Disable typecheck tests
        with typecheck.disable_checks():
            obj = InputOutputTypes()

            # Execute function without kwarg
            result = obj(torch.zeros(10))

            if result.sum() != torch.tensor(10.0):
                raise AssertionError
            if hasattr(result, "neural_type") is not False:
                raise AssertionError

            # Test passing wrong key for input
            _ = obj(a=torch.zeros(10), x=torch.zeros(5))

    @pytest.mark.unit
    def test_nested_shape_mismatch(self):
        """Test nested shape mismatch"""

        class NestedShapeMismatch(Typing):
            """Test class"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": [[NeuralType(("D",), ElementType())]]}  # Each element of nest will have 4 values

            @property
            def output_types(self):
                """Set output types"""
                return {"y": [[NeuralType(("D",), ElementType())]]}  # Each element of nest will have 4 values

            @typecheck()
            def __call__(self, x):
                # v-- this is to satisfy 1 output constraint, python will otherwise interpret x as a 3 output value
                return x

        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim)

        obj = NestedShapeMismatch()

        # Arbitrary nest 1 (should pass)
        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        result = obj(x=data)

        recursive_assert_shape(result, torch.Size([4]))
        recursive_assert_homogeneous_type(result, NeuralType(("D",), ElementType()))

        # Arbitrary nest 2 (should pass)
        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim, dim)

        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        # Fails since input shape is incorrect
        with pytest.raises(TypeError):
            _ = obj(x=data)

        # Arbitrary nest 3
        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim)

        data = [[[bb(), bb(), bb()]], [[bb()], [bb(), bb()]]]
        # Check should fail since nest level is 3!
        with pytest.raises(TypeError):
            result = obj(x=data)

    @pytest.mark.unit
    def test_nested_mixed_shape_mismatch(self):
        """Test nested mixed shape mismatch"""

        class NestedMixedShapeMismatch(Typing):
            """Test class"""

            @property
            def input_types(self):
                """Set input types"""
                return {"x": [[NeuralType(("D",), ElementType())]]}  # Each element of nest will have 4 values

            @property
            def output_types(self):
                """Set output types"""
                return {"y": [NeuralType(("D",), ElementType())]}  # Each element of nest will have 4 values

            @typecheck()
            def __call__(self, x):
                # v-- this is to satisfy 1 output constraint, python will otherwise interpret x as a 3 output value
                x = x[0]
                return x

        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim)

        obj = NestedMixedShapeMismatch()

        # Arbitrary nest 1 (should pass)
        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        result = obj(x=data)

        recursive_assert_shape(result, torch.Size([4]))
        recursive_assert_homogeneous_type(result, NeuralType(("D",), ElementType()))

        # Arbitrary nest 2 (should pass)
        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim, dim)

        data = [[bb(), bb(), bb()], [bb()], [bb(), bb()]]
        # Fails since input shape is incorrect
        with pytest.raises(TypeError):
            _ = obj(x=data)

        # Arbitrary nest 3
        def bb(dim=4):
            """Basic block"""
            return torch.zeros(dim)

        data = [[[bb(), bb(), bb()]], [[bb()], [bb(), bb()]]]
        # Check should fail since nest level is 3!
        with pytest.raises(TypeError):
            result = obj(x=data)
