import numpy as np
import pytest

pytest.importorskip("discopy")

from discopy.markov import Box, Copy, Discard, Id, Swap, Ty
from discopy.tensor import Dim

from fxtensor_salmon import FXTensor
from fxtensor_salmon.discopy import (
    FXTensorFunctor,
    from_box,
    from_diagram,
    from_tensor,
    to_box,
    to_diagram,
    to_tensor,
)


class TestTensorConversion:
    def test_unlabeled_roundtrip(self):
        tensor = FXTensor(
            [[2], [3]],
            data=np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]),
        )
        converted = to_tensor(tensor)
        assert converted.dom == Dim(2)
        assert converted.cod == Dim(3)
        assert np.allclose(converted.array, tensor.data)
        assert from_tensor(converted) == tensor

    def test_state_roundtrip(self):
        state = FXTensor([[], [2]], data=np.array([0.3, 0.7]))
        converted = to_tensor(state)
        assert converted.dom == Dim()
        assert converted.cod == Dim(2)
        assert from_tensor(converted) == state

    def test_scalar_roundtrip(self):
        scalar = FXTensor([[], []], data=np.array(1.0))
        converted = to_tensor(scalar)
        assert converted.dom == Dim()
        assert converted.cod == Dim()
        assert from_tensor(converted) == scalar

    def test_labeled_dims_roundtrip_restores_labels(self):
        tensor = FXTensor(
            [[['a', 'b']], [['x', 'y', 'z']]],
            data=np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]),
        )
        converted = to_tensor(tensor)
        restored = from_tensor(converted, labels=tensor.labels)
        assert restored == tensor

    def test_composition_agrees(self):
        left = FXTensor(
            [[2], [2]],
            data=np.array([[0.2, 0.8], [0.6, 0.4]]),
        )
        right = FXTensor(
            [[2], [2]],
            data=np.array([[0.3, 0.7], [0.9, 0.1]]),
        )
        composed = left.composition(right)
        discopy_composed = to_tensor(left) >> to_tensor(right)
        assert np.allclose(composed.data, [[0.78, 0.22], [0.54, 0.46]])
        assert from_tensor(discopy_composed) == composed

    def test_tensor_product_agrees(self):
        left = FXTensor([[], [2]], data=np.array([0.3, 0.7]))
        right = FXTensor([[], [3]], data=np.array([0.2, 0.3, 0.5]))
        product = left.tensor_product(right)
        discopy_product = to_tensor(left) @ to_tensor(right)
        assert np.allclose(product.data, [[0.06, 0.09, 0.15], [0.14, 0.21, 0.35]])
        assert from_tensor(discopy_product) == product


class TestBoxConversion:
    def test_unlabeled_box_roundtrip(self):
        tensor = FXTensor(
            [[2], [3]],
            data=np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]),
        )
        box = to_box(tensor, name="f")
        assert box.name == "f"
        assert from_box(box) == tensor

    def test_labeled_box_roundtrip(self):
        tensor = FXTensor(
            [[['Sunny', 'Rainy']], [['Sunny', 'Rainy']]],
            data=np.array([[0.8, 0.2], [0.4, 0.6]]),
        )
        box = to_box(tensor, name="forecast")
        restored = from_box(box)
        assert restored == tensor
        assert restored.labels == ([['Sunny', 'Rainy']], [['Sunny', 'Rainy']])

    def test_to_diagram_is_box(self):
        tensor = FXTensor([[2], [2]], data=np.eye(2))
        diagram = to_diagram(tensor, name="id")
        assert from_diagram(diagram) == tensor

    def test_from_box_requires_data(self):
        x = Ty("x")
        with pytest.raises(ValueError, match="no data"):
            from_box(Box("empty", x, x))


class TestFXTensorFunctor:
    def test_interprets_copy_discard_swap_id(self):
        x = Ty("x")
        y = Ty("y")
        functor = FXTensorFunctor({x: 2, y: 3}, {})
        assert functor(Copy(x)) == FXTensor.copy_tensor([2])
        assert functor(Discard(x)) == FXTensor.exclamation([2])
        assert functor(Swap(x, y)) == FXTensor.swap([2], [3])
        assert functor(Id(x)) == FXTensor.identity_tensor([2])

    def test_interprets_custom_box(self):
        x = Ty("x")
        kernel = FXTensor(
            [[2], [2]],
            data=np.array([[0.8, 0.2], [0.4, 0.6]]),
        )
        box = Box("forecast", x, x)
        functor = FXTensorFunctor({x: 2}, {box: kernel})
        assert functor(box) == kernel
        assert functor(Id(x) >> box) == kernel
        assert functor(box >> Copy(x)) == kernel.composition(FXTensor.copy_tensor([2]))

    def test_tensor_product_of_boxes(self):
        x = Ty("x")
        y = Ty("y")
        left = FXTensor([[], [2]], data=np.array([0.3, 0.7]))
        right = FXTensor([[], [3]], data=np.array([0.2, 0.3, 0.5]))
        box_x = Box("px", Ty(), x)
        box_y = Box("py", Ty(), y)
        functor = FXTensorFunctor({x: 2, y: 3}, {box_x: left, box_y: right})
        assert functor(box_x @ box_y) == left.tensor_product(right)

    def test_labeled_copy(self):
        weather = Ty("Weather")
        labels = ["Sunny", "Rainy"]
        functor = FXTensorFunctor({weather: labels}, {})
        copied = functor(Copy(weather))
        assert copied == FXTensor.copy_tensor([labels])
        discarded = functor(Discard(weather))
        assert discarded.profile == [[2], []]
        assert discarded.labels == ([['Sunny', 'Rainy']], None)

    def test_array_ar_map(self):
        x = Ty("x")
        box = Box("f", x, x)
        functor = FXTensorFunctor({x: 2}, {box: [[0.8, 0.2], [0.4, 0.6]]})
        result = functor(box)
        assert result.profile == [[2], [2]]
        assert np.allclose(result.data, [[0.8, 0.2], [0.4, 0.6]])

    def test_from_diagram_evaluates_box_then_copy(self):
        kernel = FXTensor(
            [[['a', 'b']], [['a', 'b']]],
            data=np.array([[0.8, 0.2], [0.4, 0.6]]),
        )
        box = to_box(kernel, name="k")
        diagram = box >> Copy(box.cod)
        result = from_diagram(diagram)
        expected = kernel.composition(FXTensor.copy_tensor([['a', 'b']]))
        assert result == expected
