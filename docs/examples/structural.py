"""Structural-morphism examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example identity-copy
import numpy as np
from fxtensor_salmon import FXTensor

labels = [["a", "b"]]
identity = FXTensor.identity_tensor(labels)
assert identity.labels == ([["a", "b"]], [["a", "b"]])
assert identity.is_deterministic()

numeric_identity = FXTensor.identity_tensor([2, 3])
assert numeric_identity.profile == [[2, 3], [2, 3]]
assert FXTensor.delta_tensor([2]) == FXTensor.identity_tensor([2])
assert FXTensor.delta_tensor([2]).profile == [[2], [2]]
assert FXTensor.identity_tensor([]).profile == [[], []]

copy = FXTensor.copy_tensor([2])
assert copy.profile == [[2], [2, 2]]
assert np.array_equal(
    copy.data,
    [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
    ],
)
labeled_copy = FXTensor.copy_tensor([["a", "b"]])
assert labeled_copy.labels == ([["a", "b"]], [["a", "b"], ["a", "b"]])
assert FXTensor.copy_tensor([2], n=1) == FXTensor.identity_tensor([2])
discarding_copy = FXTensor.copy_tensor([["a", "b"]], n=0)
assert discarding_copy.profile == [[2], []]
assert discarding_copy.labels == ([["a", "b"]], None)
assert np.array_equal(discarding_copy.data, [1.0, 1.0])
try:
    FXTensor.copy_tensor([2], n=-1)
except ValueError:
    pass
else:
    raise AssertionError("copy_tensor rejects a negative count")
# docs:end identity-copy

# docs:example swap-discard
import numpy as np
from fxtensor_salmon import FXTensor

swap = FXTensor.swap([2], [3])
assert swap.profile == [[2, 3], [3, 2]]
assert swap.is_deterministic()
assert swap.composition(FXTensor.swap([3], [2])) == FXTensor.identity_tensor([2, 3])

discard = FXTensor.exclamation([2, 3])
assert discard.profile == [[2, 3], []]
assert np.all(discard.data == 1)
assert discard.is_deterministic()

unit = FXTensor.unit_tensor([2, 3])
assert unit.profile == [[], [2, 3]]
assert unit.labels == (None, None)
assert np.all(unit.data == 1)
# docs:end swap-discard

# docs:example copy-laws
import numpy as np
from fxtensor_salmon import FXTensor

copy = FXTensor.copy_tensor([2])
identity = FXTensor.identity_tensor([2])
discard = FXTensor.exclamation([2])
assert copy.composition(FXTensor.swap([2], [2])) == copy
assert copy.composition(identity.tensor_product(discard)) == identity
assert copy.composition(discard.tensor_product(identity)) == identity
assert copy.composition(copy.tensor_product(identity)) == FXTensor.copy_tensor([2], n=3)

kernel = FXTensor([[2], [2]], data=np.array([[0.8, 0.2], [0.4, 0.6]]))
assert kernel.composition(FXTensor.exclamation([2])) == FXTensor.exclamation([2])
assert identity.composition(copy) == copy.composition(identity.tensor_product(identity))
# docs:end copy-laws
