"""Concept examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example profiles
import numpy as np
from fxtensor_salmon import FXTensor

labeled = FXTensor(
    [[["Urban", "Rural"]], [["Spring", "Summer", "Other"], ["Sunny", "Rainy"]]],
    data=np.zeros((2, 3, 2)),
)
assert labeled.profile == [[2], [3, 2]]
assert labeled.labels == (
    [["Urban", "Rural"]],
    [["Spring", "Summer", "Other"], ["Sunny", "Rainy"]],
)
assert labeled.get_label_index(0, "Rural") == 1
assert labeled.get_label_index(1, "Summer") == 1
assert labeled.get_index_label(2, 0) == "Sunny"

numeric = FXTensor([[2], [3, 2]], data=np.zeros((2, 3, 2)))
assert numeric.labels == (None, None)

as_tuple = FXTensor(([2], [3]), data=np.zeros((2, 3)))
assert as_tuple.profile == [[2], [3]]
assert as_tuple.labels == (None, None)
# docs:end profiles

# docs:example defaults
import numpy as np
from fxtensor_salmon import FXTensor

zeros = FXTensor([[2], [2]])
assert np.array_equal(zeros.data, np.zeros((2, 2)))

scalar = FXTensor([[], []], data=np.array(1.0))
assert scalar.profile == [[], []]
assert scalar.data.shape == ()
assert float(scalar.data) == 1.0

try:
    FXTensor([[2], [2]], data=np.ones((2, 3)))
except ValueError:
    pass
else:
    raise AssertionError("data shape must match the profile")

plain = FXTensor([[2], [2]], data=np.eye(2))
try:
    plain.get_label_index(0, "a")
except ValueError:
    pass
else:
    raise AssertionError("unlabeled tensors have no label lookup")
# docs:end defaults
