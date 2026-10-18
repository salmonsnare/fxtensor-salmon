"""Probabilistic-property examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example markov
import numpy as np
from fxtensor_salmon import FXTensor

state = FXTensor([[], [2]], data=np.array([0.2, 0.8]))
assert not state.is_markov()
assert np.isclose(state.data.sum(), 1.0)

kernel = FXTensor([[2], [2]], data=np.array([[0.2, 0.8], [0.0, 0.0]]))
assert kernel.is_markov()
assert not kernel.is_deterministic()

bad = FXTensor([[2], [2]], data=np.array([[0.2, 0.2], [0.5, 0.5]]))
assert not bad.is_markov()

identity = FXTensor.identity_tensor([2])
assert identity.is_deterministic()
assert identity.is_markov()
zeros = FXTensor([[2], [2]], data=np.zeros((2, 2)))
assert not zeros.is_deterministic()
# docs:end markov

# docs:example equality
import numpy as np
from fxtensor_salmon import FXTensor

left = FXTensor([[2], [1]], data=np.array([[1.0], [1.0]]))
right = FXTensor([[2], [1]], data=np.array([[1.0], [1.0 + 1e-9]]))
assert left == right
assert hash(left) != hash(right)

labeled = FXTensor([[["a", "b"]], [["x", "y"]]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
relabeled = FXTensor([[["a", "b"]], [["y", "x"]]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
assert labeled != relabeled

cloned = labeled.copy()
cloned.data[0, 0] = 0.0
assert labeled.data[0, 0] == 0.2
assert cloned is not labeled
# docs:end equality
