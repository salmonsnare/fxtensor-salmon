"""Exchangeability examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example coin
import numpy as np
from fxtensor_salmon import FXTensor

bias = ["fair", "heads"]
side = ["H", "T"]
prior = FXTensor([[], [bias]], data=np.array([0.5, 0.5]))
kernel = FXTensor(
    [[bias], [side]],
    data=np.array([
        [0.5, 0.5],  # fair
        [0.9, 0.1],  # heads
    ]),
)
flips = FXTensor.mixture_of_iid(prior, kernel, 3)
assert flips.labels == (None, [side, side, side])
assert flips.is_exchangeable()
sequence = [FXTensor.mixture_of_iid(prior, kernel, n) for n in range(1, 5)]
assert FXTensor.is_kolmogorov_consistent(sequence)
# The two-flip marginal is not the product of its one-flip marginals.
pair = flips.marginalization(3)
assert not pair.is_independent(2)

point = FXTensor([[], [bias]], data=np.array([0.0, 1.0]))
fixed = FXTensor.mixture_of_iid(point, kernel, 3)
assert fixed.is_exchangeable()
assert fixed.is_independent(2)
assert fixed == FXTensor.iid(FXTensor([[], [side]], data=np.array([0.9, 0.1])), 3)
# docs:end coin
