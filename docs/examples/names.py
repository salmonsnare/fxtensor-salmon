"""Named-axis examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example chain
import numpy as np
from fxtensor_salmon import FXTensor

level = ["low", "high"]
prior_data = np.array([0.25, 0.75])
given_a = np.array([
    [0.8, 0.2],  # A = low
    [0.4, 0.6],  # A = high
])
given_c = np.array([
    [0.9, 0.1],  # C = low
    [0.3, 0.7],  # C = high
])
joint = FXTensor(
    [[], [level, level, level]],
    data=prior_data[:, None, None] * given_a[:, :, None] * given_c[None, :, :],
    names=([], ["A", "C", "D"]),
)
post, evidence = joint.observe(C="high")
assert post.names == (None, ["A", "D"])
assert post.marginal("D").names == (None, ["D"])

prior = FXTensor([[], [level]], data=prior_data, names=([], ["A"]))
kernel = FXTensor([[level], [level]], data=given_a, names=(["A"], ["C"]))
inversion = kernel.bayesian_inversion(prior)
assert np.allclose(post.marginal("A").data, inversion.data[level.index("high")])
assert np.isclose(evidence, joint.marginal("C").data[level.index("high")])
# docs:end chain
