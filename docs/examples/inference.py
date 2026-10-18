"""Conditioning and inference examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example conditionalization
import numpy as np
from fxtensor_salmon import FXTensor

location_labels = ["Urban", "Rural"]
season_labels = ["Spring", "Summer", "Other"]
weather_labels = ["Sunny", "Rainy"]
joint = FXTensor(
    [[], [location_labels, season_labels, weather_labels]],
    data=np.array(
        [
            [[0.08, 0.04], [0.12, 0.04], [0.08, 0.04]],
            [[0.06, 0.12], [0.12, 0.12], [0.06, 0.12]],
        ]
    ),
)
cond_tensor = joint.conditionalization(concat_start_index=3)
assert cond_tensor.labels == ([location_labels, season_labels], [weather_labels])
assert cond_tensor.is_markov()
assert np.allclose(
    cond_tensor.data,
    [
        [[2 / 3, 1 / 3], [0.75, 0.25], [2 / 3, 1 / 3]],
        [[1 / 3, 2 / 3], [0.50, 0.50], [1 / 3, 2 / 3]],
    ],
)

joint_state = FXTensor([[], [2, 2]], data=np.array([[0.1, 0.2], [0.0, 0.0]]))
kernel = joint_state.conditionalization(2)
assert kernel.profile == [[2], [2]]
assert kernel.is_markov()
assert np.allclose(kernel.data, [[1 / 3, 2 / 3], [0.0, 0.0]])

not_a_state = FXTensor([[2], [2]], data=np.eye(2))
try:
    not_a_state.conditionalization(1)
except ValueError:
    pass
else:
    raise AssertionError("conditionalization requires a state")
# docs:end conditionalization

# docs:example conditional
import numpy as np
from fxtensor_salmon import FXTensor

f = FXTensor(
    [[["a"]], [["x0", "x1"], ["y0", "y1"]]],
    data=np.array([[[0.1, 0.3], [0.2, 0.4]]]),
)
c = f.conditional(2)
assert c.labels == ([["a"], ["x0", "x1"]], [["y0", "y1"]])
assert np.allclose(c.data, [[[0.25, 0.75], [1 / 3, 2 / 3]]])

state = FXTensor([[], [2, 2]], data=np.array([[0.1, 0.2], [0.0, 0.0]]))
assert np.allclose(state.conditional(2).data, state.conditionalization(2).data)
# docs:end conditional

# docs:example bayes
import numpy as np
from fxtensor_salmon import FXTensor

health = ["Sick", "Healthy"]
result = ["Positive", "Negative"]
prior = FXTensor([[], [health]], data=np.array([0.01, 0.99]))
test = FXTensor([[health], [result]], data=np.array([[0.9, 0.1], [0.05, 0.95]]))
posterior = test.bayesian_inversion(prior)
assert posterior.labels == ([result], [health])
assert np.isclose(posterior.data[0, 0], 2 / 13)

x = prior.labels[1]
y = test.labels[1]
forward = prior.composition(FXTensor.copy_tensor(x)).composition(
    FXTensor.identity_tensor(x).tensor_product(test)
)
backward = prior.composition(test).composition(FXTensor.copy_tensor(y)).composition(
    posterior.tensor_product(FXTensor.identity_tensor(y))
)
assert forward == backward

point = FXTensor([[], [2]], data=np.array([1.0, 0.0]))
kernel = FXTensor([[2], [2]], data=np.array([[0.0, 1.0], [0.3, 0.7]]))
inverted = kernel.bayesian_inversion(point)
assert np.allclose(inverted.data[0], [0.0, 0.0])
assert np.allclose(inverted.data[1], [1.0, 0.0])
# docs:end bayes

# docs:example almost-sure
import numpy as np
from fxtensor_salmon import FXTensor

prior = FXTensor([[], [["x0", "x1", "x2"]]], data=np.array([0.4, 0.6, 0.0]))
f = FXTensor(
    [[["x0", "x1", "x2"]], [["y0", "y1"]]],
    data=np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]),
)
g = FXTensor(
    [[["x0", "x1", "x2"]], [["y0", "y1"]]],
    data=np.array([[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]),
)
assert np.array_equal(prior.support().data, [1.0, 1.0, 0.0])
assert f.almost_surely_equal(g, prior)
assert not f.almost_surely_equal(g, FXTensor([[], [3]], data=np.array([0.2, 0.2, 0.6])))
assert f.is_absolutely_continuous(g)
assert not g.is_absolutely_continuous(f)

relabeled = FXTensor([[["a", "b", "c"]], [["p", "q"]]], data=np.array(f.data))
assert f.almost_surely_equal(relabeled, prior)
# docs:end almost-sure
