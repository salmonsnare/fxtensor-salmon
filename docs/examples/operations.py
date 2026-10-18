"""Operation examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example marginal
import numpy as np
from fxtensor_salmon import FXTensor

location_labels = ["Urban", "Rural"]
season_labels = ["Spring", "Summer", "Other"]
weather_labels = ["Sunny", "Rainy"]
process_data = np.array(
    [
        [[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]],
        [[0.1, 0.2], [0.2, 0.2], [0.1, 0.2]],
    ]
)
process_tensor = FXTensor(
    [[location_labels], [season_labels, weather_labels]],
    data=process_data,
)
season_tensor = process_tensor.marginalization(start_B=2)
assert season_tensor.labels == ([location_labels], [season_labels])
assert np.allclose(
    season_tensor.data,
    [
        [0.3, 0.4, 0.3],
        [0.3, 0.4, 0.3],
    ],
)

state = FXTensor(
    [[], [2, 3]],
    data=np.array(
        [
            [0.1, 0.2, 0.3],
            [0.15, 0.05, 0.2],
        ]
    ),
)
assert state.marginalization(3) is state
marginal = state.marginalization(2)
assert marginal.profile == [[], [2]]
assert np.allclose(marginal.data, [0.6, 0.4])
assert state.marginalization(1).profile == [[], []]
# docs:end marginal

# docs:example discard-prefix
import numpy as np
from fxtensor_salmon import FXTensor

labeled = FXTensor(
    [[], [["a", "b"], ["x", "y"]]],
    data=np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    ),
)
assert labeled.discard_prefix(1) is labeled
prefix = labeled.discard_prefix(2)
assert prefix.labels == (None, [["x", "y"]])
assert np.allclose(prefix.data, [0.4, 0.6])
kept = labeled.marginalization(2)
assert kept.labels == (None, [["a", "b"]])
assert np.allclose(kept.data, [0.3, 0.7])
# docs:end discard-prefix

# docs:example partial
import numpy as np
from fxtensor_salmon import FXTensor

f = FXTensor(
    [[2], [2, 2]],
    data=np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ),
)
g = FXTensor(
    [[2], [2]],
    data=np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    ),
)
partial = f.partial_composition(g, 2)
assert partial.profile == [[2], [2, 2]]
assert partial.labels == (None, None)
assert np.allclose(
    partial.data,
    [
        [[0.2, 0.8], [0.6, 0.4]],
        [[0.6, 0.4], [0.2, 0.8]],
    ],
)

whole = FXTensor([[2], [2]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
nxt = FXTensor([[2], [2]], data=np.array([[0.3, 0.7], [0.9, 0.1]]))
assert np.allclose(whole.partial_composition(nxt, 1).data, whole.composition(nxt).data)
# docs:end partial

# docs:example joint
import numpy as np
from fxtensor_salmon import FXTensor

location_labels = ["Urban", "Rural"]
traffic_labels = ["Low", "High"]
location_state = FXTensor([[], [location_labels]], data=np.array([0.6, 0.4]))
traffic_state = FXTensor([[], [traffic_labels]], data=np.array([0.7, 0.3]))
joint_state = location_state.jointification(traffic_state)
assert joint_state.labels == (None, [location_labels, traffic_labels])
assert np.allclose(
    joint_state.data,
    [
        [0.42, 0.18],
        [0.28, 0.12],
    ],
)

px = FXTensor([[], [["a", "b"]]], data=np.array([0.3, 0.7]))
py = FXTensor([[], [["x", "y", "z"]]], data=np.array([0.2, 0.3, 0.5]))
assert np.allclose(px.jointification(py).data, px.tensor_product(py).data)

kernel = FXTensor([[2], [2]], data=np.eye(2))
try:
    location_state.jointification(kernel)
except ValueError:
    pass
else:
    raise AssertionError("jointification requires two states")
# docs:end joint

# docs:example diagram-marginals
import numpy as np
from fxtensor_salmon import FXTensor

labeled = FXTensor(
    [[], [["a", "b"], ["x", "y"]]],
    data=np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    ),
)
first = labeled.first_marginalization(2)
second = labeled.second_marginalization(2)
assert first.labels == (None, None)
assert second.labels == (None, None)
assert np.allclose(first.data, labeled.marginalization(2).data)
assert np.allclose(second.data, labeled.discard_prefix(2).data)
# docs:end diagram-marginals

# docs:example product-labels
import numpy as np
from fxtensor_salmon import FXTensor

f = FXTensor([[["a", "b"]], [["x", "y"]]], data=np.array([[0.2, 0.8], [0.6, 0.4]]))
g = FXTensor.identity_tensor([2])
assert f.tensor_product(g).labels == (None, None)
scalar = FXTensor([[], []], data=np.array(1.0))
assert f.tensor_product(scalar).labels == f.labels

left = FXTensor([[2], [2]], data=np.eye(2))
right = FXTensor([[3], [1]], data=np.ones((3, 1)))
try:
    left.composition(right)
except ValueError:
    pass
else:
    raise AssertionError("composition requires a matching codomain and domain")
# docs:end product-labels
