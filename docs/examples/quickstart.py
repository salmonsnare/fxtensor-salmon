"""Quickstart examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example labeled-tensor
import numpy as np
from fxtensor_salmon import FXTensor

profile = [[["a", "b"]], [["x", "y", "z"]]]
data = np.array(
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
    ]
)
tensor = FXTensor(profile, data=data)
assert tensor.profile == [[2], [3]]
assert tensor.labels == ([["a", "b"]], [["x", "y", "z"]])
assert tensor.get_label_index(0, "a") == 0
assert tensor.get_index_label(1, 2) == "z"
# docs:end labeled-tensor

# docs:example unlabeled-tensor
import numpy as np
from fxtensor_salmon import FXTensor

profile = [[2], [3]]
data = np.array(
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
    ]
)
tensor = FXTensor(profile, data=data)
assert tensor.profile == [[2], [3]]
assert tensor.labels == (None, None)
# docs:end unlabeled-tensor

# docs:example strands
import numpy as np
from fxtensor_salmon import FXTensor

profile = [[["a", "b"]], [["x", "y", "z"]]]
strands = {
    "[[['a']], [['x']]]": 0.1,
    "[[['a']], [['y']]]": 0.2,
    "[[['a']], [['z']]]": 0.7,
    "[[['b']], [['x']]]": 0.3,
    "[[['b']], [['y']]]": 0.3,
    "[[['b']], [['z']]]": 0.4,
}
tensor = FXTensor.from_strands(profile, strands)
assert tensor.labels == ([["a", "b"]], [["x", "y", "z"]])
assert np.allclose(
    tensor.data,
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
    ],
)
# docs:end strands

# docs:example composition
import numpy as np
from fxtensor_salmon import FXTensor

tensor1 = FXTensor(
    [[["a", "b"]], [["x", "y"]]],
    data=np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    ),
)
tensor2 = FXTensor(
    [[["x", "y"]], [["p", "q"]]],
    data=np.array(
        [
            [0.3, 0.7],
            [0.9, 0.1],
        ]
    ),
)
result = tensor1.composition(tensor2)
assert result.labels == ([["a", "b"]], [["p", "q"]])
assert np.allclose(
    result.data,
    [
        [0.2 * 0.3 + 0.8 * 0.9, 0.2 * 0.7 + 0.8 * 0.1],
        [0.6 * 0.3 + 0.4 * 0.9, 0.6 * 0.7 + 0.4 * 0.1],
    ],
)
# docs:end composition

# docs:example tensor-product
import numpy as np
from fxtensor_salmon import FXTensor

tensor1 = FXTensor([[], [["a", "b"]]], data=np.array([0.3, 0.7]))
tensor2 = FXTensor([[], [["x", "y", "z"]]], data=np.array([0.2, 0.3, 0.5]))
result = tensor1.tensor_product(tensor2)
assert result.profile == [[], [2, 3]]
assert result.labels == (None, [["a", "b"], ["x", "y", "z"]])
assert np.allclose(result.data, np.outer(np.array([0.3, 0.7]), np.array([0.2, 0.3, 0.5])))
# docs:end tensor-product

# docs:example weather-en
import numpy as np
from fxtensor_salmon import FXTensor

weather_states = ["Sunny", "Rainy"]
sunny_today = FXTensor([[], [weather_states]], data=np.array([1.0, 0.0]))
forecast_matrix = np.array(
    [
        [0.8, 0.2],
        [0.4, 0.6],
    ]
)
forecast_tensor = FXTensor([[weather_states], [weather_states]], data=forecast_matrix)
sunny_tomorrow = sunny_today.composition(forecast_tensor)
sunny_idx = sunny_tomorrow.get_label_index(0, "Sunny")
assert np.isclose(sunny_tomorrow.data[sunny_idx], 0.8)
assert sunny_tomorrow.labels == (None, [["Sunny", "Rainy"]])
# docs:end weather-en

# docs:example weather-ja
import numpy as np
from fxtensor_salmon import FXTensor

weather_states = ["晴れ", "雨"]
sunny_today = FXTensor([[], [weather_states]], data=np.array([1.0, 0.0]))
forecast_matrix = np.array(
    [
        [0.8, 0.2],
        [0.4, 0.6],
    ]
)
forecast_tensor = FXTensor([[weather_states], [weather_states]], data=forecast_matrix)
sunny_tomorrow = sunny_today.composition(forecast_tensor)
sunny_idx = sunny_tomorrow.get_label_index(0, "晴れ")
assert np.isclose(sunny_tomorrow.data[sunny_idx], 0.8)
assert sunny_tomorrow.labels == (None, [["晴れ", "雨"]])
# docs:end weather-ja
