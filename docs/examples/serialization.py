"""Serialization examples included by the documentation."""

import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example json
import numpy as np
from fxtensor_salmon import FXTensor

json_data = {
    "profile": [[["a", "b"]], [["x", "y"]]],
    "data": [[0.2, 0.8], [0.6, 0.4]],
}
tensor = FXTensor.from_json(json_data)
assert tensor.labels == ([["a", "b"]], [["x", "y"]])
payload = tensor.to_json()
assert payload["profile"] == [[["a", "b"]], [["x", "y"]]]
assert payload["dtype"] == str(tensor.data.dtype)
assert FXTensor.from_json(payload) == tensor
# docs:end json

# docs:example file
import tempfile
from pathlib import Path

import numpy as np
from fxtensor_salmon import FXTensor

source = FXTensor(
    [[["a", "b"]], [["x", "y"]]],
    data=np.array([[0.2, 0.8], [0.6, 0.4]]),
)
with tempfile.TemporaryDirectory() as tmp:
    filename = str(Path(tmp) / "tensor.json")
    source.save_to_file(filename)
    loaded = FXTensor.load_from_file(filename)
assert loaded == source
# docs:end file

# docs:example strands-io
import numpy as np
from fractions import Fraction
from fxtensor_salmon import FXTensor

profile = [[2], [3]]
strands = {
    "[[[1]], [[1]]]": 0.1,
    "[[[1]], [[2]]]": 0.2,
    "[[[1]], [[3]]]": 0.7,
    "[[[2]], [[1]]]": 0.3,
    "[[[2]], [[2]]]": 0.3,
    "[[[2]], [[3]]]": 0.4,
}
tensor = FXTensor.from_strands(profile, strands)
assert np.allclose(
    tensor.data,
    [
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
    ],
)

half = FXTensor.from_strands([[2], [1]], {"[[[1]], [[1]]]": Fraction(1, 2)})
assert np.isclose(half.data[0, 0], 0.5)

original = FXTensor([[2], [3]], data=np.array([[0.1, 0.0, 0.7], [0.0, 0.3, 0.0]]))
encoded = original.to_original_dict()
assert encoded["profile"] == [[2], [3]]
assert float(encoded["strands"]["[[1], [1]]"]) == 0.1
assert "[[1], [2]]" not in encoded["strands"]

try:
    FXTensor.from_strands([[["a", "b"]], [["x", "y"]]], {"not-a-list": 1.0})
except ValueError:
    pass
else:
    raise AssertionError("strand keys must be nested lists")
# docs:end strands-io
