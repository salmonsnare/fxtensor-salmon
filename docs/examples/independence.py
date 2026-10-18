"""Conditional independence examples included by the documentation."""

import numpy as np

from fxtensor_salmon import FXTensor

# docs:example weather
import numpy as np
from fxtensor_salmon import FXTensor

forecast = ["Sunny", "Rainy"]
weather = ["Dry", "Wet"]
umbrella = ["Leave", "Take"]
# Joints on (weather, forecast) and (umbrella, forecast). Both forecast
# marginals are [0.5, 0.5].
p_wf = FXTensor(
    [[], [weather, forecast]],
    data=np.array(
        [
            [0.45, 0.05],  # Dry
            [0.05, 0.45],  # Wet
        ]
    ),
)
p_uf = FXTensor(
    [[], [umbrella, forecast]],
    data=np.array(
        [
            [0.40, 0.10],  # Leave
            [0.10, 0.40],  # Take
        ]
    ),
)
joint = FXTensor.conditional_product(p_wf, p_uf, 2)
assert joint.labels == (None, [weather, umbrella, forecast])
assert joint.is_conditionally_independent(2, 3)
weather_umbrella = joint.marginalization(3)
assert weather_umbrella.labels == (None, [weather, umbrella])
assert not weather_umbrella.is_independent(2)
# docs:end weather

# docs:example kernel
import numpy as np
from fxtensor_salmon import FXTensor

context = ["Home", "Away"]
report = ["Dry", "Wet"]
carry = ["Leave", "Take"]
outlook = ["Sunny", "Rainy"]
# Each row is a distribution on report ⊗ outlook, and the outlook marginals agree.
p_report = FXTensor(
    [[context], [report, outlook]],
    data=np.array(
        [
            [[0.36, 0.04], [0.04, 0.56]],  # Home
            [[0.16, 0.16], [0.24, 0.44]],  # Away
        ]
    ),
)
p_carry = FXTensor(
    [[context], [carry, outlook]],
    data=np.array(
        [
            [[0.32, 0.12], [0.08, 0.48]],  # Home
            [[0.30, 0.18], [0.10, 0.42]],  # Away
        ]
    ),
)
kernel = FXTensor.conditional_product(p_report, p_carry, 2)
assert kernel.labels == ([context], [report, carry, outlook])
assert kernel.is_conditionally_independent(2, 3)
# docs:end kernel
