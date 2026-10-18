"""DisCoPy bridge examples included by the documentation."""

import numpy as np
from discopy.markov import Copy, Discard, Id, Swap, Ty

from fxtensor_salmon import FXTensor
from fxtensor_salmon.discopy import (
    FXTensorFunctor,
    from_box,
    from_diagram,
    from_tensor,
    to_box,
    to_diagram,
    to_tensor,
)

# docs:example discopy
import numpy as np
from discopy.markov import Copy, Discard, Id, Swap, Ty
from fxtensor_salmon import FXTensor
from fxtensor_salmon.discopy import (
    FXTensorFunctor,
    from_box,
    from_tensor,
    to_box,
    to_diagram,
    to_tensor,
)

kernel = FXTensor([[2], [2]], data=np.array([[0.8, 0.2], [0.4, 0.6]]))
assert from_tensor(to_tensor(kernel)) == kernel

weather = ["Sunny", "Rainy"]
forecast = FXTensor([[weather], [weather]], data=np.array([[0.8, 0.2], [0.4, 0.6]]))
assert from_tensor(to_tensor(forecast)).labels == (None, None)
assert from_tensor(to_tensor(forecast), labels=forecast.labels) == forecast
box = to_box(forecast, name="forecast")
assert from_box(box) == forecast
assert from_diagram(to_diagram(forecast, name="forecast")) == forecast

x, y = Ty("x"), Ty("y")
functor = FXTensorFunctor({x: 2, y: 3}, {})
assert functor(Copy(x)) == FXTensor.copy_tensor([2])
assert functor(Discard(x)) == FXTensor.exclamation([2])
assert functor(Swap(x, y)) == FXTensor.swap([2], [3])
assert functor(Id(x)) == FXTensor.identity_tensor([2])
# docs:end discopy
