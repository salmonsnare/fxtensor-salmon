Probabilistic properties
========================

:meth:`~fxtensor_salmon.FXTensor.is_markov` is true when every domain point's
codomain slice sums to 1 or to 0 (the zero rows produced by conditioning).
A state has an empty domain, so ``is_markov`` returns ``False``. Check a
normalized state with ``numpy.isclose(state.data.sum(), 1)``.

:meth:`~fxtensor_salmon.FXTensor.is_deterministic` is true when every
codomain slice is a point mass: one entry 1 and the rest 0. A zero slice is
not deterministic. Identity, copy, swap, and discard are deterministic;
a generic stochastic kernel is not.

.. literalinclude:: ../../examples/properties.py
   :language: python
   :start-after: # docs:example markov
   :end-before: # docs:end markov

Equality and copying
--------------------

``==`` compares profile, labels, and data. Data uses ``allclose`` with
``FXTensor._RTOL`` and ``FXTensor._ATOL``. ``hash`` uses the raw bytes, so
two tensors can compare equal and still hash differently when the stored
values differ inside that tolerance. Do not use that pair as interchangeable
dictionary keys.

``copy`` (and ``copy.copy``) duplicates the array. Later writes to one
tensor do not change the other.

.. literalinclude:: ../../examples/properties.py
   :language: python
   :start-after: # docs:example equality
   :end-before: # docs:end equality
