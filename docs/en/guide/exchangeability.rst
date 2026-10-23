Exchangeability
===============

A state ``p: I → X^{⊗n}`` is exchangeable when a permutation of the factors
leaves it unchanged. Adjacent transpositions generate the symmetric group, so
it is enough to test those. ``p`` is exchangeable at position ``k``
(1-based) when

``p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p``.

Swapping the ``k``-th and ``(k+1)``-th wires leaves ``p`` unchanged. The
identity wires and the two swapped wires use ``(k-1) + 2 + (n-k-1) = n``
factors.

.. figure:: ../../_static/figures/exchangeability.svg
   :alt: p composed with an adjacent swap equals p
   :width: 100%
   :align: center

   Read top to bottom. Left:
   ``p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)})``.
   Right: ``p``. The crossing swaps the ``k``-th and ``(k+1)``-th wires.

:meth:`~fxtensor_salmon.FXTensor.is_exchangeable` checks ``k = 1 .. n-1``
and compares ``data`` with ``numpy.allclose``. ``swap`` carries no labels,
so labels are not part of the comparison. ``atol`` and ``rtol`` default to
``FXTensor._ATOL`` and ``FXTensor._RTOL``. ``n < 2`` is vacuously true.

``block`` is the dimension list of one factor ``X``. The default, ``None``,
takes each codomain axis as one factor, and those axes must share a size.
A codomain that mixes sizes raises ``ValueError``. The receiver must be a
state (empty domain).

Symmetrize
----------

:meth:`~fxtensor_salmon.FXTensor.symmetrize` averages the state over every
permutation of the ``n`` factors. The result has the same profile and labels
and satisfies ``is_exchangeable`` for the same ``block``. Averaging again
changes nothing, and an already exchangeable state is left unchanged.
``n > 8`` warns, because the average walks through ``n!`` permutations.

Kolmogorov consistency
----------------------

A sequence ``p_n: I → X^{⊗n}`` is Kolmogorov consistent when discarding the
``n``-th wire recovers the previous state:

``p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}``.

.. figure:: ../../_static/figures/kolmogorov_consistency.svg
   :alt: discarding the n-th wire of p_n yields p_{n-1}
   :width: 70%
   :align: center

   Read top to bottom. Left: ``p_n ; (id^{⊗(n-1)} ⊗ !)``. The dot is
   discard on the ``n``-th wire. Right: ``p_{n-1}``.

:meth:`~fxtensor_salmon.FXTensor.is_kolmogorov_consistent` takes
``[p_1, p_2, ..., p_N]``, so ``states[0]`` is ``p_1``. ``!`` is
:meth:`~fxtensor_salmon.FXTensor.exclamation` on ``X``. A list of length 0
or 1 has no pair to compare; a single entry must still be a state
``I → X``. ``block`` is the same as for ``is_exchangeable``. Values are
compared on ``data``. A sequence whose marginal disagrees returns ``False``.
A state that is not ``I → X^{⊗n}`` for its position raises ``ValueError``.

i.i.d. and mixtures
-------------------

:meth:`~fxtensor_salmon.FXTensor.iid` is the tensor product of ``n`` copies
of ``p``. Labels are taken from ``p``. For a state ``I → X`` the result is
``I → X^{⊗n}``.

:meth:`~fxtensor_salmon.FXTensor.mixture_of_iid` builds
``prior ; copy_Θ^{(n)} ; kernel^{⊗n}`` from a state ``prior: I → Θ`` and a
kernel ``Θ → X``. One draw from the prior is copied and then passed through
``n`` independent copies of the kernel. The resulting state ``I → X^{⊗n}``
is exchangeable, and the sequence in ``n`` is Kolmogorov consistent. It is
independent only when the prior is a point mass.

A prior on the bias of a coin makes the sequence of flips exchangeable.
The flips are not independent, unless that prior is a Dirac mass.

.. literalinclude:: ../../examples/exchangeability.py
   :language: python
   :start-after: # docs:example coin
   :end-before: # docs:end coin

The finite form of de Finetti's theorem in a Markov category is proved in
Fritz, Gonda, and Perrone, `De Finetti's theorem in categorical probability
(arXiv:2105.02639) <https://arxiv.org/abs/2105.02639>`_. The same two
conditions — exchangeability along injections, and Kolmogorov consistency
along insertions — appear as coherence of a statistical model in Vaccarino,
`Statistical models as natural transformations (arXiv:2609.22929)
<https://arxiv.org/abs/2609.22929>`_.
