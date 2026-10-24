Named axes
==========

Value labels name the points on an axis, such as ``low`` and ``high``.
An axis name names the wire itself, such as ``A`` or ``C``. The same labels
can sit on several wires; the names still tell them apart. Omit ``names``
and the tensor behaves as before.

Pass names as ``(domain names, codomain names)``. They are unique in one
tensor. :attr:`~fxtensor_salmon.FXTensor.names` is ``(None, None)`` when
the tensor is unnamed. :meth:`~fxtensor_salmon.FXTensor.rename` replaces
names, and :meth:`~fxtensor_salmon.FXTensor.axis` returns the index with
domain axes counted first, as in :meth:`~fxtensor_salmon.FXTensor.get_label_index`.

Marginal
--------

:meth:`~fxtensor_salmon.FXTensor.marginal` keeps the named codomain wires
in the order given and sums out the rest, including a wire in the middle.
:meth:`~fxtensor_salmon.FXTensor.marginalize` does the same by listing the
wires to drop; the others stay in their original order.

For a state ``p: I → A ⊗ B ⊗ C``, keeping ``B`` is the deterministic
wiring that discards ``A`` and ``C``:

``p ; (! ⊗ id_B ⊗ !) = p_B``.

.. figure:: ../../_static/figures/named_marginal.svg
   :alt: Discarding named wires A and C leaves B
   :width: 80%
   :align: center

   Read top to bottom. Left: ``p ; (! ⊗ id_B ⊗ !)``. The dots are
   discard. Right: the marginal state on ``B``.

Labels move with the kept wires. The domain is unchanged.

Permute
-------

:meth:`~fxtensor_salmon.FXTensor.permute` reorders the codomain into the
given names. The domain stays put, and labels move with their wires.
For a state this reorders every axis. The reordering is a permutation of
the wires, drawn as crossings:

.. figure:: ../../_static/figures/named_permute.svg
   :alt: Named wires A, B, C reordered to C, A, B
   :width: 80%
   :align: center

   Read top to bottom. Left: ``p`` with wires ``A``, ``B``, ``C`` crossed
   into the order ``C``, ``A``, ``B``. Right: the reordered state.

Observe
-------

:meth:`~fxtensor_salmon.FXTensor.observe` conditions a state on outcomes
written as ``name=label``. Without labels, the value is an integer index.
The method moves the observed wires to the front with :meth:`~fxtensor_salmon.FXTensor.permute`,
applies :meth:`~fxtensor_salmon.FXTensor.conditionalization`, and reads
that row. It returns the posterior joint on the remaining wires, in their
original order, and the probability of the evidence. Several outcomes at
once match observing them one by one; the probability is then the joint
probability of all of them. Evidence 0 raises ``ValueError``.

On a chain ``A → C → D``, the ``A`` marginal of ``observe(C="high")`` is
the row of :meth:`~fxtensor_salmon.FXTensor.bayesian_inversion` for that
value of ``C``.

.. literalinclude:: ../../examples/names.py
   :language: python
   :start-after: # docs:example chain
   :end-before: # docs:end chain

How names travel
----------------

:meth:`~fxtensor_salmon.FXTensor.composition` checks the codomain names of
``self`` against the domain names of ``other`` when both are present.
The result keeps the domain names of ``self`` and the codomain names of
``other``. :meth:`~fxtensor_salmon.FXTensor.tensor_product` concatenates
names and raises ``ValueError`` if a name is repeated. If one factor has
axes but no names, the product is unnamed.
:meth:`~fxtensor_salmon.FXTensor.bayesian_inversion` swaps domain and
codomain names. :meth:`~fxtensor_salmon.FXTensor.conditional` and
:meth:`~fxtensor_salmon.FXTensor.conditionalization` split names at the
same index as the axes.

:meth:`~fxtensor_salmon.FXTensor.to_json` stores names under ``names``
as ``[domain names, codomain names]``. JSON with no ``names`` key loads
as an unnamed tensor.
