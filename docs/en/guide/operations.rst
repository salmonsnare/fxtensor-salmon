Operations
==========

Indices that split a codomain (``start_B``, ``concat_start_index``) are
**1-based** positions along the codomain factors.

Composition and tensor product
------------------------------

:meth:`~fxtensor_salmon.FXTensor.composition` contracts the receiver's codomain
with the argument's domain via ``numpy.tensordot``. The profiles must match or
it raises ``ValueError``. Labels are kept only when both operands have labels.

:meth:`~fxtensor_salmon.FXTensor.tensor_product` concatenates domains and
codomains and multiplies the broadcast arrays. Labels are dropped when one
operand is unlabeled and has at least one axis. An unlabeled scalar (no axes)
does not wipe the other tensor's labels.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example product-labels
   :end-before: # docs:end product-labels

.. figure:: ../../diagrams/composition.svg
   :alt: Sequential composition, read top to bottom
   :width: 140px
   :align: center

.. figure:: ../../diagrams/tensor_product.svg
   :alt: Tensor product of two morphisms
   :width: 180px
   :align: center

Marginalization
---------------

:meth:`~fxtensor_salmon.FXTensor.marginalization` sums codomain axes from
``start_B`` through the end and keeps the labels of the surviving prefix.
``start_B`` equal to ``len(codomain) + 1`` sums nothing and returns the same
object. ``start_B == 1`` sums the whole codomain. The valid range is ``1``
through ``len(codomain) + 1``.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example marginal
   :end-before: # docs:end marginal

:meth:`~fxtensor_salmon.FXTensor.discard_prefix` sums the codomain axes
*before* ``start_B`` and keeps the suffix, including its labels. ``start_B ==
1`` returns the same object.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example discard-prefix
   :end-before: # docs:end discard-prefix

:meth:`~fxtensor_salmon.FXTensor.first_marginalization` and
:meth:`~fxtensor_salmon.FXTensor.second_marginalization` implement the same
splits by composing with identity and :meth:`~fxtensor_salmon.FXTensor.exclamation`.
Those constructors are numeric, so the results are unlabeled. ``first`` drops
the suffix that starts at ``concat_start_index``; ``second`` drops the prefix
before it.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example diagram-marginals
   :end-before: # docs:end diagram-marginals

Partial composition
-------------------

For ``f: A → B ⊗ C`` and ``g: C → D``, ``f.partial_composition(g, k)`` keeps
the codomain prefix before the 1-based index ``k`` and feeds the suffix into
``g``. ``k == 1`` feeds the whole codomain, which matches
:meth:`~fxtensor_salmon.FXTensor.composition` on the values. The implementation
builds a numeric identity on the kept prefix, so labels on the result are
cleared.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example partial
   :end-before: # docs:end partial

Jointification
--------------

:meth:`~fxtensor_salmon.FXTensor.jointification` is the tensor product of two
states. Both domains must be empty. Labels are kept only when both states are
labeled; the domain label group of a state is ``None``.

.. literalinclude:: ../../examples/operations.py
   :language: python
   :start-after: # docs:example joint
   :end-before: # docs:end joint
