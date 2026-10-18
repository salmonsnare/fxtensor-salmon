Structural morphisms
====================

These constructors build the deterministic wiring of a Markov category.
``copy_tensor``, ``identity_tensor``, and ``delta_tensor`` accept either
integer lengths or lists of string labels. ``swap``, ``exclamation``, and
``unit_tensor`` take integer lengths only.

Identity, delta, and copy
-------------------------

:meth:`~fxtensor_salmon.FXTensor.identity_tensor` is the identity kernel.
:meth:`~fxtensor_salmon.FXTensor.delta_tensor` is an alias of it.
:meth:`~fxtensor_salmon.FXTensor.copy_tensor` is the diagonal
``X → X⊗n``. ``n=1`` is the identity, ``n=0`` is discard (all ones on the
domain), and a negative ``n`` raises ``ValueError``. An empty factor list
builds the scalar ``1`` on the monoidal unit.

.. literalinclude:: ../../examples/structural.py
   :language: python
   :start-after: # docs:example identity-copy
   :end-before: # docs:end identity-copy

Swap, discard, and the all-ones state
--------------------------------------

:meth:`~fxtensor_salmon.FXTensor.swap` exchanges two blocks of factors.
:meth:`~fxtensor_salmon.FXTensor.exclamation` is discard ``X → I``, an
all-ones array on the domain and an empty codomain.
:meth:`~fxtensor_salmon.FXTensor.unit_tensor` is a state of all ones. It is
not normalized, and it is not the monoidal unit (that is the empty profile).

.. literalinclude:: ../../examples/structural.py
   :language: python
   :start-after: # docs:example swap-discard
   :end-before: # docs:end swap-discard

Copy laws
---------

On these tensors, copy is cocommutative, coassociative, and counital, and
discard is natural: ``f ; ! = !`` when ``f`` is a kernel. A deterministic
kernel commutes with copy.

.. figure:: ../../diagrams/copy_associativity.svg
   :alt: Coassociativity of copy
   :width: 200px
   :align: center

.. figure:: ../../diagrams/copy_counitality.svg
   :alt: Counitality of copy
   :width: 200px
   :align: center

.. figure:: ../../diagrams/copy_commutativity.svg
   :alt: Cocommutativity of copy
   :width: 180px
   :align: center

.. figure:: ../../diagrams/discard_naturality.svg
   :alt: Discard is natural, so f composed with discard equals discard
   :width: 180px
   :align: center

.. literalinclude:: ../../examples/structural.py
   :language: python
   :start-after: # docs:example copy-laws
   :end-before: # docs:end copy-laws
