Concepts
========

Profile
-------

The constructor accepts three profile forms.

- **Labeled.** ``[[domain groups], [codomain groups]]`` where each group is a
  list of strings. The dimension of a group is its length.
- **Numeric.** ``[[domain lengths], [codomain lengths]]`` where each entry is
  an ``int``.
- **Tuple.** ``(domain lengths, codomain lengths)``. This form is unlabeled.

``profile`` always returns numeric lengths, ``[[domain], [codomain]]``.
``labels`` returns ``(domain groups, codomain groups)``. An empty side, or a
side that was never labeled, is ``None`` rather than ``[]``.

Axis ``0`` is the first domain factor. After the domain factors, axis numbers
continue into the codomain. ``get_label_index(axis, label)`` returns the
position of ``label`` on that axis. ``get_index_label(axis, index)`` is the
inverse. Both raise ``ValueError`` when the tensor has no labels, or when
``label`` is not on that axis.

.. literalinclude:: ../../examples/concepts.py
   :language: python
   :start-after: # docs:example profiles
   :end-before: # docs:end profiles

Data
----

``data`` is a NumPy array of shape ``tuple(domain_lengths + codomain_lengths)``.
Omitted data is all zeros. A shape that does not match the profile raises
``ValueError``. The empty profile ``[[], []]`` is the monoidal unit; its data
is a 0-d array.

.. literalinclude:: ../../examples/concepts.py
   :language: python
   :start-after: # docs:example defaults
   :end-before: # docs:end defaults

States and kernels
------------------

A state has an empty domain. A kernel has a nonempty domain. Several methods
are restricted to states because the corresponding string diagram starts from
the monoidal unit: :meth:`~fxtensor_salmon.FXTensor.conditionalization`,
:meth:`~fxtensor_salmon.FXTensor.jointification`, and the ``prior`` argument of
:meth:`~fxtensor_salmon.FXTensor.bayesian_inversion` and
:meth:`~fxtensor_salmon.FXTensor.almost_surely_equal`.

Equality uses ``numpy.allclose`` with ``FXTensor._RTOL`` (``1e-05``) and
``FXTensor._ATOL`` (``1e-08``), and it also compares profiles and labels.
``copy`` returns a new tensor with its own array.
