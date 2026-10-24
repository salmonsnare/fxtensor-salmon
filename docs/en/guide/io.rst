Input and output
================

JSON
----

:meth:`~fxtensor_salmon.FXTensor.to_json` returns a dictionary with
``profile``, ``data``, and ``dtype``. The profile is labeled when the tensor
has labels. :meth:`~fxtensor_salmon.FXTensor.from_json` rebuilds a tensor from
``profile`` and ``data``. A ``names`` entry, when present, is the pair of
domain and codomain axis names. JSON with no ``names`` key loads as an
unnamed tensor.

:meth:`~fxtensor_salmon.FXTensor.save_to_file` writes that dictionary as JSON.
:meth:`~fxtensor_salmon.FXTensor.load_from_file` reads it back.

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example json
   :end-before: # docs:end json

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example file
   :end-before: # docs:end file

Strands
-------

:meth:`~fxtensor_salmon.FXTensor.from_strands` reads a map from a strand key
to a weight. A key is a Python literal of nested lists of strings or
integers, parsed without evaluating code; any other key raises
``ValueError``. Labeled strands name the points. Numeric strands use
**1-based** indices, each index wrapped in its own list. Weights may be
``float`` or ``fractions.Fraction`` and are stored with ``float``.

:meth:`~fxtensor_salmon.FXTensor.to_original_dict` goes the other way for the
numeric form: nonzero entries become 1-based index lists. The weights are the
NumPy scalars from ``data``, and the profile is the numeric profile.

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example strands-io
   :end-before: # docs:end strands-io
