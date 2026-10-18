DisCoPy
=======

``fxtensor_salmon.discopy`` converts tensors to and from `DisCoPy
<https://discopy.org>`_ Markov diagrams. Importing the module requires the
``discopy`` extra and Python 3.10 or newer:

.. code-block:: console

   pip install "fxtensor-salmon[discopy]"

Each profile factor is one wire. :func:`~fxtensor_salmon.discopy.to_tensor`
drops labels. Pass them back to :func:`~fxtensor_salmon.discopy.from_tensor`
if you need them. :func:`~fxtensor_salmon.discopy.to_box` stores labels in the
Markov type names, and :func:`~fxtensor_salmon.discopy.from_box` restores
them. :func:`~fxtensor_salmon.discopy.to_diagram` is the box as a diagram.

:class:`~fxtensor_salmon.discopy.FXTensorFunctor` sends DisCoPy ``Copy``,
``Discard``, ``Swap``, and ``Id`` to :meth:`~fxtensor_salmon.FXTensor.copy_tensor`,
:meth:`~fxtensor_salmon.FXTensor.exclamation`,
:meth:`~fxtensor_salmon.FXTensor.swap`, and
:meth:`~fxtensor_salmon.FXTensor.identity_tensor`. An object map sends an
atomic type to an ``int``, a label list, or a DisCoPy ``Dim``.

.. literalinclude:: ../../examples/discopy_bridge.py
   :language: python
   :start-after: # docs:example discopy
   :end-before: # docs:end discopy
