DisCoPy 連携
=============

``fxtensor_salmon.discopy`` は、テンソルを `DisCoPy <https://discopy.org>`_ の
マルコフ図式と相互に変換します。このモジュールの import には ``discopy`` extra と
Python 3.10 以上が必要です。

.. code-block:: console

   pip install "fxtensor-salmon[discopy]"

プロファイルの各因子が1本のワイヤです。
:func:`~fxtensor_salmon.discopy.to_tensor` はラベルを落とします。必要なら
:func:`~fxtensor_salmon.discopy.from_tensor` に戻してください。
:func:`~fxtensor_salmon.discopy.to_box` はラベルをマルコフ型の名前に格納し、
:func:`~fxtensor_salmon.discopy.from_box` が復元します。
:func:`~fxtensor_salmon.discopy.to_diagram` はその箱を図式にしたものです。

:class:`~fxtensor_salmon.discopy.FXTensorFunctor` は、DisCoPy の ``Copy``、
``Discard``、``Swap``、``Id`` を
:meth:`~fxtensor_salmon.FXTensor.copy_tensor`、
:meth:`~fxtensor_salmon.FXTensor.exclamation`、
:meth:`~fxtensor_salmon.FXTensor.swap`、
:meth:`~fxtensor_salmon.FXTensor.identity_tensor` に送ります。対象の対応は、
原子型を ``int``、ラベルのリスト、または DisCoPy の ``Dim`` に写します。

.. literalinclude:: ../../examples/discopy_bridge.py
   :language: python
   :start-after: # docs:example discopy
   :end-before: # docs:end discopy
