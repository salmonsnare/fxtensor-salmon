確率的な性質
============

:meth:`~fxtensor_salmon.FXTensor.is_markov` は、どの domain の点でも codomain
スライスの和が 1 または 0 のときに真です。0 は、条件付けでできるゼロ行のためです。
状態は domain が空なので、``is_markov`` は ``False`` を返します。正規化された
状態は ``numpy.isclose(state.data.sum(), 1)`` で確認します。

:meth:`~fxtensor_salmon.FXTensor.is_deterministic` は、すべての codomain
スライスが点質量（1 が1つで残りが 0）のときに真です。ゼロのスライスは決定的では
ありません。恒等、copy、swap、discard は決定的です。一般の確率核はそうでは
ありません。

.. literalinclude:: ../../examples/properties.py
   :language: python
   :start-after: # docs:example markov
   :end-before: # docs:end markov

等価と比較と複製
----------------

``==`` はプロファイル、ラベル、データを比べます。データは ``FXTensor._RTOL`` と
``FXTensor._ATOL`` による ``allclose`` です。``hash`` は生のバイト列を使うので、
許容誤差の内側で値が違っていても ``==`` が真になり、ハッシュは異なることが
あります。その組を辞書のキーとして同一視しないでください。

``copy``（および ``copy.copy``）は配列を複製します。片方への書き込みは、もう
片方を変えません。

.. literalinclude:: ../../examples/properties.py
   :language: python
   :start-after: # docs:example equality
   :end-before: # docs:end equality
