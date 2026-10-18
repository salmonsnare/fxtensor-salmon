概念
====

プロファイル
------------

コンストラクタは3種類のプロファイルを受け取ります。

- **ラベル付き。** ``[[domain の群], [codomain の群]]``。各群は文字列のリストで、
  次元はその長さです。
- **数値。** ``[[domain の長さ], [codomain の長さ]]``。各要素は ``int`` です。
- **タプル。** ``(domain の長さ, codomain の長さ)``。この形はラベルなしです。

``profile`` は常に数値の長さ ``[[domain], [codomain]]`` を返します。
``labels`` は ``(domain の群, codomain の群)`` を返します。空の側や、もともと
ラベルが無い側は ``[]`` ではなく ``None`` です。

軸 ``0`` は domain の最初の因子です。domain の因子のあと、軸番号は codomain に
続きます。``get_label_index(axis, label)`` はその軸での ``label`` の位置です。
``get_index_label(axis, index)`` はその逆です。ラベルが無いとき、またはその軸に
``label`` が無いときは、どちらも ``ValueError`` です。

.. literalinclude:: ../../examples/concepts.py
   :language: python
   :start-after: # docs:example profiles
   :end-before: # docs:end profiles

データ
------

``data`` は形状 ``tuple(domain の長さ + codomain の長さ)`` の NumPy 配列です。
省略するとすべて 0 です。プロファイルと形状が合わないと ``ValueError`` です。
空のプロファイル ``[[], []]`` はモノイド単位で、データは 0 次元配列です。

.. literalinclude:: ../../examples/concepts.py
   :language: python
   :start-after: # docs:example defaults
   :end-before: # docs:end defaults

状態と核
--------

状態は domain が空です。核は domain が空ではありません。図式がモノイド単位から
始まる操作は状態に限られます。
:meth:`~fxtensor_salmon.FXTensor.conditionalization`、
:meth:`~fxtensor_salmon.FXTensor.jointification`、そして
:meth:`~fxtensor_salmon.FXTensor.bayesian_inversion` と
:meth:`~fxtensor_salmon.FXTensor.almost_surely_equal` の ``prior`` です。

等価比較は、プロファイルとラベルに加え、``FXTensor._RTOL``（``1e-05``）と
``FXTensor._ATOL``（``1e-08``）による ``numpy.allclose`` でデータを比べます。
``copy`` は配列を複製した新しいテンソルを返します。
