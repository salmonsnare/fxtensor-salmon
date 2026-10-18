クイックスタート
==================

ラベル付きの核は、文字列リストのプロファイルと配列です。``profile`` は数値の
形状、``labels`` は文字列を返します。ラベルの検索に使う軸番号は NumPy の軸で、
domain の軸が先、そのあとが codomain です。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example labeled-tensor
   :end-before: # docs:end labeled-tensor

同じ配列をラベルなしでも持てます。タプル ``(domain, codomain)`` も受け取れ、
その場合は常にラベルなしです。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example unlabeled-tensor
   :end-before: # docs:end unlabeled-tensor

``from_strands`` は辞書から成分を埋めます。ラベル付きプロファイルでは、キーは
そのラベルの入れ子リストです。無いキーは 0 のままです。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example strands
   :end-before: # docs:end strands

合成
----

``composition`` は直列の接続です。受け手の codomain と引数の domain は一致して
いる必要があります。結果のラベルは、受け手の domain ラベルと引数の codomain
ラベルです。両方がラベル付きのときだけ残ります。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example composition
   :end-before: # docs:end composition

.. figure:: ../_static/figures/composition.svg
   :alt: 上から下へ読む、2つの射の逐次合成
   :width: 140px
   :align: center

   逐次合成 ``f ; g``。

テンソル積
----------

``tensor_product`` は2つの射を並べます。状態では domain が空のままなので、
そのラベル群は ``None`` です。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example tensor-product
   :end-before: # docs:end tensor-product

.. figure:: ../_static/figures/tensor_product.svg
   :alt: 2つの射を並べたテンソル積
   :width: 180px
   :align: center

   テンソル積 ``f ⊗ g``。

天気
----

ラベルは任意の文字列です。同じ予報を英語でも日本語でも名付けられます。変わるのは
プロファイルに保存する文字列だけです。

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example weather-en
   :end-before: # docs:end weather-en

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # docs:example weather-ja
   :end-before: # docs:end weather-ja

:doc:`ユーザーガイド <guide/index>` では、条件付け、ベイズ反転、複製と破棄、
DisCoPy 連携を扱います。
