条件付き独立性
================

:meth:`~fxtensor_salmon.FXTensor.is_conditionally_independent` は、状態
``I → X ⊗ Y ⊗ Z`` または核 ``A → X ⊗ Y ⊗ Z`` が ``X ⊥ Y | Z`` を満たすかを
返します。``split_x`` と ``split_y`` は、``conditional`` と同じく codomain に
沿った **1 始まり** の位置です。``split_x`` より前が ``X``、``split_x`` から
``split_y`` の直前までが ``Y``、残りが ``Z`` です。どのブロックも空で構いません。

判定は ``marginalization``、``conditional``、``copy_tensor`` で
``p(x|z) p(y|z) p(z)`` を組み直し、``numpy.allclose`` で比べます。``atol`` と
``rtol`` の既定値は ``FXTensor._ATOL`` と ``FXTensor._RTOL`` です。

:meth:`~fxtensor_salmon.FXTensor.is_independent` は ``Z`` が空の場合です。
``split`` は ``Y`` の先頭因子です。

.. figure:: ../../_static/figures/conditional_independence.svg
   :alt: copy–discard 等式としての条件付き独立性と conditional product
   :width: 100%
   :align: center

   図は下から上に読む（下が域、上が余域）。同時状態の紐は、図の通り
   ``X ⊗ Z ⊗ Y`` の順である。左は ``X ⊥ Y | Z`` の copy–discard 等式、右は
   ``conditional_product`` の構成
   ``p_Z ; copy_Z ; (p(x|z) ⊗ id_Z ⊗ q(y|z))``、下は ``Z`` が自明なときの
   周辺の積である。API が返す余域の順は ``X ⊗ Y ⊗ Z``（``Z`` が最後）であり、
   図に描かれた ``X ⊗ Z ⊗ Y`` とは並びが異なる。

条件付き積
----------

:meth:`~fxtensor_salmon.FXTensor.conditional_product` は、同時状態
``p_{XZ}: I → X ⊗ Z`` と ``q_{YZ}: I → Y ⊗ Z``（あるいは domain を共有する核）
から、``Z`` を条件として条件付き独立で、それらの周辺を持つ ``X ⊗ Y ⊗ Z`` 上の
同時分布を返します。``on`` は、両方の codomain で ``Z`` が始まる 1 始まりの
位置です。``X`` と ``Y`` の因子数が違うときは組 ``(on_p, on_q)`` を渡します。
``Z`` の因子サイズが違う、または ``Z`` の周辺が ``atol`` / ``rtol`` の範囲で
一致しないときは ``ValueError`` です。範囲外の split も ``ValueError`` です。

構成は ``p_Z ; copy_Z ; (p(x|z) ⊗ id_Z ⊗ q(y|z))`` で、図の右パネル
（紐は ``X ⊗ Z ⊗ Y``）に対応します。メソッドは続いて ``Z`` を ``Y`` の後ろへ
移し、返す余域を ``X ⊗ Y ⊗ Z`` にします。

予報、天気、傘
--------------

予報が、出かける先の天気と、傘を持っていくかどうかの両方を決めます。予報を
条件にすれば、天気と傘は条件付き独立です。天気と傘だけの周辺では両者はまだ
連動するので、そこでは ``is_independent`` は偽です。

.. literalinclude:: ../../examples/independence.py
   :language: python
   :start-after: # docs:example weather
   :end-before: # docs:end weather

domain を共有する核
-------------------

両方の引数が同じ domain を持つ核 ``A → ·`` でも、同じ構成が使えます。``on``
は、どちらの codomain でも末尾の ``Z`` 因子を指します。

.. literalinclude:: ../../examples/independence.py
   :language: python
   :start-after: # docs:example kernel
   :end-before: # docs:end kernel
