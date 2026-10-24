概要
====

:class:`~fxtensor_salmon.FXTensor` は有限集合のあいだの射です。配列の軸は
domain（入力）と codomain（出力）に分かれます。この2つの次元リストが
プロファイルです。各因子には文字列ラベルのリストを付けられます。ラベルを
付けない因子は、長さだけを持ちます。軸には、値のラベルとは別に名前を
付けられます。名前は省略できます。

マルコフ圏では、対象は有限集合、射はマルコフ核です。有限集合上の核は、
どの domain の点でも codomain 方向の和が 1 になる非負配列です。状態は
一点集合からの核で、このライブラリでは domain が空のテンソルです。codomain
の因子が複数なら同時分布です。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 要素
     - FXTensor での対応
   * - 対象
     - プロファイルの1因子。例: ``['Sunny', 'Rainy']`` または長さ ``2``
   * - 射
     - ``data`` の形状が ``domain + codomain`` である ``FXTensor``
   * - 状態 ``I → X``
     - domain が空。``profile`` は ``[[], ...]``
   * - 逐次合成
     - :meth:`~fxtensor_salmon.FXTensor.composition`
   * - モノイド積
     - :meth:`~fxtensor_salmon.FXTensor.tensor_product`
   * - 複製と破棄
     - :meth:`~fxtensor_salmon.FXTensor.copy_tensor` と :meth:`~fxtensor_salmon.FXTensor.exclamation`

このドキュメントのストリング図は **上から下** に読みます（入力が上、出力が下）。
条件付き独立性の図だけは **下から上** に読みます。
箱は射、三角形は状態、下へ枝分かれする黒丸は copy、黒丸で終わる紐は discard
です。図の描き方は Fritz, arXiv:1908.07021 に合わせ、図自体は別に描いています。
TikZ のソースはリポジトリの ``docs/diagrams/`` にあります。
このサイトが使う SVG は ``docs/_static/figures/`` のローカルコピーです。
Issue を解決したときに図を足す手順は :doc:`contributing` にあります。

``tests/test_axioms.py`` は、モノイド構造と余モノイドの等式（結合律、交換律、
copy の法則、discard の自然性）を、具体的な有限テンソルの上で確認します。

参考文献
--------

- `K. Cho and B. Jacobs, Disintegration and Bayesian Inversion via String Diagrams (arXiv:1709.00322) <https://arxiv.org/abs/1709.00322>`_
- `T. Fritz, A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics (arXiv:1908.07021) <https://arxiv.org/abs/1908.07021>`_
- `T. Fritz, T. Gonda, A. Lorenzin, P. Perrone, D. Stein, Absolute continuity, supports and idempotent splitting in categorical probability (arXiv:2308.00651) <https://arxiv.org/abs/2308.00651>`_
- `T. Fritz, T. Gonda, P. Perrone, De Finetti's theorem in categorical probability (arXiv:2105.02639) <https://arxiv.org/abs/2105.02639>`_
- `F. Vaccarino, Statistical models as natural transformations (arXiv:2609.22929) <https://arxiv.org/abs/2609.22929>`_
- `マルコフ圏 A First Look <https://m-hiyama.hatenablog.com/entry/2020/06/09/154044>`_
- `マルコフ圏におけるテンソル計算の手順とコツ <https://m-hiyama.hatenablog.com/entry/2021/04/05/153325>`_
