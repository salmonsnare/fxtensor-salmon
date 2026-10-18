インストール
============

ライブラリは Python 3.8 以上で動き、実行時の依存は NumPy だけです。

.. code-block:: console

   pip install fxtensor-salmon

DisCoPy 連携は追加機能で、Python 3.10 以上が必要です。

.. code-block:: console

   pip install "fxtensor-salmon[discopy]"

クラスはパッケージの直下から読みます。

.. code-block:: python

   from fxtensor_salmon import FXTensor

このドキュメントのビルド
------------------------

ドキュメントは wheel には入りません。リポジトリを取得し、Python 3.10 以上で
``docs`` extra を入れます。extra は Sphinx と Furo を入れ、Python 3.10 以上では
DisCoPy も入れます。autodoc が ``fxtensor_salmon.discopy`` を import するためです。

.. code-block:: console

   pip install -e ".[docs]"
   python docs/check_examples.py
   sphinx-build -W --keep-going -b html docs/en docs/_build/html/en
   sphinx-build -W --keep-going -b html docs/ja docs/_build/html/ja
   cp docs/index.html docs/_build/html/index.html

``docs/build.sh`` は例の実行と両言語の Sphinx ビルドを行い、サイトのルートを
コピーします。同じサードパーティの一覧は ``docs/requirements.txt`` にあります。

``sphinx-build -W`` は警告をエラーにします。ガイドのコードブロックは
``docs/examples/`` のスクリプトで、``docs/check_examples.py`` がそれらを実行します。
``pytest`` も同じスクリプトを実行します（DisCoPy が無い環境では DisCoPy の例だけ
スキップします）。

公開先は https://salmonsnare.github.io/fxtensor-salmon/ です
（英語は ``en/``、日本語は ``ja/``）。GitHub Pages のソースは **GitHub Actions**
にしてください。このワークフローから、その設定自体を切り替えることはできません。
