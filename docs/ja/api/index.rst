API リファレンス
=================

インストールしたパッケージが公開するのは :class:`fxtensor_salmon.FXTensor` です。
実装は ``core``、``io``、``operations``、``special``、``profile`` に分かれています。
これらは別の公開 API ではありません。メソッドは ``FXTensor`` の頁にまとめてあります。

DisCoPy の関数は :mod:`fxtensor_salmon.discopy` です。このモジュールの import には
``discopy`` extra が必要です。

docstring はソースから取り出したもので、本文は英語です。

.. toctree::
   :maxdepth: 2

   fxtensor
   discopy
