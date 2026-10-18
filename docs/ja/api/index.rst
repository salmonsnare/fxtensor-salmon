API リファレンス
=================

インストールしたパッケージが公開するのは :class:`fxtensor_salmon.FXTensor` です。
実装は ``core``、``io``、``operations``、``special``、``profile``、
``independence`` に分かれています。条件付き独立性は
:mod:`fxtensor_salmon.independence` にあり、そのモジュールと ``FXTensor`` の
両方にドキュメントがあります。他のモジュールは別の公開 API ではありません。

DisCoPy の関数は :mod:`fxtensor_salmon.discopy` です。このモジュールの import には
``discopy`` extra が必要です。

docstring はソースから取り出したもので、本文は英語です。

.. toctree::
   :maxdepth: 2

   fxtensor
   discopy
