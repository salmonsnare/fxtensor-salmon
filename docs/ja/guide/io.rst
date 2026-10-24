入出力
======

JSON
----

:meth:`~fxtensor_salmon.FXTensor.to_json` は ``profile``、``data``、``dtype`` を
持つ辞書を返します。ラベルがあるときはプロファイルもラベル付きです。
:meth:`~fxtensor_salmon.FXTensor.from_json` は ``profile`` と ``data`` から
テンソルを作り直します。``names`` があるときは、domain と codomain の
軸の名前の組です。``names`` のない JSON は、名前のないテンソルとして読みます。

:meth:`~fxtensor_salmon.FXTensor.save_to_file` はその辞書を JSON として書き、
:meth:`~fxtensor_salmon.FXTensor.load_from_file` が読み戻します。

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example json
   :end-before: # docs:end json

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example file
   :end-before: # docs:end file

ストランド
----------

:meth:`~fxtensor_salmon.FXTensor.from_strands` は、ストランドのキーから重みへの
対応を読みます。キーは文字列または整数の入れ子リストという Python リテラルで、
コードとしては評価しません。それ以外のキーは ``ValueError`` です。ラベル付きの
ストランドは点の名前を書きます。数値のストランドは **1 始まり** の添字で、
各添字をそれぞれのリストで包みます。重みは ``float`` または
``fractions.Fraction`` で、``float`` として保存します。

:meth:`~fxtensor_salmon.FXTensor.to_original_dict` は数値形式の逆向きです。
非ゼロ成分を 1 始まりの添字リストにします。重みは ``data`` の NumPy スカラーで、
プロファイルは数値のプロファイルです。

.. literalinclude:: ../../examples/serialization.py
   :language: python
   :start-after: # docs:example strands-io
   :end-before: # docs:end strands-io
