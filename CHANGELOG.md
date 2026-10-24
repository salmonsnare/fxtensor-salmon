# Changelog

## 1.5.0

### English

- New: named axes (#26)
  - `FXTensor(..., names=(domain names, codomain names))`, `names`, `rename`, and `axis`. Names are variable names on wires, separate from the labels of the values on an axis. `names=None` leaves the tensor unnamed. `axis` counts domain axes first, as in `get_label_index`.
  - `FXTensor.permute(order)` reorders the codomain wires into `order`. Labels move with their wires.
  - `FXTensor.marginal(*keep)` keeps the named codomain wires, including a middle wire. For a state `p: I → A ⊗ B ⊗ C`, `p.marginal("B")` is `p ; (! ⊗ id_B ⊗ !)`.
  - `FXTensor.marginalize(drop=)` sums out the named wires and leaves the rest in their original order.
  - `FXTensor.observe(name=label)` conditions a state and returns `(posterior, evidence)`. Evidence 0 raises `ValueError`. Several outcomes at once match observing them one by one.
  - Names propagate through `composition`, `tensor_product`, `bayesian_inversion`, and `conditional` / `conditionalization`. `to_json` / `from_json` store them under `names`.
- Tests: `tests/test_names.py`.
- Docs: new "Named axes" guide page (English and Japanese) with string diagrams, and an `A → C → D` example in the READMEs.

PyPI: https://pypi.org/project/fxtensor-salmon/1.5.0/
Docs: https://salmonsnare.github.io/fxtensor-salmon/en/guide/names.html

### 日本語

- 新機能: 名前付きの軸 (#26)
  - `FXTensor(..., names=(domain の名前, codomain の名前))`、`names`、`rename`、`axis`。名前は紐の変数名で、値のラベルとは別です。`names=None` のテンソルは名前がありません。`axis` は `get_label_index` と同じく domain の軸を先に数えます。
  - `FXTensor.permute(order)`: codomain の紐を `order` の順に並べ替えます。ラベルも紐といっしょに動きます。
  - `FXTensor.marginal(*keep)`: 指定した codomain の紐だけを残します（真ん中の紐も残せます）。状態 `p: I → A ⊗ B ⊗ C` では `p.marginal("B")` は `p ; (! ⊗ id_B ⊗ !)` です。
  - `FXTensor.marginalize(drop=)`: 指定した紐を和で落とし、残りは元の順のままです。
  - `FXTensor.observe(名前=ラベル)`: 状態を条件付け、`(事後の同時状態, 証拠の確率)` を返します。証拠が 0 のときは `ValueError` です。複数の観測を一度に渡すと、1 つずつ観測したときと一致します。
  - 名前は `composition`、`tensor_product`、`bayesian_inversion`、`conditional` / `conditionalization` で伝播します。`to_json` / `from_json` は `names` として保存します。
- テスト: `tests/test_names.py` を追加。
- ドキュメント: ストリング図つきの「名前付きの軸」ガイドページ（日本語・英語）と、README に `A → C → D` の例を追加。

PyPI: https://pypi.org/project/fxtensor-salmon/1.5.0/
ドキュメント: https://salmonsnare.github.io/fxtensor-salmon/ja/guide/names.html

## 1.4.0

### English

- New: exchangeability and Kolmogorov consistency (#16)
  - `FXTensor.is_exchangeable(block=None)` checks that a state `p: I → X^{⊗n}` satisfies `p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p` at every adjacent position `k` (1-based). Swapping the `k`-th and `(k+1)`-th wires leaves `p` unchanged, and `(k-1) + 2 + (n-k-1) = n`.
  - `FXTensor.symmetrize(block=None)` averages a state over every permutation of the factors. `n > 8` warns.
  - `FXTensor.is_kolmogorov_consistent(states)` checks `p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}` for `[p_1, ..., p_N]`. Discarding the `n`-th wire of `p_n` yields `p_{n-1}`.
  - `FXTensor.iid(p, n)` is the `n`-fold tensor product of `p`.
  - `FXTensor.mixture_of_iid(prior, kernel, n)` builds `prior ; copy_Θ^{(n)} ; kernel^{⊗n}`.
- Tests: `tests/test_exchangeability.py`.
- Docs: new "Exchangeability" guide page (English and Japanese) with string diagrams, and a coin-bias example in the READMEs.

PyPI: https://pypi.org/project/fxtensor-salmon/1.4.0/
Docs: https://salmonsnare.github.io/fxtensor-salmon/en/guide/exchangeability.html

### 日本語

- 新機能: 交換可能性と Kolmogorov 整合性 (#16)
  - `FXTensor.is_exchangeable(block=None)`: 状態 `p: I → X^{⊗n}` が、隣接する位置 `k`（1 始まり）すべてで `p ; (id^{⊗(k-1)} ⊗ swap(X, X) ⊗ id^{⊗(n-k-1)}) = p` を満たすかを判定します。`k` 番目と `(k+1)` 番目の紐を入れ替えても `p` は変わらず、`(k-1) + 2 + (n-k-1) = n` です。
  - `FXTensor.symmetrize(block=None)`: 因子のあらゆる置換で状態を平均します。`n > 8` では警告します。
  - `FXTensor.is_kolmogorov_consistent(states)`: 列 `[p_1, ..., p_N]` について `p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}` を判定します。`p_n` の `n` 番目の紐を破棄すると `p_{n-1}` になります。
  - `FXTensor.iid(p, n)`: `p` の `n` 重テンソル積です。
  - `FXTensor.mixture_of_iid(prior, kernel, n)`: `prior ; copy_Θ^{(n)} ; kernel^{⊗n}` を作ります。
- テスト: `tests/test_exchangeability.py` を追加。
- ドキュメント: ストリング図つきの「交換可能性」ガイドページ（日本語・英語）と、README にコインのバイアスの例を追加。

PyPI: https://pypi.org/project/fxtensor-salmon/1.4.0/
ドキュメント: https://salmonsnare.github.io/fxtensor-salmon/ja/guide/exchangeability.html
