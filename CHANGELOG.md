# Changelog

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
