# FXTensor

FXTensorは、テンソルベースの計算を行うためのPythonライブラリであり、特に圏論に基づいた確率的なシステムやプロセスのモデル化に適しています。NumPyを活用して効率的な数値計算を実現しています。ラベル付きインデックスを主にサポートしつつ、ラベルなしの数値インデックスも扱える柔軟な設計が特徴です。

## 中核となる概念

FXTensorのテンソルは、`profile`（次元情報）と`data`（値）によって定義されます。オプションとして、文字列ラベルを付与することで、テンソルをより直感的で可読性の高いものにできます。

- **プロファイル (Profile)**: `[domain, codomain]` の形式で、入力（domain）と出力（codomain）の次元を指定します。ラベル付きの場合、例えば `[[['a', 'b']], [['x', 'y', 'z']]]` は、2つの入力ラベルと3つの出力ラベルを持つ2x3行列を表します。ラベルなしの場合、`[[2], [3]]` のように数値で次元を指定します。
- **ラベル (Labels, オプション)**: 各次元に文字列ラベルを付与可能で、システムの意味を明確にします。例えば、入力軸に `['a', 'b']`、出力軸に `['x', 'y', 'z']` を設定できます。ラベルなしの場合は、`labels` は `None` になります。
- **データ (Data)**: テンソルの値を保持するNumPy配列。形状はプロファイルに基づき、domainとcodomainの次元の合計（`len(domain) + len(codomain)`）と一致します。

## 使用例

### 基本的な例：ラベル付きテンソル

```python
import numpy as np
from fxtensor_salmon import FXTensor

# 文字列ラベルを使用して2x3行列を作成
profile = [[['a', 'b']], [['x', 'y', 'z']]]
data = np.array([
    [0.1, 0.2, 0.7],  # a -> x, y, z
    [0.3, 0.3, 0.4]   # b -> x, y, z
])
tensor = FXTensor(profile, data=data)

# ラベルを使用して要素にアクセス
assert tensor.get_label_index(0, 'a') == 0  # 入力軸でラベル 'a' のインデックス
assert tensor.get_index_label(1, 2) == 'z'  # 出力軸でインデックス2のラベル
```

### ラベルなしテンソル

```python
# 数値インデックスを使用して2x3行列を作成
profile = [[2], [3]]
data = np.array([
    [0.1, 0.2, 0.7],
    [0.3, 0.3, 0.4]
])
tensor = FXTensor(profile, data=data)
assert tensor.labels == (None, None)  # ラベルなし
```

### ストランドからテンソルを作成

```python
# ラベル付きストランドからテンソルを作成
profile = [[['a', 'b']], [['x', 'y', 'z']]]
strands = {
    "[[['a']], [['x']]]": 0.1,
    "[[['a']], [['y']]]": 0.2,
    "[[['a']], [['z']]]": 0.7,
    "[[['b']], [['x']]]": 0.3,
    "[[['b']], [['y']]]": 0.3,
    "[[['b']], [['z']]]": 0.4
}
tensor = FXTensor.from_strands(profile, strands)
assert tensor.labels == ([['a', 'b']], [['x', 'y', 'z']])
```

### ラベル付きテンソルの合成

```python
# P(Y|X) ここで X={a,b}, Y={x,y}
tensor1 = FXTensor(
    [[['a', 'b']], [['x', 'y']]],
    data=np.array([
        [0.2, 0.8],  # a -> x, y
        [0.6, 0.4]   # b -> x, y
    ])
)

# P(Z|Y) ここで Y={x,y}, Z={p,q}
tensor2 = FXTensor(
    [[['x', 'y']], [['p', 'q']]],
    data=np.array([
        [0.3, 0.7],  # x -> p, q
        [0.9, 0.1]   # y -> p, q
    ])
)

# 合成: P(Z|X) = P(Y|X) ; P(Z|Y)
result = tensor1.composition(tensor2)
assert result.labels == ([['a', 'b']], [['p', 'q']])
assert np.allclose(result.data, [
    [0.2*0.3 + 0.8*0.9, 0.2*0.7 + 0.8*0.1],  # a -> p, q
    [0.6*0.3 + 0.4*0.9, 0.6*0.7 + 0.4*0.1],  # b -> p, q
])
```

### ラベル付きテンソル積

```python
# P(X) ここで X={a,b}
tensor1 = FXTensor(
    [[], [['a', 'b']]],
    data=np.array([0.3, 0.7])
)

# P(Y) ここで Y={x,y,z}
tensor2 = FXTensor(
    [[], [['x', 'y', 'z']]],
    data=np.array([0.2, 0.3, 0.5])
)

# テンソル積: P(X,Y) = P(X) ⊗ P(Y)
result = tensor1.tensor_product(tensor2)
assert result.labels == (None, [['a', 'b'], ['x', 'y', 'z']])
assert np.allclose(result.data, np.outer(
    np.array([0.3, 0.7]),
    np.array([0.2, 0.3, 0.5]),
))
```

## 単純な例：天気予報（ラベル付き）

天候が「晴れ」または「雨」のシステムをモデル化します。

- **状態テンソル**: 現在の天気の確率をラベル付きで表現。今日が晴れなら、状態は `[1, 0]`。

  ```python
  weather_states = ['晴れ', '雨']
  sunny_today = FXTensor([[], [weather_states]], data=np.array([1, 0]))
  ```

- **プロセス・テンソル**: 天気予報をラベル付きのマルコフ核として表現。

  ```python
  forecast_matrix = np.array([
      [0.8, 0.2],  # 晴れ -> 晴れ: 0.8, 雨: 0.2
      [0.4, 0.6]   # 雨 -> 晴れ: 0.4, 雨: 0.6
  ])
  forecast_tensor = FXTensor([[weather_states], [weather_states]], data=forecast_matrix)
  ```

- **合成**: 今日の状態と予報を合成し、明日の天気を予測。

  ```python
  sunny_tomorrow = sunny_today.composition(forecast_tensor)
  sunny_idx = sunny_tomorrow.get_label_index(0, '晴れ')
  p_sunny = sunny_tomorrow.data[sunny_idx]  # 0.8
  ```

## 発展的な例：多次元システム（ラベル付き）

**場所**（市街地、田舎）を条件とする **季節**（春、夏、その他）と **天気**（晴れ、雨）をモデル化します。

`conditionalization` と `jointification` は **状態（空の domain）専用** です。`marginalization` は核にも使えます。

### 核: P(季節, 天気 | 場所)

プロファイル `[[['市街地', '田舎']], [['春', '夏', 'その他'], ['晴れ', '雨']]]`、データの形状は `(2, 3, 2)`。各場所のブロックの和は 1 です。

```python
location_labels = ['市街地', '田舎']
season_labels = ['春', '夏', 'その他']
weather_labels = ['晴れ', '雨']
process_data = np.array([
    [[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]],  # 市街地
    [[0.1, 0.2], [0.2, 0.2], [0.1, 0.2]],  # 田舎
])
process_tensor = FXTensor(
    [[location_labels], [season_labels, weather_labels]],
    data=process_data,
)

# 天気を周辺化して P(季節 | 場所)
season_tensor = process_tensor.marginalization(start_B=2)
assert season_tensor.labels == ([['市街地', '田舎']], [['春', '夏', 'その他']])
assert np.allclose(season_tensor.data, [
    [0.3, 0.4, 0.3],
    [0.3, 0.4, 0.3],
])
```

### 同時状態: P(場所, 季節, 天気)

状態は domain が空です。最後の軸を `conditionalization(3)` で割ると `P(天気 | 場所, 季節)` になります。

```python
joint_data = np.array([
    [[0.08, 0.04], [0.12, 0.04], [0.08, 0.04]],  # 市街地、合計 0.4
    [[0.06, 0.12], [0.12, 0.12], [0.06, 0.12]],  # 田舎、合計 0.6
])
joint = FXTensor(
    [[], [location_labels, season_labels, weather_labels]],
    data=joint_data,
)

cond_tensor = joint.conditionalization(concat_start_index=3)
assert cond_tensor.labels == (
    [['市街地', '田舎'], ['春', '夏', 'その他']],
    [['晴れ', '雨']],
)
assert cond_tensor.is_markov()
assert np.allclose(cond_tensor.data, [
    [[2/3, 1/3], [0.75, 0.25], [2/3, 1/3]],
    [[1/3, 2/3], [0.50, 0.50], [1/3, 2/3]],
])
```

### 2つの状態の同時化

引数はどちらも状態である必要があります。空の domain のラベルは `None` になります。

```python
location_state = FXTensor([[], [location_labels]], data=np.array([0.6, 0.4]))
traffic_labels = ['少ない', '多い']
traffic_state = FXTensor([[], [traffic_labels]], data=np.array([0.7, 0.3]))
joint_state = location_state.jointification(traffic_state)
assert joint_state.labels == (None, [['市街地', '田舎'], ['少ない', '多い']])
assert np.allclose(joint_state.data, [
    [0.42, 0.18],
    [0.28, 0.12],
])
```

### Key Method Applications

#### `from_json(json_data)`

JSONデータからテンソルを作成します。プロファイルとデータをJSON形式で読み込み、FXTensorインスタンスを返します。

```python
json_data = {
    "profile": [[['a', 'b']], [['x', 'y']]],
    "data": [[0.2, 0.8], [0.6, 0.4]]
}
tensor = FXTensor.from_json(json_data)
assert tensor.labels == ([['a', 'b']], [['x', 'y']])
```

#### `from_strands(profile, strands)`

ストランド（文字列表現のスパースデータ）からテンソルを作成します。非ゼロ要素を効率的に指定可能。

```python
profile = [[['a', 'b']], [['x', 'y', 'z']]]
strands = {
    "[[['a']], [['x']]]": 0.1,
    "[[['a']], [['y']]]": 0.2,
    "[[['a']], [['z']]]": 0.7,
    "[[['b']], [['x']]]": 0.3,
    "[[['b']], [['y']]]": 0.3,
    "[[['b']], [['z']]]": 0.4
}
tensor = FXTensor.from_strands(profile, strands)
assert tensor.labels == ([['a', 'b']], [['x', 'y', 'z']])
```

#### `identity_tensor(list_x)`

指定された次元に対する単位テンソル（アイデンティティ）を作成します。ラベル付きまたは数値次元に対応。

```python
labels = [['a', 'b']]
id_tensor = FXTensor.identity_tensor(labels)
assert id_tensor.labels == ([['a', 'b']], [['a', 'b']])
```

#### `unit_tensor(dims)`

指定された次元に対する単位状態テンソル（すべて1のベクトル）を作成します。

```python
dims = [2, 3]
unit = FXTensor.unit_tensor(dims)
assert unit.profile == [[], dims]
assert np.all(unit.data == 1)
```

#### `delta_tensor(dims)`

指定された次元に対するデルタテンソル（アイデンティティ行列）を作成します。複製操作に使用。

```python
dims = [2]
delta = FXTensor.delta_tensor(dims)
assert delta.profile == [[dims], [dims]]
```

## 理論的背景：マルコフ圏との関係

`fxtensor-salmon` は、圏論的確率論の **マルコフ圏** に基づいて設計されています。マルコフ圏は確率的なシステムを抽象的に扱う数学的構造です。

### マルコフ圏の基本要素

- **対象**: 状態空間。`FXTensor` では、`profile` の `domain` や `codomain`（例: `[['市街地', '田舎']]` や `[[2]]`）で表現。
- **射**: マルコフ核（確率的な遷移）。`FXTensor` のインスタンスは、プロファイルとデータで射を表現。

### マルコフ圏の操作

| メソッド | 役割 | いつ使うか |
|---|---|---|
| `composition` | 逐次合成 `A→B` のあと `B→C` | プロセスを直列につなぐ。状態に核を適用する |
| `tensor_product` | モノイド積 | 独立な系を並列に置く |
| `marginalization` | 出力の接尾を破棄 | 不要な出力軸を和で落とす |
| `conditionalization` | 同時状態 → 核 | 同時分布を条件付きに割る（状態専用） |
| `partial_composition` | 一部のワイヤーだけ合成 | 出力の接頭は残し、接尾だけ次の核へつなぐ |
| `jointification` | 2つの状態の同時化 | 独立な2つの状態から同時状態を作る（状態専用） |
| `exclamation` | 破棄射 `X → I` | すべて1の破棄テンソルを作る |
| `delta_tensor` | 複製射 | 決定性のコピー / 対角 |

#### `composition` — 直列の接続

`f` の codomain と `g` の domain が一致するときに使います。数値は上のラベル付き例と同じで、`P(Z|X) = P(Y|X) ; P(Z|Y)` です。

```python
result = tensor1.composition(tensor2)
assert np.allclose(result.data, [[0.78, 0.22], [0.54, 0.46]])
```

#### `tensor_product` — 独立な系の並列

2つの射（または2つの状態）を結合せずに並べるときに使います。

```python
px = FXTensor([[], [['a', 'b']]], data=np.array([0.3, 0.7]))
py = FXTensor([[], [['x', 'y', 'z']]], data=np.array([0.2, 0.3, 0.5]))
pxy = px.tensor_product(py)
assert pxy.labels == (None, [['a', 'b'], ['x', 'y', 'z']])
assert np.allclose(pxy.data, [[0.06, 0.09, 0.15], [0.14, 0.21, 0.35]])
```

#### `marginalization` — 出力の接尾を落とす

`start_B`（1始まり）は、和をとって捨てる codomain 軸の先頭です。状態 `[[], [2, 3]]` で `start_B=2` なら先頭軸だけ残ります。

```python
state = FXTensor([[], [2, 3]], data=np.array([
    [0.1, 0.2, 0.3],
    [0.15, 0.05, 0.2],
]))
marginal = state.marginalization(2)
assert marginal.profile == [[], [2]]
assert np.allclose(marginal.data, [0.6, 0.4])
```

#### `conditionalization` — 同時状態から核へ

状態にだけ使えます。`concat_start_index`（1始まり）は新しい codomain の先頭軸です。和が0のスライスは0のままです。

```python
joint_state = FXTensor([[], [2, 2]], data=np.array([[0.1, 0.2], [0.0, 0.0]]))
kernel = joint_state.conditionalization(2)
assert kernel.profile == [[2], [2]]
assert kernel.is_markov()
assert np.allclose(kernel.data, [[1/3, 2/3], [0.0, 0.0]])
```

#### `partial_composition` — 末尾の出力だけ合成

`f: A → B ⊗ C` と `g: C → D` に対し、`f.partial_composition(g, 2)` は `B` を残して `C` だけ合成し、`A → B ⊗ D` を返します。

```python
f = FXTensor([[2], [2, 2]], data=np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
]))
g = FXTensor([[2], [2]], data=np.array([
    [0.2, 0.8],
    [0.6, 0.4],
]))
partial = f.partial_composition(g, 2)
assert partial.profile == [[2], [2, 2]]
assert np.allclose(partial.data, [
    [[0.2, 0.8], [0.6, 0.4]],
    [[0.6, 0.4], [0.2, 0.8]],
])
```

`concat_start_index=1` は codomain 全体との合成です（テストがカバーしている場合）。

#### `jointification` — 2つの状態を同時化

状態同士のテンソル積と同じ数値になります。どちらも domain が空である必要があります。

```python
px = FXTensor([[], [['a', 'b']]], data=np.array([0.3, 0.7]))
py = FXTensor([[], [['x', 'y', 'z']]], data=np.array([0.2, 0.3, 0.5]))
joint_xy = px.jointification(py)
assert joint_xy.labels == (None, [['a', 'b'], ['x', 'y', 'z']])
assert np.allclose(joint_xy.data, [[0.06, 0.09, 0.15], [0.14, 0.21, 0.35]])
```

### 確率的性質

- `is_markov()`: 出力の合計が1（または0）か検証。
- ラベル付きテンソルでは、`get_label_index` と `get_index_label` で確率分布の意味を直感的に解釈可能。

## テスト

テストは `pytest` を使用し、`tests/test_fxtensor.py` に実装されています。

```bash
pytest
```

## 参考文献
- [1] [檜山正幸のキマイラ飼育記 (はてなBlog), マルコフ圏 A First Look -- 圏論的確率論の最良の定式化](https://m-hiyama.hatenablog.com/entry/2020/06/09/154044)
- [2] [檜山正幸のキマイラ飼育記 (はてなBlog), マルコフ圏におけるテンソル計算の手順とコツ](https://m-hiyama.hatenablog.com/entry/2021/04/05/153325)
