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
  sunny_idx = sunny_tomorrow.get_label_index(1, '晴れ')
  p_sunny = sunny_tomorrow.data[sunny_idx]  # 0.8
  ```

## 発展的な例：多次元システム（ラベル付き）

**場所**（市街地、田舎）に基づく **季節**（春、夏、その他）と **天気**（晴れ、雨）の同時確率をモデル化。

- **プロファイル**: `[[['市街地', '田舎']], [['春', '夏', 'その他'], ['晴れ', '雨']]]`
- **データ**: 形状 `(2, 3, 2)` の3次元配列。

  ```python
  location_labels = ['市街地', '田舎']
  season_labels = ['春', '夏', 'その他']
  weather_labels = ['晴れ', '雨']
  process_data = np.random.rand(2, 3, 2)
  process_data /= process_data.sum(axis=(1, 2), keepdims=True)
  process_tensor = FXTensor([[location_labels], [season_labels, weather_labels]], data=process_data)
  ```

### 主要メソッドの実践

#### `marginalization(start_B)`

```python
# P(季節 | 場所) を取得（天気を周辺化）
season_tensor = process_tensor.marginalization(start_B=2)
assert season_tensor.labels == ([['市街地', '田舎']], [['春', '夏', 'その他']])
```

#### `discard_prefix(start_B)`

```python
# P(天気 | 場所) を取得（季節を周辺化）
weather_tensor = process_tensor.discard_prefix(start_B=2)
assert weather_tensor.labels == ([['市街地', '田舎']], [['晴れ', '雨']])
```

#### `conditionalization(start_B)`

```python
# P(天気 | 場所, 季節) を計算
cond_tensor = process_tensor.conditionalization(start_B=2)
assert cond_tensor.labels == ([['市街地', '田舎'], ['春', '夏', 'その他']], [['晴れ', '雨']])
```

#### `tensor_product(other)`

```python
# 交通量（少ない、多い）を追加
traffic_labels = ['少ない', '多い']
traffic_state = FXTensor([[], [traffic_labels]], data=np.array([0.7, 0.3]))
joint_tensor = process_tensor.tensor_product(traffic_state)
assert joint_tensor.labels == (None, [['市街地', '田舎'], ['春', '夏', 'その他'], ['晴れ', '雨'], ['少ない', '多い']])
```

## 理論的背景：マルコフ圏との関係

`fxtensor-salmon` は、圏論的確率論の **マルコフ圏** に基づいて設計されています。マルコフ圏は確率的なシステムを抽象的に扱う数学的構造です。

### マルコフ圏の基本要素

- **対象**: 状態空間。`FXTensor` では、`profile` の `domain` や `codomain`（例: `[['市街地', '田舎']]` や `[[2]]`）で表現。
- **射**: マルコフ核（確率的な遷移）。`FXTensor` のインスタンスは、プロファイルとデータで射を表現。

### 圏論的操作とメソッド

1. **合成 (`composition`)**: 射 `f: A -> B` と `g: B -> C` を結合。ストリング図ではワイヤーの接続。
2. **テンソル積 (`tensor_product`)**: 独立システムの結合。ストリング図では図の並列配置。
3. **破棄 (`marginalization`, `discard_prefix`)**: 出力の一部を合計して消去。
4. **複製 (`delta_tensor`)**: 決定性の複製操作。

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
