# Figures for the documentation site

Published string diagrams for the Sphinx site live in this folder. The HTML build copies it with the rest of `docs/_static/`, and the English and Japanese pages reference these files (they do not hotlink GitHub).

TikZ sources stay in [`docs/diagrams/`](../../diagrams/). Shared drawing style: `docs/diagrams/fritzstrings.sty`. Diagrams are read top to bottom.

When an issue is resolved, add a corresponding TikZ figure to the docs:

1. Add the `.tex` source under `docs/diagrams/` and rebuild the SVG with `docs/diagrams/build.sh`.
2. Copy the SVG into this folder.
3. Place it on the relevant page in both `docs/en` and `docs/ja`, with a caption in that language.

Figures currently published here, and the pages that show them:

| SVG | English | Japanese |
| --- | --- | --- |
| `composition.svg` | Quickstart, Operations | クイックスタート、操作 |
| `tensor_product.svg` | Quickstart, Operations | クイックスタート、操作 |
| `copy_associativity.svg` | Structural morphisms | 構造射 |
| `copy_counitality.svg` | Structural morphisms | 構造射 |
| `copy_commutativity.svg` | Structural morphisms | 構造射 |
| `discard_naturality.svg` | Structural morphisms | 構造射 |
| `bayesian_inversion.svg` | Conditioning and inference | 条件付けと推論 |
| `almost_surely_equal.svg` | Conditioning and inference | 条件付けと推論 |
