#!/usr/bin/env bash
# Build both documentation languages into docs/_build/html.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v sphinx-build >/dev/null 2>&1; then
  SPHINX=(sphinx-build)
else
  SPHINX=(python3 -m sphinx)
fi
python3 docs/check_examples.py
"${SPHINX[@]}" -W --keep-going -b html docs/en docs/_build/html/en -d docs/_build/doctrees/en
"${SPHINX[@]}" -W --keep-going -b html docs/ja docs/_build/html/ja -d docs/_build/doctrees/ja
cp docs/index.html docs/_build/html/index.html
echo "HTML: ${ROOT}/docs/_build/html/index.html"
