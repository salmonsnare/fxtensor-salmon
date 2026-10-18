"""Japanese Sphinx configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sphinx_shared import configure

configure(
    globals(),
    language="ja",
    other_language="en",
    other_label="English",
    source_directory="docs/ja/",
)
