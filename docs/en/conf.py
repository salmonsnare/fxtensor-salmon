"""English Sphinx configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sphinx_shared import configure

configure(
    globals(),
    language="en",
    other_language="ja",
    other_label="日本語",
    source_directory="docs/en/",
)
