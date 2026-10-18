"""Run the scripts that the documentation includes."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "docs" / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[path.name for path in EXAMPLES])
def test_documentation_example(script: Path) -> None:
    if script.name.startswith("discopy"):
        pytest.importorskip("discopy")
    runpy.run_path(str(script), run_name="__main__")
