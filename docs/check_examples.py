"""Execute every documentation example.

The HTML pages include these scripts with ``literalinclude``. A failure means
a published snippet no longer matches the library.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent / "examples"


def main() -> int:
    scripts = sorted(EXAMPLES.glob("*.py"))
    failures = []
    for path in scripts:
        print(f"== {path.name}")
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Exception as exc:  # noqa: BLE001 - report every example, then fail.
            failures.append(f"{path.name}: {exc}")
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
    if failures:
        return 1
    print(f"ok ({len(scripts)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
