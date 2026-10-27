#!/usr/bin/env python3
"""Export an interactive HTML viewer for a grooved-core case.

Offline only. Needs ``uv sync --extra interactive``.

    uv run python examples/offline/interactive_view.py [case.json] [out.html]
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.viz import GroovedCoreView

HERE = Path(__file__).parent


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE.parent / "mfem_patterns" / "two_sided.json"
    )
    out = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "out" / "core.html")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    GroovedCoreView.from_json(case).serve(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())