#!/usr/bin/env python3
"""Social-media explainer animation (MP4 + GIF) for a grooved-core case.

Offline only — not exposed via ``make``. Needs ``uv sync --extra anim`` and MFEM.

    uv run python examples/offline/explainer.py [case.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.viz.animate import render_explainer

HERE = Path(__file__).parent
OUT = HERE / "out"


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE.parent / "mfem_patterns" / "two_sided.json"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stem = Path(case).stem
    paths = render_explainer(case, OUT / f"{stem}_explainer.mp4", gif=True)
    print("wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())