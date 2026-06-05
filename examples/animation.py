#!/usr/bin/env python3
"""Render the grooved-core social-media explainer animation (MP4 + GIF).

    uv run python examples/animation.py [case.json]

Walks through geometry -> resin infusion -> FE mesh -> orthogonal slices ->
the curvature sim -> homogenised properties. Needs the [anim] extra
(`uv sync --extra anim`) and the MFEM backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.viz.animate import render_explainer

HERE = Path(__file__).parent


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "mfem_patterns" / "two_sided.json")
    out = HERE / "anim_out"
    out.mkdir(exist_ok=True)
    paths = render_explainer(case, out / "explainer.mp4", gif=True)
    print("wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
