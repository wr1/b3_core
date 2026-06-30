#!/usr/bin/env python3
"""Scratch GroovedCoreView export for ad-hoc inspection (not committed figures).

Offline only. Writes to ``examples/offline/out/viz/``.

    uv run python examples/offline/viz_scratch.py [case.json]
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
    out = HERE / "out" / "viz"
    out.mkdir(parents=True, exist_ok=True)

    view = GroovedCoreView.from_json(case)
    view.gallery(out / "gallery.png")
    view.modulus_surface_png(out / "modulus_surface.png")
    view.geometry_png(out / "cutaway.png", cutaway=True)
    view.slices_png(out / "slices.png")
    print(f"wrote scratch figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())