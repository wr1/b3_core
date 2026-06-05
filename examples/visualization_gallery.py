#!/usr/bin/env python3
"""Visualization gallery for a grooved core via ``GroovedCoreView``.

    uv run python examples/visualization_gallery.py [case.json]

Writes a composite board, the directional Young's-modulus surface and a cutaway
to examples/viz_out/. Needs the MFEM backend (``uv sync``); for an interactive
HTML viewer also install the ``[interactive]`` extra and call ``.serve(...)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.viz import GroovedCoreView

HERE = Path(__file__).parent


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "mfem_patterns" / "two_sided.json")
    out = HERE / "viz_out"
    out.mkdir(exist_ok=True)

    view = GroovedCoreView.from_json(case)
    view.gallery(out / "gallery.png")
    view.modulus_surface_png(out / "modulus_surface.png")
    view.geometry_png(out / "cutaway.png", cutaway=True)
    view.slices_png(out / "slices.png")

    print(f"wrote gallery, modulus surface, cutaway and slices to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
