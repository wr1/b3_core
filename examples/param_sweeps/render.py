#!/usr/bin/env python3
"""Render geometry strips and GroovedCoreView galleries for param_sweeps cases.

    uv run python examples/param_sweeps/render.py

Uses cached ``out/*/case.json`` when available; otherwise builds from sweep bases.
"""

from __future__ import annotations

from _viz import (
    IMG,
    case_from_cache,
    curvature_cases,
    gallery_cases,
    pattern_cases,
    render_strip,
    thickness_cases,
)
from b3_core.viz import GroovedCoreView
from b3_core.viz._deps import ensure_headless


def main() -> int:
    ensure_headless()
    IMG.mkdir(parents=True, exist_ok=True)
    gallery_dir = IMG / "galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    thickness = case_from_cache("thickness_", thickness_cases)
    curvature = case_from_cache("kx_", curvature_cases)
    patterns = case_from_cache("pattern_", pattern_cases)

    render_strip(
        thickness,
        IMG / "thickness_strip.png",
        shape=(1, len(thickness)),
        window_size=(300 * len(thickness), 520),
        camera="xz",
        parallel=True,
    )
    render_strip(
        curvature,
        IMG / "curvature_strip.png",
        shape=(1, len(curvature)),
        window_size=(300 * len(curvature), 520),
        camera="xz",
        parallel=True,
    )
    render_strip(
        patterns,
        IMG / "patterns_gallery.png",
        shape=(2, 2),
        window_size=(1100, 1000),
        camera="iso",
    )

    for stem, case in gallery_cases():
        out = gallery_dir / f"{stem}.png"
        GroovedCoreView.from_dict(case).gallery(out)
        print(f"wrote {out}")

    for name in ("thickness_strip.png", "curvature_strip.png", "patterns_gallery.png"):
        print(f"wrote {IMG / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())