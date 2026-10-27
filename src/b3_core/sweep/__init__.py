"""Parametric sweep homogenisation and publication figures (mainline)."""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.sweep.context import SweepContext, default_root

__all__ = ["run", "SweepContext", "default_root"]


def _run_viz(ctx: SweepContext) -> int:
    from b3_core.sweep import plots, render_viz

    code = plots.run(ctx)
    return render_viz.run(ctx) or code


def run(what: str, root: Path | None = None) -> int:
    """Run a parametric sweep stage (no GIF export — see ``examples/offline/``).

    *what*: ``thickness``, ``curvature``, ``patterns``, ``homogenise``,
    ``plots``, ``render``, ``viz``, ``all``.
    """
    from b3_core.sweep import homogenize as homogenize_mod
    from b3_core.sweep import plots, render_viz

    ctx = SweepContext(root or default_root())
    key = what.lower()

    dispatch = {
        "thickness": lambda: homogenize_mod.run_thickness(ctx),
        "curvature": lambda: homogenize_mod.run_curvature(ctx),
        "patterns": lambda: homogenize_mod.run_patterns(ctx),
        "homogenize": lambda: homogenize_mod.run_all_homogenise(ctx),
        "homogenise": lambda: homogenize_mod.run_all_homogenise(ctx),
        "plots": lambda: plots.run(ctx),
        "render": lambda: render_viz.run(ctx),
        "viz": lambda: _run_viz(ctx),
        "all": lambda: homogenize_mod.run_all_homogenise(ctx) or _run_viz(ctx),
    }

    try:
        handler = dispatch[key]
    except KeyError:
        valid = ", ".join(sorted(dispatch))
        print(
            f"unknown sweep stage {what!r}; expected one of: {valid}", file=sys.stderr
        )
        return 1

    return handler()
