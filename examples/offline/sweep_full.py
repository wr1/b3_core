#!/usr/bin/env python3
"""Full param_sweep study including GIF export.

Homogenise + matplotlib curves + PyVista galleries + GIFs. Offline only —
the mainline is ``b3_core sweep homogenise`` (or ``make sweep``); this script
adds plots, renders, and GIFs.

    uv run python examples/offline/sweep_full.py

Needs MFEM, CalculiX (pattern CCX check), and ``uv sync --extra anim``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.sweep import homogenise, plots, render_viz, gifs
from b3_core.sweep.context import SweepContext, default_root


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root()
    ctx = SweepContext(root)
    code = homogenise.run_all_homogenise(ctx)
    code = plots.run(ctx) or code
    code = render_viz.run(ctx) or code
    code = gifs.run(ctx) or code
    return code


if __name__ == "__main__":
    raise SystemExit(main())