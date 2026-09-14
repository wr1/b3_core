#!/usr/bin/env python3
"""Build param_sweep looping GIFs from cached homogenisation results.

Not part of the main CLI or Makefile. Needs ``uv sync --extra anim``.

    uv run python examples/offline/sweep_gifs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from b3_core.sweep import gifs
from b3_core.sweep.context import SweepContext, default_root

SWEEP_ROOT = default_root()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else SWEEP_ROOT
    return gifs.run(SweepContext(root))


if __name__ == "__main__":
    raise SystemExit(main())