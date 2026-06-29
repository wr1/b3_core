#!/usr/bin/env python3
"""Run all param_sweeps homogenisations and visualizations.

    uv run python examples/param_sweeps/run_all.py

Skips re-solve when cached ``run*.json`` already exists in ``out/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [
    "sweep_thickness.py",
    "sweep_curvature.py",
    "sweep_patterns.py",
    "plot_responses.py",
    "render.py",
    "make_gifs.py",
]


def main() -> int:
    code = 0
    for name in SCRIPTS:
        print(f"\n=== {name} ===")
        result = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE, check=False)
        if result.returncode != 0:
            code = result.returncode
            print(f"{name} failed with exit {result.returncode}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())