#!/usr/bin/env python3
"""Map a curvature distribution along a panel onto a graded-property field.

This is the "piecewise approx a given curve" step: the panel follows a curve
whose local curvature kx(s) varies along the arc length s. At each station we
build the flat RVE with grooves opened/closed to that local curvature and
homogenise it, producing the effective core tensor as a function of position —
the field a downstream shell/laminate model would sample.

    uv run python examples/curved_panel/curve_field.py

Requires the optional MFEM stack (`uv sync --extra mfem`).
"""

from __future__ import annotations

import copy
import glob
import json
import math
from pathlib import Path

from rich.console import Console
from rich.table import Table

from b3_core.core.cprop import cprop

HERE = Path(__file__).parent
BASE = json.loads((HERE / "base.json").read_text())

# Panel of arc length L draped over a mold whose curvature is zero at the ends
# and peaks mid-span (a smooth bump). Sample N stations.
L = 500.0
N = 7
KAPPA_MAX = 0.008  # 1/mm at mid-span (R ~ 125 mm)


def kappa_of_s(s: float) -> float:
    return KAPPA_MAX * math.sin(math.pi * s / L)


def run_station(idx: int, kx: float) -> dict:
    out_dir = HERE / "out" / f"station_{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    case = copy.deepcopy(BASE)
    case["curvature"] = {"kx": kx, "ky": 0.0}
    case_path = out_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2))
    try:
        return cprop(str(case_path))
    except FileExistsError:
        cached = glob.glob(str(out_dir / "run*.json"))
        return json.loads(Path(cached[0]).read_text())


def main() -> int:
    console = Console()
    table = Table(title=f"Graded core field along a curved panel (L={L:.0f} mm, moduli in GPa)")
    table.add_column("station", justify="right", style="bold")
    table.add_column("s [mm]", justify="right")
    table.add_column("kx [1/mm]", justify="right")
    table.add_column("R [mm]", justify="right")
    table.add_column("resin_vf", justify="right")
    table.add_column("Eyy", justify="right")
    table.add_column("Ezz", justify="right")

    for idx in range(N):
        s = L * idx / (N - 1)
        kx = kappa_of_s(s)
        out = run_station(idx, kx)
        radius = "flat" if abs(kx) < 1e-12 else f"{1.0 / kx:.0f}"
        table.add_row(
            str(idx),
            f"{s:.0f}",
            f"{kx:+.4f}",
            radius,
            f"{out['resin_vf']:.4f}",
            f"{out['Eyy'] / 1e9:.4f}",
            f"{out['Ezz'] / 1e9:.4f}",
        )

    console.print(table)
    console.print(
        "[dim]kx(s) = kappa_max*sin(pi*s/L): flat at the ends, most open mid-span. "
        "Eyy and resin_vf peak where the mold curvature opens the grooves widest.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
