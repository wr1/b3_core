#!/usr/bin/env python3
"""Sweep mold curvature and tabulate how it grades the effective core tensor.

The grooves are dry-laid into a curved mold, which opens or closes them before
infusion; this sweep keeps the RVE flat but tapers the groove geometry to a range
of curvatures `kx` (1/mm) and homogenises each with the MFEM backend. Opening
(kx>0 for the base case's top-mouth grooves) admits more resin and stiffens the
core; closing pinches the grooves toward the ungrooved foam.

    uv run python examples/curved_panel/sweep.py

Requires the optional MFEM stack (`uv sync --extra mfem`).
"""

from __future__ import annotations

import copy
import glob
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from b3_core.core.cprop import cprop

HERE = Path(__file__).parent
BASE = json.loads((HERE / "base.json").read_text())
# Curvatures in 1/mm. The base groove (pitch 10, depth 10, width 3) pinches shut
# near |kx| = w/(p*d) = 0.030, so stay comfortably inside that.
KX = [-0.008, -0.004, 0.0, 0.004, 0.008]
MODULI = ["Exx", "Eyy", "Ezz", "Gxy"]


def _tag(kx: float) -> str:
    return f"kx_{kx:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")


def run_kx(kx: float) -> dict:
    out_dir = HERE / "out" / _tag(kx)
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
    table = Table(title="Curvature grading of an infused grooved core (moduli in GPa)")
    table.add_column("kx [1/mm]", justify="right", style="bold")
    table.add_column("R [mm]", justify="right")
    table.add_column("state", justify="center")
    table.add_column("resin_vf", justify="right")
    table.add_column("rho_inf", justify="right")
    for key in MODULI:
        table.add_column(key, justify="right")

    for kx in KX:
        out = run_kx(kx)
        radius = "flat" if kx == 0 else f"{1.0 / abs(kx):.0f}"
        state = "open" if kx > 0 else ("closed" if kx < 0 else "—")
        table.add_row(
            f"{kx:+.4f}",
            radius,
            state,
            f"{out['resin_vf']:.4f}",
            f"{out['rho_infused']:.0f}",
            *[f"{out[k] / 1e9:.4f}" for k in MODULI],
        )

    console.print(table)
    console.print(
        "[dim]Base: single top-mouth x-groove family; kx>0 opens (more resin), "
        "kx<0 closes (pinches toward foam).[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
