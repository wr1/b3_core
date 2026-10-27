#!/usr/bin/env python3
"""Run every MFEM groove-pattern example and tabulate the homogenised properties.

Each case JSON in this directory selects ``backend="mfem"`` with
``validate_with_ccx=true``, so a single run solves the six periodic-homogenisation
load cases with MFEM *and* cross-checks them against CalculiX. This script drives
all of them and prints one comparison row per pattern, ending with the worst-case
MFEM-vs-CCX relative error so the table doubles as a smoke test.

    uv run python examples/mfem_patterns/compare.py

Requires the optional MFEM stack (``uv sync --extra mfem``) and CalculiX on PATH.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from b3_core.core.cprop import cprop

HERE = Path(__file__).parent
# Fixed order, simplest topology first. Bare (skinless) RVEs, solved with full
# 3D periodicity — the proper unit cell for extracting an effective core tensor.
CASES = ["plain", "uniaxial", "crossed", "two_sided"]
MODULI = ["Exx", "Eyy", "Ezz", "Gxy", "Gxz", "Gyz"]


def run_case(name: str) -> dict:
    """Run one case into its own out/<name>/ dir, reusing a cached result.

    ``cprop`` writes run<HASH>.json next to the input file and refuses to
    overwrite it, so on a rerun we read the cached output back instead.
    """
    out_dir = HERE / "out" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_path = out_dir / f"{name}.json"
    case_path.write_text((HERE / f"{name}.json").read_text())
    try:
        return cprop(str(case_path))
    except FileExistsError:
        cached = glob.glob(str(out_dir / "run*.json"))
        return json.loads(Path(cached[0]).read_text())


def max_rel_err(output: dict) -> tuple[float, bool]:
    validation = output["ccx_validation"]
    errs = [p["rel_error"] for p in validation["properties"].values()]
    return (max(errs) if errs else 0.0), bool(validation["passed"])


def main() -> int:
    console = Console()
    table = Table(title="MFEM homogenisation across groove patterns (moduli in GPa)")
    table.add_column("pattern", style="bold")
    table.add_column("resin_vf", justify="right")
    table.add_column("area_inc", justify="right")
    table.add_column("rho_inf", justify="right")
    for key in MODULI:
        table.add_column(key, justify="right")
    table.add_column("CCX err", justify="right")

    all_passed = True
    for name in CASES:
        out = run_case(name)
        rel_err, passed = max_rel_err(out)
        all_passed = all_passed and passed
        table.add_row(
            name,
            f"{out['resin_vf']:.3f}",
            f"{out['area_increase']:.2f}",
            f"{out['rho_infused']:.0f}",
            *[f"{out[k] / 1e9:.3f}" for k in MODULI],
            f"{rel_err * 100:.1f}% {'✓' if passed else '✗'}",
        )

    console.print(table)
    if not all_passed:
        console.print("[bold red]A case diverged from CCX beyond rtol.[/bold red]")
        return 1
    console.print("[bold green]All patterns: MFEM agrees with CCX within tolerance.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
