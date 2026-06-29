#!/usr/bin/env python3
"""Sweep core thickness on the canonical uniaxial grooved RVE.

Groove depth scales proportionally from the 30 mm reference (8 mm deep),
clamped so the ligament never drops below 3 mm.

    uv run python examples/param_sweeps/sweep_thickness.py

Requires the MFEM stack (``uv sync --extra mfem``).
"""

from __future__ import annotations

from rich.console import Console

from _common import (
    HERE,
    THICKNESSES,
    case_for_thickness,
    load_base,
    print_moduli_table,
    run_case,
    tag_thickness,
)


def main() -> int:
    console = Console()
    base = load_base("uniaxial")
    rows: list[tuple[str, dict]] = []

    for t in THICKNESSES:
        case = case_for_thickness(base, t)
        out_dir = HERE / "out" / tag_thickness(t)
        out = run_case(case, {}, out_dir)
        depth = case["xgr"][0][2] if case["xgr"] else 0
        rows.append((f"t={t:.0f} mm  d={depth:+.1f}", out))

    print_moduli_table(
        console,
        title="Thickness grading of a uniaxial grooved core (moduli in GPa)",
        rows=rows,
    )
    console.print(
        "[dim]Base: single x-groove family; depth scales as 8·t/30, capped at t−3 mm.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())