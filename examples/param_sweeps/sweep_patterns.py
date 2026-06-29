#!/usr/bin/env python3
"""Compare homogenised properties across groove-topology patterns.

    uv run python examples/param_sweeps/sweep_patterns.py

Requires MFEM (``uv sync --extra mfem``) and CalculiX on PATH for CCX cross-check.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from _common import HERE, MODULI, PATTERNS, load_pattern, max_rel_err, run_case, tag_pattern


def main() -> int:
    console = Console()
    table = Table(title="Groove-pattern homogenisation (moduli in GPa)")
    table.add_column("pattern", style="bold")
    table.add_column("resin_vf", justify="right")
    table.add_column("area_inc", justify="right")
    table.add_column("rho_inf", justify="right")
    for key in MODULI:
        table.add_column(key, justify="right")
    table.add_column("CCX err", justify="right")

    all_passed = True
    for name in PATTERNS:
        base = load_pattern(name)
        out_dir = HERE / "out" / tag_pattern(name)
        out = run_case(base, {}, out_dir)
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
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())