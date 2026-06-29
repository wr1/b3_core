#!/usr/bin/env python3
"""Sweep mold curvature on the deep-groove curved-panel RVE.

    uv run python examples/param_sweeps/sweep_curvature.py

Requires the MFEM stack (``uv sync --extra mfem``).
"""

from __future__ import annotations

from rich.console import Console

from _common import HERE, KX, load_base, print_moduli_table, run_case, tag_kx


def _state(kx: float) -> str:
    if kx > 0:
        return "open"
    if kx < 0:
        return "closed"
    return "—"


def main() -> int:
    console = Console()
    base = load_base("curved")
    rows: list[tuple[str, dict]] = []

    for kx in KX:
        out_dir = HERE / "out" / tag_kx(kx)
        out = run_case(base, {"curvature": {"kx": kx, "ky": 0.0}}, out_dir)
        radius = "flat" if kx == 0 else f"R={1.0 / abs(kx):.0f}"
        rows.append((f"kx={kx:+.4f}  {radius}  {_state(kx)}", out))

    print_moduli_table(
        console,
        title="Curvature grading of a deep-groove infused core (moduli in GPa)",
        rows=rows,
    )
    console.print(
        "[dim]Base: deep x-grooves (27 mm into 30 mm core); kx>0 opens, kx<0 closes.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())