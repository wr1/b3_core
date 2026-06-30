"""Homogenisation sweeps: thickness, curvature, groove patterns."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from b3_core.sweep.context import (
    KX,
    MODULI,
    PATTERNS,
    THICKNESSES,
    SweepContext,
    case_for_thickness,
    load_base,
    load_pattern,
    max_rel_err,
    print_moduli_table,
    run_case,
    tag_kx,
    tag_pattern,
    tag_thickness,
)


def run_thickness(ctx: SweepContext) -> int:
    console = Console()
    base = load_base(ctx, "uniaxial")
    rows: list[tuple[str, dict]] = []
    for t in THICKNESSES:
        case = case_for_thickness(base, t)
        out = run_case(case, {}, ctx.out / tag_thickness(t))
        depth = case["xgr"][0][2] if case["xgr"] else 0
        rows.append((f"t={t:.0f} mm  d={depth:+.1f}", out))
    print_moduli_table(
        console,
        title="Thickness grading of a uniaxial grooved core (moduli in GPa)",
        rows=rows,
    )
    return 0


def run_curvature(ctx: SweepContext) -> int:
    console = Console()

    def _state(kx: float) -> str:
        if kx > 0:
            return "open"
        if kx < 0:
            return "closed"
        return "—"

    base = load_base(ctx, "curved")
    rows: list[tuple[str, dict]] = []
    for kx in KX:
        out = run_case(base, {"curvature": {"kx": kx, "ky": 0.0}}, ctx.out / tag_kx(kx))
        radius = "flat" if kx == 0 else f"R={1.0 / abs(kx):.0f}"
        rows.append((f"kx={kx:+.4f}  {radius}  {_state(kx)}", out))
    print_moduli_table(
        console,
        title="Curvature grading of a deep-groove infused core (moduli in GPa)",
        rows=rows,
    )
    return 0


def run_patterns(ctx: SweepContext) -> int:
    console = Console()
    table = Table(title="Groove-pattern homogenisation (moduli in GPa)")
    table.add_column("pattern", style="bold")
    table.add_column("resin_vf", justify="right")
    table.add_column("area_inc", justify="right")
    table.add_column("rho_inf", justify="right")
    for key in MODULI:
        table.add_column(key, justify="right")
    table.add_column("CCX err", justify="right")

    code = 0
    for name in PATTERNS:
        base = load_pattern(ctx, name)
        out = run_case(base, {}, ctx.out / tag_pattern(name))
        rel_err, passed = max_rel_err(out)
        if not passed:
            code = 1
        table.add_row(
            name,
            f"{out['resin_vf']:.3f}",
            f"{out['area_increase']:.2f}",
            f"{out['rho_infused']:.0f}",
            *[f"{out[k] / 1e9:.3f}" for k in MODULI],
            f"{rel_err * 100:.1f}% {'✓' if passed else '✗'}",
        )
    console.print(table)
    return code


def run_all_homogenise(ctx: SweepContext) -> int:
    code = run_thickness(ctx)
    code = run_curvature(ctx) or code
    return run_patterns(ctx) or code