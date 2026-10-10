"""Percent-change, density, and kerf-state figures for a curvature grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from b3_core.units import curvature_tick
from b3_core.viz.theme import DPI, FIGSIZE_ONE_ROW, percent_vmax

_MODULI = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz")
_HEAT = ("Ex", "Ez", "Gxz")


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str | Path):
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        return [row for row in payload["rows"] if row.get("ok", True)]
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("figure input needs a grid object with rows, or a list of rows")


def _flat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if float(row.get("kx") or 0.0) == 0.0 and float(row.get("ky") or 0.0) == 0.0:
            return row
    raise ValueError("figure input has no flat row (kx = ky = 0)")


def _pct(row: dict[str, Any], flat: dict[str, Any], name: str) -> float | None:
    base = flat.get(name)
    value = row.get(name)
    if base in (None, 0) or value is None:
        return None
    return (float(value) - float(base)) / abs(float(base)) * 100.0


def _style(fig) -> None:
    from b3_core.viz.theme import DEFAULT_THEME

    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.tick_params(labelsize=8)
        ax.set_facecolor(DEFAULT_THEME.background)


def _line_panel(
    ax, rows: list[dict[str, Any]], flat: dict[str, Any], axis: str
) -> None:
    other = "ky" if axis == "kx" else "kx"
    cut = [row for row in rows if float(row.get(other) or 0.0) == 0.0]
    cut.sort(key=lambda row: float(row[axis]))
    xs = [float(row[axis]) for row in cut]
    labels = [curvature_tick(value) for value in xs]
    for name in _MODULI:
        ys = [_pct(row, flat, name) for row in cut]
        if any(value is None for value in ys):
            continue
        ax.plot(range(len(xs)), ys, marker="o", label=name)
    ax.set_xticks(range(len(xs)), labels, rotation=30, ha="right")
    ax.axhline(0.0, color="#888888", linewidth=0.6)
    ax.set_ylabel("change from flat [%]")
    ax.set_title(f"{axis} = vary, {other} = 0 (flat)")
    ax.legend(fontsize=7, ncol=2, frameon=False)


def plot_props_curvature(payload: Any):
    """Percent change of E and G along the two zero lines."""
    import matplotlib.pyplot as plt

    rows = _rows(payload)
    flat = _flat(rows)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_ONE_ROW, sharey=True)
    _line_panel(axes[0], rows, flat, "kx")
    _line_panel(axes[1], rows, flat, "ky")
    base = ", ".join(
        f"{name}={float(flat[name]):.3g}" for name in _MODULI if flat.get(name)
    )
    fig.suptitle(f"Percent change from the flat card ({base})", fontsize=9)
    _style(fig)
    fig.tight_layout()
    return fig


def plot_rho_curvature(payload: Any):
    """Infused density (solid) and resin volume fraction (dashed, right axis)."""
    import matplotlib.pyplot as plt

    rows = _rows(payload)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_ONE_ROW)
    for ax, axis in zip(axes, ("kx", "ky"), strict=True):
        other = "ky" if axis == "kx" else "kx"
        cut = [row for row in rows if float(row.get(other) or 0.0) == 0.0]
        cut.sort(key=lambda row: float(row[axis]))
        xs = list(range(len(cut)))
        labels = [curvature_tick(float(row[axis])) for row in cut]
        rho = [row.get("rho_infused") for row in cut]
        vf = [row.get("resin_vf") for row in cut]
        ax.plot(xs, rho, color="#333333", marker="o", label="rho infused")
        ax.set_ylabel("rho infused [kg/m³]")
        twin = ax.twinx()
        twin.plot(xs, vf, color="#2ca7a0", linestyle="--", marker="s", label="resin vf")
        twin.set_ylabel("resin vf [-]")
        ax.set_xticks(xs, labels, rotation=30, ha="right")
        ax.set_title(f"{axis} cut")
    _style(fig)
    fig.tight_layout()
    return fig


def plot_sweep_heatmap(payload: Any):
    """Ex, Ez, and Gxz percent change on the (kx, ky) grid. One shared scale."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _rows(payload)
    flat = _flat(rows)
    kx = sorted({float(row["kx"]) for row in rows})
    ky = sorted({float(row["ky"]) for row in rows})
    lookup = {(float(row["kx"]), float(row["ky"])): row for row in rows}
    fields = []
    for name in _HEAT:
        grid = np.full((len(ky), len(kx)), np.nan)
        for i, y_value in enumerate(ky):
            for j, x_value in enumerate(kx):
                row = lookup.get((x_value, y_value))
                if row is None:
                    continue
                value = _pct(row, flat, name)
                if value is not None:
                    grid[i, j] = value
        fields.append(grid)
    finite = [value for grid in fields for value in grid.ravel() if np.isfinite(value)]
    vmax = percent_vmax(finite)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_ONE_ROW, sharey=True)
    image = None
    for ax, grid, name in zip(axes, fields, _HEAT, strict=True):
        image = ax.imshow(
            grid,
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(
            range(len(kx)),
            [curvature_tick(value) for value in kx],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(len(ky)), [curvature_tick(value) for value in ky])
        ax.set_title(f"{name} [%]")
        ax.set_xlabel("kx")
    axes[0].set_ylabel("ky")
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label=f"±{vmax:g} %")
    fig.suptitle("Change from the flat card. Shared color scale.", fontsize=9)
    _style(fig)
    fig.tight_layout()
    return fig


def _curves(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str | Path):
        payload = json.loads(Path(payload).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("curves"):
        return list(payload["curves"])
    if isinstance(payload, dict) and payload.get("grid"):
        # Kerf-map grid: one curve per family, sampled at the 3×3 curvatures.
        families: dict[str, dict[str, Any]] = {}
        for cell in payload["grid"]:
            for kerf in cell.get("kerfs") or []:
                label = f"{kerf.get('axis')} {kerf.get('mouth')} · opens for {kerf.get('opens_for')}"
                bucket = families.setdefault(
                    label,
                    {
                        "label": label,
                        "k": [],
                        "state": [],
                        "opens_for": kerf.get("opens_for"),
                    },
                )
                k = float(cell["kx"] if kerf.get("axis") == "x" else cell["ky"])
                bucket["k"].append(k)
                bucket["state"].append(kerf.get("state"))
        return list(families.values())
    raise ValueError("wedge figure needs curves or a kerf-map grid")


def plot_kerf_ranges_wedge(payload: Any):
    """Open / flat / closed against curvature. Titles use each row's opens_for."""
    import matplotlib.pyplot as plt

    curves = _curves(payload)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_ONE_ROW)
    codes = {"closed": -1, "flat": 0, "open": 1}
    for index, curve in enumerate(curves):
        order = sorted(range(len(curve["k"])), key=lambda i: float(curve["k"][i]))
        xs = [curvature_tick(float(curve["k"][i])) for i in order]
        ys = [codes.get(str(curve["state"][i]), 0) + 0.05 * index for i in order]
        ax.plot(range(len(xs)), ys, marker="o", label=curve["label"])
        ax.set_xticks(range(len(xs)), xs, rotation=30, ha="right")
    ax.set_yticks([-1, 0, 1], ["closed", "flat", "open"])
    ax.set_ylabel("kerf state")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Kerf state from hw(z). opens_for is on the legend.")
    _style(fig)
    fig.tight_layout()
    return fig


def write_figures(payload: Any, directory: str | Path) -> dict[str, str]:
    """Write the four datasheet PNGs. Missing kerf curves skip that file."""
    import matplotlib.pyplot as plt

    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    drawers = {
        "props_curvature": plot_props_curvature,
        "sweep_heatmap": plot_sweep_heatmap,
        "rho_curvature": plot_rho_curvature,
    }
    for name, drawer in drawers.items():
        fig = drawer(payload)
        path = dest / f"{name}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        written[name] = str(path)
    try:
        fig = plot_kerf_ranges_wedge(payload)
    except ValueError:
        fig = None
    if fig is not None:
        path = dest / "kerf_ranges_wedge.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        written["kerf_ranges_wedge"] = str(path)
    return written
