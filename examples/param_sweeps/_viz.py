"""Shared PyVista rendering helpers for param_sweeps strips and GIF frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import (
    HERE,
    KX,
    PATTERNS,
    THICKNESSES,
    case_for_curvature,
    case_for_thickness,
    collect_sweep,
    load_base,
    load_pattern,
)
from b3_core.core.mesh import create_grooved_mesh
from b3_core.viz._deps import require_pyvista
from b3_core.viz.theme import DEFAULT_THEME

THEME = DEFAULT_THEME
IMG = HERE / "img"


def case_from_cache(prefix: str, fallback_fn) -> list[tuple[str, dict]]:
    cached = collect_sweep(prefix)
    if cached:
        return [(tag, case) for tag, case, _ in cached]
    return fallback_fn()


def mesh_from_case(case: dict):
    curv = case.get("curvature") or {}
    return create_grooved_mesh(
        thickness=case["thickness"],
        dx=case["dx"],
        dy=case["dy"],
        xcuts=case.get("xgr", []),
        ycuts=case.get("ygr", []),
        madd=tuple(case.get("madd", [0])),
        tface=(case.get("face") or {}).get("thickness", 0.0),
        kx=curv.get("kx", 0.0),
        ky=curv.get("ky", 0.0),
    )


def add_phases(plotter, mesh, title: str, *, camera: str = "iso", parallel: bool = False) -> None:
    foam = mesh.threshold(0.5, scalars="resin", invert=True)
    resin = mesh.threshold(0.5, scalars="resin")
    if foam.n_cells:
        plotter.add_mesh(
            foam, color=THEME.core_color, opacity=THEME.core_opacity, show_edges=False
        )
    if resin.n_cells:
        plotter.add_mesh(
            resin,
            color=THEME.resin_color,
            opacity=1.0,
            show_edges=True,
            edge_color=THEME.edge_color,
            line_width=THEME.edge_width,
        )
    plotter.add_text(title, font_size=10)
    if parallel:
        plotter.enable_parallel_projection()
    plotter.camera_position = camera
    if camera == "iso":
        plotter.camera.azimuth = -45
        plotter.camera.elevation = -10
    elif camera == "xz":
        plotter.camera.zoom(1.6)


def render_frame(
    case: dict,
    title: str,
    *,
    camera: str = "iso",
    parallel: bool = False,
    window_size: tuple[int, int] = (640, 520),
) -> np.ndarray:
    """Return an RGB screenshot array for one RVE case."""
    pv = require_pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(THEME.background)
    mesh = mesh_from_case(case)
    add_phases(plotter, mesh, title, camera=camera, parallel=parallel)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(img)


def render_strip(
    cases: list[tuple[str, dict]],
    path: Path,
    *,
    shape: tuple[int, int],
    window_size: tuple[int, int],
    camera: str = "iso",
    parallel: bool = False,
) -> None:
    pv = require_pyvista()
    nrow, ncol = shape
    grid = pv.Plotter(shape=shape, off_screen=True, window_size=list(window_size))
    grid.set_background(THEME.background)
    for i, (label, case) in enumerate(cases):
        mesh = mesh_from_case(case)
        grid.subplot(i // ncol, i % ncol)
        add_phases(grid, mesh, label, camera=camera, parallel=parallel)
    grid.screenshot(str(path))
    grid.close()


def thickness_cases() -> list[tuple[str, dict]]:
    base = load_base("uniaxial")
    return [(f"t = {int(t)} mm", case_for_thickness(base, t)) for t in THICKNESSES]


def curvature_cases() -> list[tuple[str, dict]]:
    base = load_base("curved")
    return [(f"kx = {kx:+.3f}", case_for_curvature(base, kx)) for kx in KX]


def pattern_cases() -> list[tuple[str, dict]]:
    return [(name, load_pattern(name)) for name in PATTERNS]


def gallery_cases() -> list[tuple[str, dict]]:
    base_u = load_base("uniaxial")
    base_c = load_base("curved")
    cases: list[tuple[str, dict]] = [
        ("gallery_uniaxial_t20", case_for_thickness(base_u, 20)),
        ("gallery_uniaxial_t50", case_for_thickness(base_u, 50)),
        ("gallery_curved_kx0", case_for_curvature(base_c, 0.0)),
        ("gallery_curved_kx_open", case_for_curvature(base_c, 0.008)),
        ("gallery_curved_kx_closed", case_for_curvature(base_c, -0.008)),
    ]
    for name in PATTERNS:
        cases.append((f"gallery_pattern_{name}", load_pattern(name)))
    return cases