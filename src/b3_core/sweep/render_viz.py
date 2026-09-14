"""PyVista rendering helpers and gallery export for parametric sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_core.core.mesh import create_grooved_mesh
from b3_core.sweep.context import (
    KX,
    PATTERNS,
    THICKNESSES,
    SweepContext,
    case_for_curvature,
    case_for_thickness,
    collect_sweep,
    load_base,
    load_pattern,
)
from b3_core.viz import GroovedCoreView
from b3_core.viz._deps import ensure_headless, require_pyvista
from b3_core.viz.theme import DEFAULT_THEME

THEME = DEFAULT_THEME


def case_from_cache(
    ctx: SweepContext, prefix: str, fallback_fn
) -> list[tuple[str, dict]]:
    cached = collect_sweep(ctx, prefix)
    if cached:
        return [(tag, case) for tag, case, _ in cached]
    return fallback_fn(ctx)


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


def add_phases(
    plotter, mesh, title: str, *, camera: str = "iso", parallel: bool = False
) -> None:
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


def thickness_cases(ctx: SweepContext) -> list[tuple[str, dict]]:
    base = load_base(ctx, "uniaxial")
    return [(f"t = {int(t)} mm", case_for_thickness(base, t)) for t in THICKNESSES]


def curvature_cases(ctx: SweepContext) -> list[tuple[str, dict]]:
    base = load_base(ctx, "curved")
    return [(f"kx = {kx:+.3f}", case_for_curvature(base, kx)) for kx in KX]


def pattern_cases(ctx: SweepContext) -> list[tuple[str, dict]]:
    return [(name, load_pattern(ctx, name)) for name in PATTERNS]


def gallery_cases(ctx: SweepContext) -> list[tuple[str, dict]]:
    base_u = load_base(ctx, "uniaxial")
    base_c = load_base(ctx, "curved")
    cases: list[tuple[str, dict]] = [
        ("gallery_uniaxial_t20", case_for_thickness(base_u, 20)),
        ("gallery_uniaxial_t50", case_for_thickness(base_u, 50)),
        ("gallery_curved_kx0", case_for_curvature(base_c, 0.0)),
        ("gallery_curved_kx_open", case_for_curvature(base_c, 0.008)),
        ("gallery_curved_kx_closed", case_for_curvature(base_c, -0.008)),
    ]
    for name in PATTERNS:
        cases.append((f"gallery_pattern_{name}", load_pattern(ctx, name)))
    return cases


def run(ctx: SweepContext) -> int:
    ensure_headless()
    ctx.img.mkdir(parents=True, exist_ok=True)
    gallery_dir = ctx.img / "galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    thickness = case_from_cache(ctx, "thickness_", thickness_cases)
    curvature = case_from_cache(ctx, "kx_", curvature_cases)
    patterns = case_from_cache(ctx, "pattern_", pattern_cases)

    render_strip(
        thickness,
        ctx.img / "thickness_strip.png",
        shape=(1, len(thickness)),
        window_size=(300 * len(thickness), 520),
        camera="xz",
        parallel=True,
    )
    render_strip(
        curvature,
        ctx.img / "curvature_strip.png",
        shape=(1, len(curvature)),
        window_size=(300 * len(curvature), 520),
        camera="xz",
        parallel=True,
    )
    render_strip(
        patterns,
        ctx.img / "patterns_gallery.png",
        shape=(2, 2),
        window_size=(1100, 1000),
        camera="iso",
    )

    for stem, case in gallery_cases(ctx):
        out = gallery_dir / f"{stem}.png"
        GroovedCoreView.from_dict(case).gallery(out)
        print(f"wrote {out}")

    for name in ("thickness_strip.png", "curvature_strip.png", "patterns_gallery.png"):
        print(f"wrote {ctx.img / name}")
    return 0
