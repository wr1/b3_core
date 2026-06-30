#!/usr/bin/env python3

from pathlib import Path

from treeparse import argument, cli, command, group, option
from treeparse.models.chain import chain

from b3_core.core.cprop import cprop

_CASE_ARG = argument(
    name="path",
    arg_type=str,
    help="Case file (.yaml, .yml, or .json).",
)
_SWEEP_ROOT_OPT = option(
    flags=["--root"],
    arg_type=str,
    default="",
    inherit=True,
    help="Study root (default: examples/param_sweeps/).",
)
_SWEEP_ROOT_STATE: list[str] = [""]


def cmd_run(path: str):
    cprop(path)


def cmd_skill(stdout: bool):
    from b3_core.skill import read_skill, skill_path

    if stdout:
        print(read_skill(), end="")
    else:
        print(skill_path())


def _sweep_context(root: str = ""):
    from b3_core.sweep.context import SweepContext, default_root

    r = root or _SWEEP_ROOT_STATE[0]
    return SweepContext(Path(r) if r else default_root())


def _sweep_exit(code: int) -> None:
    import sys

    sys.exit(code)


def _bind_sweep_root(root: str) -> None:
    _SWEEP_ROOT_STATE[0] = root


def cmd_sweep_thickness(root: str):
    from b3_core.sweep import homogenise

    _bind_sweep_root(root)
    _sweep_exit(homogenise.run_thickness(_sweep_context(root)))


def cmd_sweep_curvature(root: str):
    from b3_core.sweep import homogenise

    _bind_sweep_root(root)
    _sweep_exit(homogenise.run_curvature(_sweep_context(root)))


def cmd_sweep_curvature_chained():
    from b3_core.sweep import homogenise

    _sweep_exit(homogenise.run_curvature(_sweep_context()))


def cmd_sweep_patterns(root: str):
    from b3_core.sweep import homogenise

    _bind_sweep_root(root)
    _sweep_exit(homogenise.run_patterns(_sweep_context(root)))


def cmd_sweep_patterns_chained():
    from b3_core.sweep import homogenise

    _sweep_exit(homogenise.run_patterns(_sweep_context()))


def _sweep_subgroup() -> group:
    thickness = command(
        name="thickness",
        help="Homogenise thickness sweep (20–50 mm).",
        callback=cmd_sweep_thickness,
        options=[_SWEEP_ROOT_OPT],
        sort_key=0,
    )
    curvature = command(
        name="curvature",
        help="Homogenise curvature sweep (kx).",
        callback=cmd_sweep_curvature,
        options=[_SWEEP_ROOT_OPT],
        sort_key=1,
    )
    patterns = command(
        name="patterns",
        help="Homogenise groove-pattern sweep.",
        callback=cmd_sweep_patterns,
        options=[_SWEEP_ROOT_OPT],
        sort_key=2,
    )
    curvature_chained = command(
        name="curvature",
        help="Homogenise curvature sweep (kx).",
        callback=cmd_sweep_curvature_chained,
        sort_key=1,
    )
    patterns_chained = command(
        name="patterns",
        help="Homogenise groove-pattern sweep.",
        callback=cmd_sweep_patterns_chained,
        sort_key=2,
    )
    return group(
        name="sweep",
        help="Parametric homogenisation studies.",
        options=[_SWEEP_ROOT_OPT],
        commands=[
            thickness,
            curvature,
            patterns,
            chain(
                name="homogenise",
                help="thickness ➜ curvature ➜ patterns",
                chained_commands=[thickness, curvature_chained, patterns_chained],
                sort_key=3,
            ),
        ],
    )


def cmd_viz_view(path: str, what: str, output: str, serve: str, warp: float):
    from b3_core.viz import GroovedCoreView

    view = GroovedCoreView.from_json(path)
    stem = Path(path).stem
    if serve:
        view.serve(serve)
        print(f"Wrote interactive viewer {serve}")
        return

    single = {
        "geometry": lambda p: view.geometry_png(p, cutaway=False),
        "slices": view.slices_png,
        "deformation": lambda p: view.deformation_png(p, warp=warp),
        "modulus": view.modulus_surface_png,
        "polar": view.modulus_polar_png,
        "heatmap": view.stiffness_heatmap_png,
    }
    if what == "gallery":
        out = output or f"{stem}_gallery.png"
        view.gallery(out)
        print(f"Wrote {out}")
    elif what == "all":
        out = Path(output) if output else Path(f"{stem}_viz")
        out.mkdir(parents=True, exist_ok=True)
        for name, fn in single.items():
            fn(out / f"{name}.png")
        view.gallery(out / "gallery.png")
        print(f"Wrote {len(single) + 1} figures to {out}")
    else:
        out = output or f"{stem}_{what}.png"
        single[what](out)
        print(f"Wrote {out}")


def cmd_viz_halo(path: str, output: str, sharp: str):
    import json as _json

    from b3_core.viz.halo import render_halo_figures

    scored = _json.loads(Path(path).read_text())
    out = Path(output) if output else Path(path).parent / "img"
    sharp_inp = _json.loads(Path(sharp).read_text()) if sharp else None
    if sharp_inp is None:
        sibling = Path(path).with_name("diab_gs30.json")
        if sibling.is_file() and path.endswith("_scored.json"):
            sharp_inp = _json.loads(sibling.read_text())
    paths = render_halo_figures(scored, out, sharp_inp=sharp_inp)
    print("Wrote " + ", ".join(str(p) for p in paths))


def cmd_viz_datasheet(path: str, output: str, png: str):
    from b3_core.datasheet import generate

    out_pdf = output or str(Path(path).with_suffix(".pdf"))
    generate(path, out_pdf, out_png=(png or None))
    print(f"Wrote {out_pdf}" + (f" and {png}" if png else ""))


def cmd_viz_deformed(path: str, output: str, warp: float):
    from b3_core.deformed import render_deformed_modes

    out = output or str(Path(path).with_name(Path(path).stem + "_deformed.png"))
    render_deformed_modes(path, out, warp=warp)
    print(f"Wrote {out}")


def _viz_subgroup() -> group:
    return group(
        name="viz",
        help="Figures and reports (optional; not needed for FEA handoff).",
        commands=[
            command(
                name="view",
                help="Geometry, slices, modulus, gallery board.",
                callback=cmd_viz_view,
                arguments=[_CASE_ARG],
                options=[
                    option(flags=["--what"], arg_type=str, default="gallery",
                           choices=["geometry", "slices", "deformation", "modulus",
                                    "polar", "heatmap", "gallery", "all"],
                           help="Which view to render (default: gallery)."),
                    option(flags=["--output", "-o"], arg_type=str, default="",
                           help="Output file (single view) or directory (--what all)."),
                    option(flags=["--serve"], arg_type=str, default="",
                           help="Export interactive HTML (needs [interactive] extra)."),
                    option(flags=["--warp"], arg_type=float, default=0.3,
                           help="Warp factor for --what deformation."),
                ],
            ),
            command(
                name="halo",
                help="Resin-halo figure bundle (PNG).",
                callback=cmd_viz_halo,
                arguments=[_CASE_ARG],
                options=[
                    option(flags=["--output", "-o"], arg_type=str, default="",
                           help="Output directory (default: <case-dir>/img/)."),
                    option(flags=["--sharp"], arg_type=str, default="",
                           help="Sharp-kerf case JSON for before/after comparison."),
                ],
            ),
            command(
                name="datasheet",
                help="One-page datasheet (PDF/PNG).",
                callback=cmd_viz_datasheet,
                arguments=[_CASE_ARG],
                options=[
                    option(flags=["--output", "-o"], arg_type=str, default="",
                           help="Output PDF path (default: <case>.pdf)."),
                    option(flags=["--png"], arg_type=str, default="",
                           help="Also export a PNG to this path."),
                ],
            ),
            command(
                name="deformed",
                help="Six periodic deformation modes (PNG).",
                callback=cmd_viz_deformed,
                arguments=[_CASE_ARG],
                options=[
                    option(flags=["--output", "-o"], arg_type=str, default="",
                           help="Output PNG path (default: <case>_deformed.png)."),
                    option(flags=["--warp"], arg_type=float, default=0.3,
                           help="Displacement warp factor (unit strain = 1.0)."),
                ],
            ),
        ],
    )


def main():
    app = cli(
        name="b3_core",
        help="FEA homogenisation of grooved-core sandwich panels.",
        default="run",
        commands=[
            command(
                name="run",
                help="Homogenise a case from YAML or JSON.",
                callback=cmd_run,
                arguments=[_CASE_ARG],
            ),
            command(
                name="skill",
                help="Print the packaged agent SKILL.md path (or dump with --stdout).",
                callback=cmd_skill,
                options=[
                    option(flags=["--stdout"], arg_type=bool, default=False,
                           help="Print the full SKILL.md to stdout."),
                ],
            ),
        ],
        subgroups=[_sweep_subgroup(), _viz_subgroup()],
    )
    app.run()


if __name__ == "__main__":
    main()