#!/usr/bin/env python3

from pathlib import Path

from treeparse import argument, cli, command, group, option
from treeparse.models.chain import chain

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
    help=(
        "Study root. With no value, the current directory is used when it "
        "contains bases/ or mfem_patterns/; otherwise the command exits."
    ),
)
_SWEEP_CACHE_OPT = option(
    flags=["--cache"],
    arg_type=str,
    default="",
    inherit=True,
    help="Disk cache directory. Default: <study root>/.b3cache.",
)


def cmd_run(path: str, ccx_ortho: str = "", backend: str = "", cache: str = ""):
    from b3_core.api import homogenize_to_disk
    from b3_core.cache import DiskCache

    kwargs: dict = {}
    if backend:
        kwargs["backend"] = backend
    if cache:
        kwargs["cache"] = DiskCache(cache)
    result, written = homogenize_to_disk(path, **kwargs)
    print(f"Wrote {written}")
    if not ccx_ortho:
        return
    dest = Path(ccx_ortho)
    dest.write_text(result.ccx_ortho())
    print(f"Wrote {dest}")


def cmd_skill(stdout: bool):
    from b3_core.skill import read_skill, skill_path

    if stdout:
        print(read_skill(), end="")
    else:
        print(skill_path())


def resolve_sweep_root(root: str = "") -> Path:
    """Directory a sweep reads bases and patterns from.

    An explicit ``--root`` is used as given. Otherwise the current directory
    qualifies when it contains ``bases/`` or ``mfem_patterns/``.
    """
    import sys

    if root:
        return Path(root)
    cwd = Path.cwd()
    if (cwd / "bases").is_dir() or (cwd / "mfem_patterns").is_dir():
        return cwd
    print("no study root found; pass --root", file=sys.stderr)
    raise SystemExit(2)


def _sweep_context(root: str = "", cache: str = ""):
    from b3_core.sweep.context import SweepContext

    study = resolve_sweep_root(root)
    return SweepContext(study, Path(cache) if cache else None)


def _sweep_exit(code: int) -> None:
    import sys

    sys.exit(code)


def cmd_sweep_thickness(root: str, cache: str):
    from b3_core.sweep import homogenize

    _sweep_exit(homogenize.run_thickness(_sweep_context(root, cache)))


def cmd_sweep_curvature(root: str, cache: str):
    from b3_core.sweep import homogenize

    _sweep_exit(homogenize.run_curvature(_sweep_context(root, cache)))


def cmd_sweep_patterns(root: str, cache: str):
    from b3_core.sweep import homogenize

    _sweep_exit(homogenize.run_patterns(_sweep_context(root, cache)))


def _sweep_subgroup() -> group:
    # treeparse validates each chained callback against that command's own
    # options, and a chain rejects a repeated option dest. The first step
    # owns --root and stores it here; later steps take no arguments.
    # Steps must not sys.exit: chain_runner would stop after thickness.
    shared: dict[str, str | int] = {"root": "", "cache": "", "code": 0}

    def _thickness_chain(root: str, cache: str):
        from b3_core.sweep import homogenize

        shared["root"] = root
        shared["cache"] = cache
        shared["code"] = homogenize.run_thickness(_sweep_context(root, cache))

    def _curvature_chain():
        from b3_core.sweep import homogenize

        if shared["code"]:
            return
        shared["code"] = homogenize.run_curvature(
            _sweep_context(str(shared["root"]), str(shared["cache"]))
        )

    def _patterns_chain():
        from b3_core.sweep import homogenize

        code = int(shared["code"])
        if not code:
            code = homogenize.run_patterns(
                _sweep_context(str(shared["root"]), str(shared["cache"]))
            )
        _sweep_exit(code)

    leaf_options = [_SWEEP_ROOT_OPT, _SWEEP_CACHE_OPT]
    thickness = command(
        name="thickness",
        help="Homogenise thickness sweep (20–50 mm).",
        callback=cmd_sweep_thickness,
        options=leaf_options,
        sort_key=0,
    )
    curvature = command(
        name="curvature",
        help="Homogenise curvature sweep (kx).",
        callback=cmd_sweep_curvature,
        options=leaf_options,
        sort_key=1,
    )
    patterns = command(
        name="patterns",
        help="Homogenise groove-pattern sweep.",
        callback=cmd_sweep_patterns,
        options=leaf_options,
        sort_key=2,
    )

    def _chained(name: str, sort_key: int):
        # Each chain needs its own command objects; a repeated dest across
        # one chain is rejected, and one command object cannot sit in two.
        return chain(
            name=name,
            help="thickness ➜ curvature ➜ patterns",
            chained_commands=[
                command(
                    name="thickness",
                    help="Homogenise thickness sweep (20–50 mm).",
                    callback=_thickness_chain,
                    options=leaf_options,
                    sort_key=0,
                ),
                command(
                    name="curvature",
                    help="Homogenise curvature sweep (kx).",
                    callback=_curvature_chain,
                    sort_key=1,
                ),
                command(
                    name="patterns",
                    help="Homogenise groove-pattern sweep.",
                    callback=_patterns_chain,
                    sort_key=2,
                ),
            ],
            sort_key=sort_key,
        )

    return group(
        name="sweep",
        help="Parametric homogenisation studies.",
        options=leaf_options,
        commands=[
            thickness,
            curvature,
            patterns,
            _chained("homogenize", 3),
            _chained("homogenise", 4),
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


def _sharp_sibling(path: str) -> Path | None:
    """If *path* is ``*_halo`` / ``*_scored``, return the matching sharp case."""
    p = Path(path)
    for tag in ("_halo", "_scored"):
        if p.stem.endswith(tag):
            cand = p.with_name(p.stem[: -len(tag)] + p.suffix)
            if cand.is_file():
                return cand
    return None


def cmd_viz_halo(path: str, output: str, sharp: str):
    import json as _json

    from b3_core.viz.halo import render_halo_figures

    scored = _json.loads(Path(path).read_text())
    out = Path(output) if output else Path(path).parent / "img"
    sharp_inp = _json.loads(Path(sharp).read_text()) if sharp else None
    if sharp_inp is None:
        sibling = _sharp_sibling(path)
        if sibling is not None:
            sharp_inp = _json.loads(sibling.read_text())
    paths = render_halo_figures(scored, out, sharp_inp=sharp_inp)
    print("Wrote " + ", ".join(str(p) for p in paths))


def cmd_viz_halo_curvature(output: str, case: str, kx_open: float, kx_closed: float):
    """Side-by-side closed|flat|open halo cuts + wall-normal strip."""
    import json as _json

    from b3_core.viz.halo import render_halo_curvature_figures

    base = _json.loads(Path(case).read_text()) if case else None
    out = (
        Path(output)
        if output
        else (Path(case).parent / "img" if case else Path("examples/img"))
    )
    paths = render_halo_curvature_figures(
        out, base_inp=base, kx_open=kx_open, kx_closed=kx_closed
    )
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
                    option(
                        flags=["--what"],
                        arg_type=str,
                        default="gallery",
                        choices=[
                            "geometry",
                            "slices",
                            "deformation",
                            "modulus",
                            "polar",
                            "heatmap",
                            "gallery",
                            "all",
                        ],
                        help="Which view to render (default: gallery).",
                    ),
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output file (single view) or directory (--what all).",
                    ),
                    option(
                        flags=["--serve"],
                        arg_type=str,
                        default="",
                        help="Export interactive HTML (needs [interactive] extra).",
                    ),
                    option(
                        flags=["--warp"],
                        arg_type=float,
                        default=0.3,
                        help="Warp factor for --what deformation.",
                    ),
                ],
            ),
            command(
                name="halo",
                help="Resin-halo figure bundle (PNG).",
                callback=cmd_viz_halo,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output directory (default: <case-dir>/img/).",
                    ),
                    option(
                        flags=["--sharp"],
                        arg_type=str,
                        default="",
                        help="Sharp-kerf case JSON for before/after comparison.",
                    ),
                ],
            ),
            command(
                name="halo-curvature",
                help="Halo + curvature composition figures (closed|flat|open).",
                callback=cmd_viz_halo_curvature,
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output directory (default: examples/img or <case>/img).",
                    ),
                    option(
                        flags=["--case"],
                        arg_type=str,
                        default="",
                        help="Optional scored case JSON (default: built-in uniaxial demo).",
                    ),
                    option(
                        flags=["--kx-open"],
                        arg_type=float,
                        default=0.012,
                        help="kx for open panel (curved_panel: + opens top-mouth).",
                    ),
                    option(
                        flags=["--kx-closed"],
                        arg_type=float,
                        default=-0.012,
                        help="kx for closed panel (curved_panel: − pinches top-mouth).",
                    ),
                ],
            ),
            command(
                name="datasheet",
                help="One-page datasheet (PDF/PNG).",
                callback=cmd_viz_datasheet,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output PDF path (default: <case>.pdf).",
                    ),
                    option(
                        flags=["--png"],
                        arg_type=str,
                        default="",
                        help="Also export a PNG to this path.",
                    ),
                ],
            ),
            command(
                name="deformed",
                help="Six periodic deformation modes (PNG).",
                callback=cmd_viz_deformed,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output PNG path (default: <case>_deformed.png).",
                    ),
                    option(
                        flags=["--warp"],
                        arg_type=float,
                        default=0.3,
                        help="Displacement warp factor (unit strain = 1.0).",
                    ),
                ],
            ),
        ],
    )


def cmd_surrogate_fit(output: str, cache: str):
    """Fit physics surrogate on a κ × ky × cell_size homogenization grid."""
    import sys

    from b3_core.cache import DiskCache
    from b3_core.physics_surrogate import fit_from_homogenization

    if not output:
        print("surrogate fit: --output is required", file=sys.stderr)
        raise SystemExit(2)
    out = Path(output)
    store = DiskCache(cache) if cache else DiskCache(out.with_name(".b3cache"))
    surr = fit_from_homogenization(cache=store)
    surr.to_json(out)
    print(f"Wrote {out}  targets={surr.targets}")


def cmd_surrogate_lookup(path: str, kx: str, cell_size: float, output: str):
    """Mass lookup of properties for a comma-separated curvature vector."""
    from b3_core.physics_surrogate import CorePhysicsSurrogate

    surr = CorePhysicsSurrogate.from_json(path)
    kx_vec = [float(x) for x in kx.replace(" ", "").split(",") if x]
    df = surr.lookup(kx_vec, cell_size=cell_size)
    if output:
        Path(output).write_text(df.to_csv(index=False))
        print(f"Wrote {output}  ({len(df)} stations)")
    else:
        print(df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))


def _surrogate_subgroup() -> group:
    return group(
        name="surrogate",
        help="Physics-based stiffness/mass surrogate vs curvature.",
        commands=[
            command(
                name="fit",
                help="Fit surrogate on homogenization grid (κ × cell_size).",
                callback=cmd_surrogate_fit,
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        required=True,
                        help="JSON path for the fitted surrogate.",
                    ),
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Homogenization grid JSON cache path.",
                    ),
                ],
            ),
            command(
                name="lookup",
                help="Batch properties for a curvature vector (mass lookup).",
                callback=cmd_surrogate_lookup,
                arguments=[
                    argument(
                        name="path",
                        arg_type=str,
                        help="Fitted surrogate JSON from `surrogate fit`.",
                    ),
                ],
                options=[
                    option(
                        flags=["--kx"],
                        arg_type=str,
                        default="0",
                        help="Comma-separated curvatures [1/mm], e.g. -0.008,0,0.008.",
                    ),
                    option(
                        flags=["--cell-size"],
                        arg_type=float,
                        default=0.0,
                        help="Halo width [mm] (0 = sharp kerf).",
                    ),
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Optional CSV path (default: print table).",
                    ),
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
                options=[
                    option(
                        flags=["--ccx-ortho"],
                        arg_type=str,
                        default="",
                        help="Write a CalculiX *elastic,type=ortho card to this path.",
                    ),
                    option(
                        flags=["--backend"],
                        arg_type=str,
                        default="",
                        help="Override the case backend (default: the case's own, usually auto).",
                    ),
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache directory. Omit to run without a cache.",
                    ),
                ],
            ),
            command(
                name="skill",
                help="Print the packaged agent SKILL.md path (or dump with --stdout).",
                callback=cmd_skill,
                options=[
                    option(
                        flags=["--stdout"],
                        arg_type=bool,
                        default=False,
                        help="Print the full SKILL.md to stdout.",
                    ),
                ],
            ),
        ],
        subgroups=[_sweep_subgroup(), _viz_subgroup(), _surrogate_subgroup()],
    )
    app.run()


if __name__ == "__main__":
    main()
