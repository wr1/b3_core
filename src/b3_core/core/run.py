#!/usr/bin/env python3

import sys
from pathlib import Path

from treeparse import argument, cli, command, group, option
from treeparse.models.chain import chain

_CASE_ARG = argument(
    name="path",
    arg_type=str,
    help="Case file (.yaml, .yml, or .json).",
)
CASE_ARG = _CASE_ARG
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


def _error_payload(exc: BaseException) -> dict:
    hint = ""
    if exc.__class__.__name__ == "BackendCapabilityError":
        hint = "Install FEniCSx or pass backend=numpy. The message says which."
    return {
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "hint": hint,
        }
    }


error_payload = _error_payload


def cmd_run(
    path: str,
    ccx_ortho: str = "",
    backend: str = "",
    cache: str = "",
    as_json: bool = False,
    units: str = "Pa",
    no_write: bool = False,
    strict: bool = False,
):
    import json
    import time

    from b3_core.api import run_case
    from b3_core.cache import DiskCache

    kwargs: dict = {}
    if backend:
        kwargs["backend"] = backend
    if cache:
        kwargs["cache"] = DiskCache(cache)
    meta: dict = {}
    started = time.perf_counter()
    payload = None
    written = None
    card_path = None
    try:
        record = run_case(path, meta=meta, **kwargs)
        if not no_write:
            from b3_core.api import write_record

            written = str(write_record(path, record, None))
        if ccx_ortho:
            from b3_core.result import CoreResult

            card_path = Path(ccx_ortho)
            card_path.write_text(CoreResult.from_record(record).ccx_ortho())
        if as_json:
            payload = record.to_agent_json(
                units=units,
                cache_hit=bool(meta.get("cache_hit")),
                elapsed_s=time.perf_counter() - started,
                written=written,
            )
    except Exception as exc:
        if as_json:
            print(json.dumps(_error_payload(exc)))
            raise SystemExit(1) from exc
        raise
    if as_json:
        print(json.dumps(payload))
    elif card_path is not None:
        print(f"Wrote {card_path}")
    elif written is not None:
        print(f"Wrote {written}")
    if strict:
        from b3_core.solvers.checks import strict_failures

        failed = strict_failures(record.diagnostics)
        if failed:
            if not as_json:
                print("diagnostics: " + ", ".join(failed))
            raise SystemExit(1)


def cmd_report_ties(
    path: str,
    kx: str = "",
    ky: str = "",
    unit: str = "1/mm",
    as_json: bool = False,
):
    """Coordinate ties at this curvature. Does not run a finite-element solve.

    Numpy and MFEM constrain the mesh with these rows. A FEniCSx solve stores
    the MPC map on the run record instead.
    """
    import json

    import numpy as np

    from b3_core.api import with_curvature
    from b3_core.loaders import normalize_case
    from b3_core.pipeline import prepare
    from b3_core.solvers.checks import tie_report
    from b3_core.units import curvature_tick, to_per_mm

    try:
        kx_value = None if kx == "" else to_per_mm(float(kx), unit)
        ky_value = None if ky == "" else to_per_mm(float(ky), unit)
        case = with_curvature(path, kx_value, ky_value)
        case_in, _workdir = normalize_case(case)
        prep = prepare(case_in)
        ties = tie_report(np.asarray(prep.mesh.points, dtype=float))
        stored_kx = float(case_in.curvature.kx)
        stored_ky = float(case_in.curvature.ky)
        payload = {
            "unit": unit.strip(),
            "kx": stored_kx,
            "ky": stored_ky,
            "kx_tick": curvature_tick(stored_kx),
            "ky_tick": curvature_tick(stored_ky),
            "ties": ties,
        }
    except Exception as exc:
        if as_json:
            print(json.dumps(_error_payload(exc)))
            raise SystemExit(1) from exc
        raise
    if as_json:
        print(json.dumps(payload))
        return
    print(
        f"curvature unit {payload['unit']} stored as 1/mm: "
        f"kx {payload['kx_tick']} ky {payload['ky_tick']}"
    )
    for row in ties:
        print(
            f"{row['face']}: top_nodes={row['top_nodes']} slaves={row['slaves']} "
            f"identity={row['identity']} multi_node_rows={row['multi_node_rows']} "
            f"ok={row['ok']}"
        )


def _curvature_unit_option():
    return option(
        flags=["--unit"],
        arg_type=str,
        default="1/mm",
        help="Curvature input: 1/mm, 1/m, R-mm, or R-m. Values are stored as 1/mm.",
    )


def _report_subgroup():
    from b3_core.cli_ext import cmd_report_build

    return group(
        name="report",
        help="Mesh reports and the report skeleton. ties does not solve.",
        commands=[
            command(
                name="ties",
                help="Periodic ties on the high faces at this curvature.",
                callback=cmd_report_ties,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--kx"],
                        arg_type=str,
                        default="",
                        help="Mould curvature kx in the --unit. Omit to keep the case value.",
                    ),
                    option(
                        flags=["--ky"],
                        arg_type=str,
                        default="",
                        help="Mould curvature ky in the --unit. Omit to keep the case value.",
                    ),
                    _curvature_unit_option(),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print {unit, kx, ky, ticks, ties}. kx and ky are 1/mm.",
                    ),
                ],
            ),
            command(
                name="build",
                help="Markdown and Typst skeleton. Numbers come from the project JSON.",
                callback=cmd_report_build,
                arguments=[argument(name="path", arg_type=str, help="project.yaml.")],
                options=[
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Output directory (default: report/).",
                    ),
                    option(
                        flags=["--template"],
                        arg_type=str,
                        default="",
                        help="Optional Typst file copied beside the skeleton.",
                    ),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print the written paths.",
                    ),
                ],
            ),
        ],
    )


def cmd_doctor(as_json: bool = False):
    import json

    from b3_core.doctor import doctor

    report = doctor()
    if as_json:
        print(json.dumps(report))
    else:
        for name, version in report["imports"].items():
            print(f"{name}: {version or 'missing'}")
        print(f"mpi_size: {report['mpi_size']}")
        print(f"mumps: {report['mumps']}")
        print(f"factorisation: {report['factorisation']}")
        if report.get("error"):
            print(report["error"])
        elif report["ok"]:
            solve = report["solve"]
            print(f"solve: {solve['cells']} cells, Ex={solve['Ex']}")
    if not report["ok"]:
        raise SystemExit(1)


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


def cmd_sweep_grid(
    path: str,
    kx: str = "-0.008:0.008:3",
    ky: str = "0",
    unit: str = "1/mm",
    backend: str = "fenicsx",
    out: str = "",
    as_json: bool = False,
    dry_run: bool = False,
    allow_inert_axis: bool = False,
    fit_result: str = "",
    workers: int = 1,
    root: str = "",
    cache: str = "",
) -> None:
    """Curvature product for one case. ``--dry-run`` prints the solve count."""
    import json

    from b3_core.sweep.curvature import InertAxisError, sweep_curvature, write_grid

    del root  # the sweep group inherits --root; this command reads the case path
    store = None
    if cache:
        from b3_core.cache import DiskCache

        store = DiskCache(cache)
    try:
        result = sweep_curvature(
            path,
            kx,
            ky,
            unit=unit,
            cache=store,
            backend=backend,
            workers=workers,
            allow_inert_axis=allow_inert_axis,
            dry_run=dry_run,
            fit_result=fit_result or None,
        )
    except (InertAxisError, ValueError) as exc:
        if as_json:
            print(json.dumps(_error_payload(exc)))
        else:
            print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    if out and not dry_run:
        write_grid(result, out)
        print(f"Wrote {out}")
    print(
        f"curvature unit {unit.strip()} stored as 1/mm; "
        f"n_solves={result['manifest']['n_solves']}"
    )
    if as_json:
        print(json.dumps(result["manifest"]))
    elif dry_run:
        print("dry-run")


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
            command(
                name="grid",
                help="Curvature grid for one case. Default backend is fenicsx.",
                callback=cmd_sweep_grid,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--kx"],
                        arg_type=str,
                        default="-0.008:0.008:3",
                        help="kx samples in --unit. One value, or start:stop:count.",
                    ),
                    option(
                        flags=["--ky"],
                        arg_type=str,
                        default="0",
                        help="ky samples in --unit. One value, or start:stop:count.",
                    ),
                    _curvature_unit_option(),
                    option(
                        flags=["--backend"],
                        arg_type=str,
                        default="fenicsx",
                        help="Pinned backend. Default fenicsx, not auto.",
                    ),
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Directory for grid.parquet, grid.csv, and manifest.json.",
                    ),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print the manifest. A failed point does not change the exit.",
                    ),
                    option(
                        flags=["--dry-run"],
                        dest="dry_run",
                        flag=True,
                        help="Print the solve count and write nothing.",
                    ),
                    option(
                        flags=["--allow-inert-axis"],
                        dest="allow_inert_axis",
                        flag=True,
                        help="Keep a kx or ky range on an axis with no grooves.",
                    ),
                    option(
                        flags=["--fit-result"],
                        arg_type=str,
                        default="",
                        help="FitResult JSON. Its hash and status go on the manifest.",
                    ),
                    option(
                        flags=["--workers"],
                        arg_type=int,
                        default=1,
                        help="Process pool size. Children set OMP_NUM_THREADS=1.",
                    ),
                ],
                sort_key=5,
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


def cmd_viz_halo_curvature(
    output: str,
    case: str,
    kx_open: float,
    kx_closed: float,
    unit: str = "1/mm",
):
    """Side-by-side closed|flat|open halo cuts + wall-normal strip."""
    import json as _json

    from b3_core.units import to_per_mm
    from b3_core.viz.halo import render_halo_curvature_figures

    kx_open = to_per_mm(kx_open, unit)
    kx_closed = to_per_mm(kx_closed, unit)
    print(f"curvature unit {unit.strip()} stored as 1/mm")
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


def cmd_viz_datasheet(
    path: str,
    output: str = "",
    png: str = "",
    full: bool = False,
    draft: bool = False,
    sweep: str = "",
    prose: str = "",
    cache: str = "",
    grid: str = "",
):
    if full or sweep or grid:
        from b3_core.cli_ext import write_full_datasheet

        pdf = write_full_datasheet(
            path,
            output,
            prose=prose,
            grid=sweep or grid,
            draft=draft,
            cache=cache,
        )
        print(f"Wrote {pdf}")
        return
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
    from b3_core.cli_ext import cmd_viz_datasheet_sweep, cmd_viz_figs

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
                        help="kx for open panel, in --unit (curved_panel: + opens top-mouth).",
                    ),
                    option(
                        flags=["--kx-closed"],
                        arg_type=float,
                        default=-0.012,
                        help="kx for closed panel, in --unit (curved_panel: − pinches top-mouth).",
                    ),
                    _curvature_unit_option(),
                ],
            ),
            command(
                name="sign",
                help="Small sign card: bottom and top mouths at one positive k.",
                callback=cmd_viz_sign,
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="PNG path (default: sign_schematic.png).",
                    ),
                    option(
                        flags=["--kx"],
                        arg_type=float,
                        default=0.004,
                        help="Curvature used to draw the walls, in --unit.",
                    ),
                    _curvature_unit_option(),
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
                    option(
                        flags=["--full"],
                        flag=True,
                        help="Curvature sheet from data.json. Needs a passing stamp or --draft.",
                    ),
                    option(
                        flags=["--draft"],
                        flag=True,
                        help="Build the full sheet when the acceptance stamp is not ok.",
                    ),
                    option(
                        flags=["--sweep"],
                        arg_type=str,
                        default="",
                        help="Grid JSON. Same sheet as viz datasheet-sweep.",
                    ),
                    option(
                        flags=["--prose"],
                        arg_type=str,
                        default="",
                        help="prose.json. The agent writes this file and nothing else.",
                    ),
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache recorded on the sheet.",
                    ),
                    option(
                        flags=["--grid"],
                        arg_type=str,
                        default="",
                        help="Alias of --sweep.",
                    ),
                ],
            ),
            command(
                name="datasheet-sweep",
                help="Alias of viz datasheet --full --sweep.",
                callback=cmd_viz_datasheet_sweep,
                arguments=[
                    argument(
                        name="path",
                        arg_type=str,
                        help="Grid JSON with rows.",
                    )
                ],
                options=[
                    option(
                        flags=["--output", "-o"],
                        arg_type=str,
                        default="",
                        help="Output directory or PDF path.",
                    ),
                    option(
                        flags=["--prose"],
                        arg_type=str,
                        default="",
                        help="prose.json.",
                    ),
                    option(
                        flags=["--draft"],
                        flag=True,
                        help="Build without an acceptance stamp.",
                    ),
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache recorded on the sheet.",
                    ),
                ],
            ),
            command(
                name="figs",
                help="Datasheet PNGs from a grid or kerf-map JSON.",
                callback=cmd_viz_figs,
                arguments=[
                    argument(name="path", arg_type=str, help="Grid or kerf-map JSON.")
                ],
                options=[
                    option(
                        flags=["--out", "-o"],
                        arg_type=str,
                        default="",
                        help="Output directory.",
                    ),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print the written paths.",
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


def cmd_viz_sign(output: str = "", kx: float = 0.004, unit: str = "1/mm"):
    """Write the per-family sign card. Titles come from each mouth's opens_for."""
    import matplotlib.pyplot as plt

    from b3_core.units import to_per_mm
    from b3_core.viz.halo.curvature_figs import plot_sign_schematic

    stored = to_per_mm(kx, unit)
    fig = plot_sign_schematic(kx=stored)
    dest = Path(output) if output else Path("sign_schematic.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"curvature unit {unit.strip()} stored as 1/mm")
    print(f"Wrote {dest}")


def cmd_surrogate_lookup(
    path: str, kx: str, cell_size: float, output: str, unit: str = "1/mm"
):
    """Mass lookup of properties for a comma-separated curvature vector."""
    from b3_core.physics_surrogate import CorePhysicsSurrogate
    from b3_core.units import to_per_mm

    surr = CorePhysicsSurrogate.from_json(path)
    kx_vec = [
        to_per_mm(float(item), unit) for item in kx.replace(" ", "").split(",") if item
    ]
    print(f"curvature unit {unit.strip()} stored as 1/mm")
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
                        help="Comma-separated curvatures in --unit, e.g. -0.008,0,0.008.",
                    ),
                    _curvature_unit_option(),
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


def _cache_dir(directory: str) -> str:
    return directory or ".b3cache"


cache_dir = _cache_dir


def cmd_cache_stats(directory: str = "", as_json: bool = False):
    import json

    from b3_core.cache import cache_stats

    payload = cache_stats(_cache_dir(directory))
    if as_json:
        print(json.dumps(payload))
        return
    print(f"{payload['n']} entries, {payload['n_stale']} stale, dir {payload['dir']}")
    for name, bucket in sorted(payload["by_backend"].items()):
        print(f"  {name}: {bucket['n']} ({bucket['n_stale']} stale)")


def cmd_cache_inspect(directory: str = "", backend: str = "", as_json: bool = False):
    import json

    from b3_core.cache import cache_rows

    rows = cache_rows(_cache_dir(directory), backend=backend)
    if as_json:
        print(json.dumps(rows))
        return
    for row in rows:
        mark = "stale" if row["stale"] else "fresh"
        print(
            f"{row['case_hash']} {row['backend']} {row['solver_stamp']} "
            f"kx={row['kx']} ky={row['ky']} {mark}"
        )


def cmd_cache_purge(
    directory: str = "",
    backend: str = "",
    stale: bool = False,
    as_json: bool = False,
):
    import json
    import sys

    from b3_core.cache import cache_purge

    try:
        removed = cache_purge(_cache_dir(directory), stale=stale, backend=backend)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    if as_json:
        print(json.dumps({"removed": removed}))
        return
    print(f"removed {len(removed)}")


def _cache_subgroup() -> group:
    from b3_core.cli_ext import cmd_cache_export, cmd_cache_import

    directory = option(
        flags=["--dir"],
        dest="directory",
        arg_type=str,
        default="",
        help="Cache directory (default: .b3cache).",
    )
    backend = option(
        flags=["--backend"],
        arg_type=str,
        default="",
        help="Only this resolved backend.",
    )
    as_json = option(
        flags=["--json", "--agent-json"],
        dest="as_json",
        flag=True,
        help="Print one JSON object.",
    )
    inspect = command(
        name="inspect",
        help="List entries: case digest, backend, stamp, kx, ky.",
        callback=cmd_cache_inspect,
        options=[directory, backend, as_json],
    )
    return group(
        name="cache",
        help="Inspect and purge the disk cache.",
        commands=[
            command(
                name="stats",
                help="Count entries and stale stamps.",
                callback=cmd_cache_stats,
                options=[directory, as_json],
            ),
            inspect,
            command(
                name="ls",
                help="Alias of inspect.",
                callback=cmd_cache_inspect,
                options=[directory, backend, as_json],
            ),
            command(
                name="purge",
                help="Delete entries. Pass --stale and/or --backend.",
                callback=cmd_cache_purge,
                options=[
                    directory,
                    backend,
                    option(
                        flags=["--stale"],
                        flag=True,
                        help="Only entries whose solver stamp is not current.",
                    ),
                    as_json,
                ],
            ),
            command(
                name="export",
                help="Zip cache entries and a stamp manifest.",
                callback=cmd_cache_export,
                arguments=[
                    argument(name="dest", arg_type=str, help="Zip path."),
                ],
                options=[directory, as_json],
            ),
            command(
                name="import",
                help="Copy a cache zip into --dir. Existing keys are kept.",
                callback=cmd_cache_import,
                arguments=[
                    argument(
                        name="bundle", arg_type=str, help="Zip from cache export."
                    ),
                ],
                options=[
                    directory,
                    option(
                        flags=["--overwrite"], flag=True, help="Replace existing keys."
                    ),
                    as_json,
                ],
            ),
        ],
    )


def _check_options() -> list:
    return [
        option(
            flags=["--backends"],
            arg_type=str,
            default="fenicsx,mfem,numpy",
            help="Comma-separated backends. A backend that cannot run is skipped with a reason.",
        ),
        option(
            flags=["--points"],
            arg_type=str,
            default="",
            help='Curvature pairs "kx,ky kx,ky" in --unit. Empty uses --kx and --ky.',
        ),
        option(
            flags=["--kx"],
            arg_type=str,
            default="",
            help="One kx in --unit, or start:stop:count when --grid is set.",
        ),
        option(
            flags=["--ky"],
            arg_type=str,
            default="",
            help="One ky in --unit, or start:stop:count when --grid is set.",
        ),
        _curvature_unit_option(),
        option(
            flags=["--grid"],
            flag=True,
            help="Use the Cartesian product of the --kx and --ky ranges.",
        ),
        option(
            flags=["--refine"],
            arg_type=int,
            default=0,
            help="Extra madd midpoint passes. 0 keeps the case mesh.",
        ),
        option(
            flags=["--rtol"],
            arg_type=float,
            default=1e-4,
            help="Flat-anchor relative tolerance. Default 1e-4 (0.01 %).",
        ),
        option(
            flags=["--cache"],
            arg_type=str,
            default="",
            help="Disk cache directory. Omit to run without a cache.",
        ),
        option(
            flags=["--out"],
            arg_type=str,
            default="",
            help="Write the JSON envelope to this path.",
        ),
        option(
            flags=["--json", "--agent-json"],
            dest="as_json",
            flag=True,
            help="Print the JSON envelope. A flat-anchor failure still exits 1.",
        ),
    ]


def _select_points(
    *,
    cross: bool,
    points: str,
    kx: str,
    ky: str,
    grid: bool,
    unit: str = "1/mm",
) -> list[tuple[float, float]]:
    from b3_core.units import axis_to_per_mm, points_to_per_mm

    if cross:
        return [(0.0, 0.0)]
    if points.strip():
        return points_to_per_mm(points, unit)
    if grid:
        return [
            (float(x_value), float(y_value))
            for x_value in axis_to_per_mm(kx, unit)
            for y_value in axis_to_per_mm(ky, unit)
        ]
    if kx.strip() or ky.strip():
        return [
            (
                axis_to_per_mm(kx or "0", unit)[0],
                axis_to_per_mm(ky or "0", unit)[0],
            )
        ]
    return [(0.0, 0.0)]


def _dispatch_check(
    cross: bool,
    path: str,
    backends: str,
    points: str,
    kx: str,
    ky: str,
    grid: bool,
    refine: int,
    rtol: float,
    cache: str,
    out: str,
    as_json: bool,
    unit: str = "1/mm",
) -> None:
    import json

    from b3_core.checks.backends import check_backends, check_cross, format_table

    names = [item.strip() for item in backends.split(",") if item.strip()]
    store = None
    if cache:
        from b3_core.cache import DiskCache

        store = DiskCache(cache)
    chosen = _select_points(
        cross=cross, points=points, kx=kx, ky=ky, grid=grid, unit=unit
    )
    runner = check_cross if cross else check_backends
    try:
        report = runner(
            path,
            backends=names,
            points=chosen,
            refine=int(refine),
            rtol=float(rtol),
            cache=store,
        )
    except Exception as exc:
        if as_json:
            print(json.dumps(_error_payload(exc)))
            raise SystemExit(1) from exc
        raise
    report = dict(report)
    report["unit"] = unit.strip()
    text = json.dumps(report, indent=2)
    if out:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
    if as_json:
        print(text)
    else:
        print(f"curvature unit {report['unit']} stored as 1/mm")
        print(format_table(report), end="")
    if not report["ok"]:
        raise SystemExit(1)


def cmd_check_backends(
    path: str,
    backends: str = "fenicsx,mfem,numpy",
    points: str = "",
    kx: str = "",
    ky: str = "",
    grid: bool = False,
    refine: int = 0,
    rtol: float = 1e-4,
    cache: str = "",
    out: str = "",
    unit: str = "1/mm",
    as_json: bool = False,
) -> None:
    _dispatch_check(
        False,
        path,
        backends,
        points,
        kx,
        ky,
        grid,
        refine,
        rtol,
        cache,
        out,
        as_json,
        unit,
    )


def cmd_check_cross(
    path: str,
    backends: str = "fenicsx,mfem,numpy",
    points: str = "",
    kx: str = "",
    ky: str = "",
    grid: bool = False,
    refine: int = 0,
    rtol: float = 1e-4,
    cache: str = "",
    out: str = "",
    unit: str = "1/mm",
    as_json: bool = False,
) -> None:
    _dispatch_check(
        True,
        path,
        backends,
        points,
        kx,
        ky,
        grid,
        refine,
        rtol,
        cache,
        out,
        as_json,
        unit,
    )


def _check_subgroup():
    from b3_core.cli_ext import (
        cmd_check_accept,
        cmd_check_convergence,
        cmd_check_reproduce,
    )

    options = _check_options()
    as_json = option(
        flags=["--json", "--agent-json"],
        dest="as_json",
        flag=True,
        help="Print one JSON object.",
    )
    return group(
        name="check",
        help="Cross-backend agreement. Flat points fail above --rtol; curved points are flagged.",
        commands=[
            command(
                name="backends",
                help="Compare backends on a curvature list or grid.",
                callback=cmd_check_backends,
                arguments=[_CASE_ARG],
                options=options,
            ),
            command(
                name="cross",
                help="Alias that forces the flat point.",
                callback=cmd_check_cross,
                arguments=[_CASE_ARG],
                options=options,
            ),
            command(
                name="reproduce",
                help="Re-solve stored points. A stamp change is drift, not a pass.",
                callback=cmd_check_reproduce,
                arguments=[
                    argument(
                        name="path",
                        arg_type=str,
                        help="values.json from an earlier solve.",
                    )
                ],
                options=[
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache for the fresh solves.",
                    ),
                    option(
                        flags=["--rtol-pct"],
                        dest="rtol_pct",
                        arg_type=float,
                        default=1e-6,
                        help="Pass limit in percent at the same solver stamp.",
                    ),
                    as_json,
                ],
            ),
            command(
                name="accept",
                help="Score a project file. Missing artifacts fail closed.",
                callback=cmd_check_accept,
                arguments=[
                    argument(name="path", arg_type=str, help="project.yaml or JSON.")
                ],
                options=[
                    option(
                        flags=["--profile"],
                        arg_type=str,
                        default="",
                        help="Named checklist YAML. Not used unless you pass it.",
                    ),
                    as_json,
                ],
            ),
            command(
                name="convergence",
                help="Compare two madd refinements. One station cannot be refined.",
                callback=cmd_check_convergence,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--backend"],
                        arg_type=str,
                        default="numpy",
                        help="Solver for both refinements.",
                    ),
                    as_json,
                ],
            ),
        ],
    )


_PAYLOAD_COMMANDS = frozenset(
    {
        "run",
        "doctor",
        "ties",
        "backends",
        "cross",
        "stats",
        "inspect",
        "ls",
        "purge",
        "fit",
        "grid",
        "sign",
        "bounds",
        "residuals",
        "estimate-halo",
        "refine",
        "sensitivity",
        "design",
        "rsm",
        "figs",
        "datasheet",
        "datasheet-sweep",
        "map",
        "build",
        "bisect",
        "reproduce",
        "accept",
        "convergence",
        "export",
        "import",
    }
)


def _rewrite_payload_json(argv: list[str]) -> list[str]:
    """Keep treeparse's schema dump on a leading ``--json``.

    treeparse exits before parsing when ``--json`` appears anywhere, and
    prints the command schema. A ``--json`` after a subcommand is the agent
    payload flag. Rewrite only that one to ``--agent-json``.
    """
    if "--json" not in argv:
        return argv
    command_at = [i for i, arg in enumerate(argv) if arg in _PAYLOAD_COMMANDS]
    json_at = [i for i, arg in enumerate(argv) if arg == "--json"]
    if not command_at or min(json_at) < min(command_at):
        return argv
    return ["--agent-json" if arg == "--json" else arg for arg in argv]


def _kerfs_subgroup():
    from b3_core.cli_ext import cmd_kerfs_map

    return group(
        name="kerfs",
        help="Kerf open, flat, and closed states.",
        commands=[
            command(
                name="map",
                help="3x3 sign grid. Geometry everywhere, solves at the corners.",
                callback=cmd_kerfs_map,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--magnitude"],
                        arg_type=float,
                        default=0.001,
                        help="|k| of the open and closed corners, in --unit.",
                    ),
                    _curvature_unit_option(),
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Write the map JSON here.",
                    ),
                    option(
                        flags=["--backend"],
                        arg_type=str,
                        default="fenicsx",
                        help="Solver for the four corners.",
                    ),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print the map.",
                    ),
                ],
            )
        ],
    )


def _diagnose_subgroup():
    from b3_core.cli_ext import cmd_diagnose_bisect

    return group(
        name="diagnose",
        help="Attribute a curvature change to mesh, field, or wall morph.",
        commands=[
            command(
                name="bisect",
                help="Flat field, curved field, curved mesh, morph off.",
                callback=cmd_diagnose_bisect,
                arguments=[_CASE_ARG],
                options=[
                    option(
                        flags=["--kx"],
                        arg_type=float,
                        default=5e-5,
                        help="Curvature kx in --unit.",
                    ),
                    option(
                        flags=["--ky"],
                        arg_type=float,
                        default=0.0,
                        help="Curvature ky in --unit.",
                    ),
                    _curvature_unit_option(),
                    option(
                        flags=["--backend"],
                        arg_type=str,
                        default="numpy",
                        help="Recorded on the report. The default solve is numpy.",
                    ),
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Write the bisection JSON here.",
                    ),
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print the four arms.",
                    ),
                ],
            )
        ],
    )


def main(argv: list[str] | None = None) -> None:
    from b3_core.fit.commands import fit_subgroup

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    sys.argv = [sys.argv[0], *_rewrite_payload_json(sys.argv[1:])]
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
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help=(
                            "Print one agent payload. Moduli carry units. "
                            "Failures print {error:{type,message,hint}} and exit 1."
                        ),
                    ),
                    option(
                        flags=["--units"],
                        arg_type=str,
                        default="Pa",
                        help="Scale moduli and stiffness in --json: Pa (default) or GPa.",
                    ),
                    option(
                        flags=["--no-write"],
                        dest="no_write",
                        flag=True,
                        help="Skip run<hash>.json. The payload's written field is null.",
                    ),
                    option(
                        flags=["--strict"],
                        flag=True,
                        help="Exit 1 when a diagnostic is warn or fail, or a tie is not ok.",
                    ),
                ],
            ),
            command(
                name="doctor",
                help="Import FEniCSx, report MPI and MUMPS, solve a 2x2x2 cube.",
                callback=cmd_doctor,
                options=[
                    option(
                        flags=["--json", "--agent-json"],
                        dest="as_json",
                        flag=True,
                        help="Print one JSON report. Exit 1 when the stack is not usable.",
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
        subgroups=[
            _sweep_subgroup(),
            _viz_subgroup(),
            _surrogate_subgroup(),
            _cache_subgroup(),
            _report_subgroup(),
            _check_subgroup(),
            fit_subgroup(),
            _kerfs_subgroup(),
            _diagnose_subgroup(),
        ],
    )
    app.run()


if __name__ == "__main__":
    main()
