"""Callbacks for the datasheet commands added after the core CLI."""

import json
from pathlib import Path


def _fail(exc: BaseException) -> None:
    from b3_core.core.run import error_payload

    print(json.dumps(error_payload(exc)))
    raise SystemExit(1) from exc


def _print(payload, *, as_json: bool) -> None:
    text = json.dumps(payload, indent=2)
    print(text if as_json else text)


def cmd_check_reproduce(
    path: str,
    cache: str = "",
    rtol_pct: float = 1e-6,
    as_json: bool = False,
):
    from b3_core.cache import DiskCache
    from b3_core.checks.reproduce import check_reproduce

    try:
        report = check_reproduce(
            path,
            cache=DiskCache(cache) if cache else None,
            rtol_pct=rtol_pct,
        )
    except Exception as exc:
        _fail(exc)
    _print(report, as_json=as_json)
    if not report["ok"]:
        raise SystemExit(1)


def cmd_check_accept(
    path: str,
    profile: str = "",
    as_json: bool = False,
):
    from b3_core.checks.accept import check_accept

    try:
        report = check_accept(path, profile=profile or None)
    except Exception as exc:
        _fail(exc)
    _print(report, as_json=as_json)
    if not report["ok"]:
        raise SystemExit(1)


def cmd_check_convergence(
    path: str,
    backend: str = "numpy",
    as_json: bool = False,
):
    from b3_core.checks.convergence import check_convergence

    try:
        report = check_convergence(path, backend=backend)
    except Exception as exc:
        _fail(exc)
    _print(report, as_json=as_json)
    if not report["ok"]:
        raise SystemExit(1)


def cmd_cache_export(directory: str = "", dest: str = "", as_json: bool = False):
    from b3_core.cache import cache_export
    from b3_core.core.run import cache_dir

    if not dest:
        raise SystemExit("cache export needs --dest")
    report = cache_export(cache_dir(directory), dest)
    _print(report, as_json=as_json)


def cmd_cache_import(
    bundle: str,
    directory: str = "",
    overwrite: bool = False,
    as_json: bool = False,
):
    from b3_core.cache import cache_import
    from b3_core.core.run import cache_dir

    report = cache_import(bundle, cache_dir(directory), overwrite=overwrite)
    _print(report, as_json=as_json)


def cmd_report_build(
    path: str,
    out: str = "",
    template: str = "",
    as_json: bool = False,
):
    from b3_core.report.build import build_report

    try:
        written = build_report(path, out or "report", template=template or None)
    except Exception as exc:
        _fail(exc)
    _print(written, as_json=as_json)


def cmd_kerfs_map(
    path: str,
    magnitude: float = 0.001,
    unit: str = "1/mm",
    out: str = "",
    backend: str = "fenicsx",
    as_json: bool = False,
):
    from b3_core.sweep.kerf_map import kerf_map
    from b3_core.units import to_per_mm

    try:
        report = kerf_map(path, to_per_mm(magnitude, unit), backend=backend)
    except Exception as exc:
        _fail(exc)
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"curvature unit {unit.strip()} stored as 1/mm")
    _print(
        {key: report[key] for key in report if key != "grid"}
        if not as_json
        else report,
        as_json=True,
    )


def cmd_diagnose_bisect(
    path: str,
    kx: float = 5e-5,
    ky: float = 0.0,
    unit: str = "1/mm",
    backend: str = "numpy",
    out: str = "",
    as_json: bool = False,
):
    from b3_core.checks.bisect import bisect
    from b3_core.units import to_per_mm

    try:
        report = bisect(
            path,
            kx=to_per_mm(kx, unit),
            ky=to_per_mm(ky, unit),
            backend=backend,
        )
    except Exception as exc:
        _fail(exc)
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"curvature unit {unit.strip()} stored as 1/mm")
    _print(report, as_json=as_json)


def cmd_viz_figs(path: str, out: str = "", as_json: bool = False):
    from b3_core.viz.figs import write_figures

    try:
        written = write_figures(path, out or "figs")
    except Exception as exc:
        _fail(exc)
    _print(written, as_json=as_json)


def _sheet_provenance(stored: object, cache: str) -> dict:
    """Keep the solve provenance and record the cache directory the CLI was given."""
    provenance = dict(stored) if isinstance(stored, dict) else {}
    if cache:
        provenance["cache"] = cache
    return provenance


def write_full_datasheet(
    case: str,
    output: str,
    *,
    prose: str,
    grid: str,
    draft: bool,
    cache: str,
) -> str:
    """Build data.json and a PDF. Refuses a missing stamp unless draft."""
    import json as _json

    from b3_core.report.prose import Prose
    from b3_core.report.sheet import assemble_data, generate_full

    prose_body = (
        Prose.model_validate(_json.loads(Path(prose).read_text(encoding="utf-8")))
        if prose
        else Prose(
            title=Path(case or grid or "datasheet").stem,
            summary="Draft datasheet.",
            intended_use="Internal review.",
            limitations="Not an accepted material card.",
        )
    )
    grid_payload = None
    if grid:
        grid_payload = _json.loads(Path(grid).read_text(encoding="utf-8"))
    rows = (grid_payload or {}).get("rows") or []
    flat = next(
        (
            row
            for row in rows
            if float(row.get("kx") or 0) == 0 and float(row.get("ky") or 0) == 0
        ),
        {},
    )
    properties = {
        key: flat[key]
        for key in ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "rho_infused", "resin_vf")
        if key in flat and flat[key] is not None
    }
    dest = Path(output) if output else Path("datasheet")
    if dest.suffix == ".pdf":
        folder = dest.parent / dest.stem
        folder.mkdir(parents=True, exist_ok=True)
    else:
        folder = dest
        folder.mkdir(parents=True, exist_ok=True)
    figures = {}
    if rows:
        from b3_core.viz.figs import write_figures

        figures = {
            name: Path(path).name
            for name, path in write_figures(grid_payload, folder).items()
        }
    data = assemble_data(
        prose=prose_body,
        properties=properties,
        convention=(grid_payload or {}).get("convention") or [],
        provenance=_sheet_provenance(flat.get("provenance"), cache),
        acceptance=(grid_payload or {}).get("acceptance"),
        figures=figures,
        kx=float(flat.get("kx") or 0.0),
        ky=float(flat.get("ky") or 0.0),
        grid_rows=rows,
        draft=draft,
    )
    pdf = generate_full(folder, data, draft=draft, compile=True)
    return str(pdf)


def cmd_viz_datasheet_sweep(
    path: str,
    output: str = "",
    prose: str = "",
    draft: bool = False,
    cache: str = "",
):
    try:
        pdf = write_full_datasheet(
            "", output, prose=prose, grid=path, draft=draft, cache=cache
        )
    except Exception as exc:
        _fail(exc)
    print(f"Wrote {pdf}")
