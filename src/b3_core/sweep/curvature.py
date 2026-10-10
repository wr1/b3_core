"""Curvature grid for a datasheet. One case, its own halo, explicit backend.

The training grid in :mod:`b3_core.sweep.curvature_grid` keeps its cell-size
list and its own backend rule. This function is the one ``sweep grid`` calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from b3_core.loaders import normalize_case
from b3_core.models import CaseInput, Curvature
from b3_core.units import axis_to_per_mm, to_per_mm

Solve = Callable[..., Any]


class InertAxisError(ValueError):
    """A curvature range acts on an axis that has no grooves."""


def _stored_axis(values: Sequence[float] | str, unit: str) -> list[float]:
    if isinstance(values, str):
        return axis_to_per_mm(values, unit)
    return [to_per_mm(float(value), unit) for value in values]


def _with_zero(values: list[float]) -> list[float]:
    if any(value == 0.0 for value in values):
        return list(values)
    return [0.0, *values]


def _active(values: list[float]) -> bool:
    return any(value != 0.0 for value in values)


def _apply_cell_size(case: CaseInput, cell_size: float | None) -> CaseInput:
    core = case.core.model_copy()
    if cell_size is None or float(cell_size) <= 0.0:
        core.cell_size = None
    else:
        core.cell_size = float(cell_size)
    return case.model_copy(update={"core": core})


def _point_case(case: CaseInput, kx: float, ky: float) -> CaseInput:
    return case.model_copy(update={"curvature": Curvature(kx=float(kx), ky=float(ky))})


def _guard(case: CaseInput, kx: list[float], ky: list[float], allow: bool) -> dict:
    flags = {"kx_inert": False, "ky_inert": False}
    problems = []
    if _active(kx) and not case.xgr:
        flags["kx_inert"] = True
        problems.append("kx")
    if _active(ky) and not case.ygr:
        flags["ky_inert"] = True
        problems.append("ky")
    if problems and not allow:
        names = " and ".join(problems)
        raise InertAxisError(
            f"{names} varies but that axis has no grooves; "
            "pass allow_inert_axis to keep the range"
        )
    return flags


def _properties(record: Any) -> dict[str, float]:
    props = record.result.properties
    geometry = record.geometry or {}

    def number(key: str, default: float = 0.0) -> float:
        if key in props:
            return float(props[key])
        if key in geometry:
            return float(geometry[key])
        return default

    return {
        "Ex": number("Ex", number("Exx")),
        "Ey": number("Ey", number("Eyy")),
        "Ez": number("Ez", number("Ezz")),
        "Gxy": number("Gxy"),
        "Gxz": number("Gxz"),
        "Gyz": number("Gyz"),
        "nuxy": number("nuxy"),
        "nuxz": number("nuxz"),
        "nuyz": number("nuyz"),
        "rho_infused": number("rho_infused"),
        "resin_vf": number("resin_vf"),
    }


def _row_from_record(
    record: Any, *, kx: float, ky: float, cell_size: float, cache_hit: bool
) -> dict[str, Any]:
    provenance = None
    if getattr(record, "provenance", None) is not None:
        provenance = record.provenance.model_dump(mode="json")
    diagnostics = getattr(record, "diagnostics", None) or {}
    row = {
        "kx": float(kx),
        "ky": float(ky),
        "cell_size": float(cell_size),
        "ok": True,
        "error": "",
        "backend": str(record.result.backend),
        "case_hash": str(record.case_hash),
        "cache_hit": bool(cache_hit),
        "provenance": provenance,
        "diagnostics": {
            "checks": list(diagnostics.get("checks", [])),
            "ties": list(diagnostics.get("ties", [])),
        },
    }
    row.update(_properties(record))
    return row


def _failed_row(kx: float, ky: float, cell_size: float, exc: BaseException) -> dict:
    return {
        "kx": float(kx),
        "ky": float(ky),
        "cell_size": float(cell_size),
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "backend": "",
        "case_hash": "",
        "cache_hit": False,
        "provenance": None,
        "diagnostics": {"checks": [], "ties": []},
        "Ex": None,
        "Ey": None,
        "Ez": None,
        "Gxy": None,
        "Gxz": None,
        "Gyz": None,
        "nuxy": None,
        "nuxz": None,
        "nuyz": None,
        "rho_infused": None,
        "resin_vf": None,
    }


class _Solved:
    """Run record plus the cache bit. The record itself stays a pydantic model."""

    def __init__(self, record: Any, cache_hit: bool) -> None:
        self.result = record.result
        self.case_hash = record.case_hash
        self.provenance = record.provenance
        self.diagnostics = record.diagnostics
        self.geometry = record.geometry
        self.meta_cache_hit = cache_hit


def _default_solve(case: CaseInput, *, backend: str, cache: Any) -> _Solved:
    from b3_core.api import run_case

    meta: dict[str, Any] = {}
    record = run_case(case, backend=backend, cache=cache, meta=meta)
    return _Solved(record, bool(meta.get("cache_hit")))


def _fit_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    raw = Path(path).read_bytes()
    import hashlib

    payload = json.loads(raw)
    residuals = payload.get("residuals") or {}
    return {
        "fit_hash": hashlib.sha256(raw).hexdigest(),
        "status": payload.get("status"),
        "residuals": {
            "chi2": residuals.get("chi2"),
            "max_abs_z": residuals.get("max_abs_z"),
            "n_used": residuals.get("n_used"),
        },
    }


def sweep_curvature(
    case: Any,
    kx: Sequence[float] | str,
    ky: Sequence[float] | str,
    *,
    unit: str = "1/mm",
    cache: Any = None,
    backend: str = "fenicsx",
    workers: int = 1,
    cell_sizes: Sequence[float] | None = None,
    allow_inert_axis: bool = False,
    dry_run: bool = False,
    fit_result: str | None = None,
    solve: Solve | None = None,
) -> dict[str, Any]:
    """Solve the curvature product. Failures are rows, not exceptions.

    ``kx`` and ``ky`` are sampled in ``unit`` and stored as 1/mm. The flat
    point and both zero lines are added when the caller omitted them.
    ``cell_sizes`` replaces the case halo; the default keeps it.
    ``workers`` above 1 uses a process pool. Each child sets
    ``OMP_NUM_THREADS=1``. The solve callable must be a module-level function.
    """
    case_in, _workdir = normalize_case(case)
    kx_values = _with_zero(_stored_axis(kx, unit))
    ky_values = _with_zero(_stored_axis(ky, unit))
    flags = _guard(case_in, kx_values, ky_values, allow_inert_axis)
    sizes: list[float | None]
    if cell_sizes is None:
        raw = case_in.core.cell_size
        sizes = [None if raw is None else float(raw)]
    else:
        sizes = [float(value) for value in cell_sizes]
    points = [(x_value, y_value) for x_value in kx_values for y_value in ky_values]
    n_solves = len(points) * len(sizes)
    manifest: dict[str, Any] = {
        "unit": unit.strip(),
        "backend": backend,
        "workers": int(workers),
        "dry_run": bool(dry_run),
        "n_solves": n_solves,
        "kx": kx_values,
        "ky": ky_values,
        "cell_sizes": [0.0 if value is None else float(value) for value in sizes],
        "ky_inert": flags["ky_inert"],
        "kx_inert": flags["kx_inert"],
        "fit": _fit_summary(fit_result),
    }
    if dry_run:
        return {"manifest": manifest, "rows": []}
    if int(workers) < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    runner = solve or _default_solve
    cache_path = str(getattr(cache, "root", "") or "")
    jobs = []
    for cell_size in sizes:
        sized = _apply_cell_size(case_in, cell_size)
        stored_size = 0.0 if cell_size is None else float(cell_size)
        for x_value, y_value in points:
            point = _point_case(sized, x_value, y_value)
            jobs.append(
                (
                    point.model_dump(mode="json"),
                    backend,
                    cache_path,
                    runner,
                    x_value,
                    y_value,
                    stored_size,
                )
            )
    from b3_core.sweep.parallel import map_workers

    rows = map_workers(_run_point, jobs, int(workers))
    manifest["n_failed"] = sum(1 for row in rows if not row["ok"])
    return {"manifest": manifest, "rows": rows}


def _run_point(job: tuple) -> dict[str, Any]:
    payload, backend, cache_path, runner, x_value, y_value, stored_size = job
    try:
        from b3_core.cache import DiskCache
        from b3_core.models import CaseInput

        cache = DiskCache(cache_path) if cache_path else None
        point = CaseInput.model_validate(payload)
        record = runner(point, backend=backend, cache=cache)
        hit = bool(getattr(record, "meta_cache_hit", False))
        return _row_from_record(
            record,
            kx=x_value,
            ky=y_value,
            cell_size=stored_size,
            cache_hit=hit,
        )
    except Exception as exc:
        return _failed_row(x_value, y_value, stored_size, exc)


def _flat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        item = dict(row)
        item["provenance"] = json.dumps(row.get("provenance"))
        item["diagnostics"] = json.dumps(row.get("diagnostics"))
        flat.append(item)
    return flat


def write_grid(result: dict[str, Any], directory: str | Path) -> Path:
    """Write ``grid.csv``, ``manifest.json``, and ``grid.parquet`` when it can."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(_flat_rows(result["rows"]))
    frame.to_csv(dest / "grid.csv", index=False)
    manifest = dict(result["manifest"])
    try:
        frame.to_parquet(dest / "grid.parquet", index=False)
        manifest["parquet"] = True
    except (ImportError, ValueError) as exc:
        manifest["parquet"] = False
        manifest["parquet_error"] = str(exc)
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dest
