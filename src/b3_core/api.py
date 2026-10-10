"""Public homogenisation API. ``homogenize`` writes nothing unless asked."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from b3_core import _version as _version_mod
from b3_core.cache import Cache, NullCache
from b3_core.hashing import case_hash
from b3_core.loaders import normalize_case
from b3_core.models import CaseInput, Curvature
from b3_core.provenance import make_provenance, with_server
from b3_core.result import CoreResult, ResultBlock, RunRecord
from b3_core.solvers.checks import build_diagnostics
from b3_core.solvers.stamps import stamp_for
from b3_core.solvers.validation import validate_against

__version__ = _version_mod.__version__

logger = logging.getLogger("b3_core.cache")


def _version() -> str:
    return __version__


def _record_from(
    case: CaseInput,
    prep: Any,
    solved: Any,
    resolved: str,
    key: str,
    validation: dict | None,
    diagnostics: dict | None = None,
) -> RunRecord:
    properties = {
        key_name: float(value)
        for key_name, value in solved.properties.items()
        if key_name
        in {
            "Ex",
            "Ey",
            "Ez",
            "Gxy",
            "Gxz",
            "Gyz",
            "nuxy",
            "nuxz",
            "nuyx",
            "nuyz",
            "nuzx",
            "nuzy",
        }
    }
    return RunRecord(
        case_hash=key,
        b3_core_version=_version(),
        solver_stamp=stamp_for(resolved),
        input=case.model_dump(mode="json"),
        geometry=prep.geometry.as_dict(),
        result=ResultBlock(
            backend=resolved,
            stiffness=np.asarray(solved.stiffness, dtype=float).tolist(),
            properties=properties,
            extras={k: float(v) for k, v in dict(solved.extras).items()},
        ),
        validation=validation,
        provenance=make_provenance(solver_stamp=stamp_for(resolved)),
        diagnostics=diagnostics
        if diagnostics is not None
        else {"checks": [], "ties": []},
    )


def _maybe_validate(
    case: CaseInput, prep: Any, solved: Any, resolved: str
) -> dict | None:
    if not case.validate_with_ccx or resolved == "ccx":
        return None
    from b3_core.pipeline import solve
    from b3_core.solvers import get_backend

    reference_backend = get_backend("ccx")
    if not reference_backend.is_available():
        logger.warning(
            "validate_with_ccx is set but ccx is not on PATH; skipping the cross-check"
        )
        return None
    reference = solve(prep, reference_backend)
    return validate_against(reference.properties, solved.properties, label=resolved)


def _with_curvature(case: Any, kx: float | None, ky: float | None) -> Any:
    """Replace mould curvature. ``None`` keeps the value already on the case.

    The numbers are curvatures in 1/mm. The mesh turns them into kerf taper.
    """
    if kx is None and ky is None:
        return case
    from b3_core.cases import CoreCase

    case_in, workdir = normalize_case(case)
    current = case_in.curvature
    updated = case_in.model_copy(
        update={
            "curvature": Curvature(
                kx=float(current.kx if kx is None else kx),
                ky=float(current.ky if ky is None else ky),
            )
        }
    )
    if isinstance(case, CoreCase):
        return case.__class__(input=updated, workdir=case.workdir)
    if isinstance(case, (str, Path)):
        return CoreCase(input=updated, workdir=workdir)
    return updated


def _with_pair(case: Any, enabled: bool | None) -> Any:
    """Set ``allow_pair_periodicity`` when the caller passed the flag."""
    if enabled is None:
        return case
    from b3_core.cases import CoreCase

    case_in, workdir = normalize_case(case)
    updated = case_in.model_copy(update={"allow_pair_periodicity": bool(enabled)})
    if isinstance(case, CoreCase):
        return case.__class__(input=updated, workdir=case.workdir)
    if isinstance(case, (str, Path)):
        return CoreCase(input=updated, workdir=workdir)
    return updated


def run_case(
    case: Any,
    *,
    backend: str | None = None,
    cache: Cache | None = None,
    kx: float | None = None,
    ky: float | None = None,
    allow_pair_periodicity: bool | None = None,
    meta: dict[str, Any] | None = None,
) -> RunRecord:
    """Solve a case and return a namespaced record. Does not write a run file.

    ``kx`` and ``ky`` [1/mm] set mould curvature for this solve. They are
    applied under the hood as a root-hinged kerf taper on a flat RVE
    (positive ``kx`` opens a top-mouth x-groove and pinches a bottom-mouth
    one). The record's ``geometry["kerfs"]`` lists root and mouth half-widths.
    """
    from b3_core.pipeline import resolve_backend, run_pipeline

    case = _with_pair(_with_curvature(case, kx, ky), allow_pair_periodicity)
    case_in, _workdir = normalize_case(case)
    resolved = resolve_backend(case_in, backend)
    key = case_hash(case_in, backend=resolved)
    store: Cache = cache if cache is not None else NullCache()
    short = key[:12]
    hit = store.get(key)
    if hit is not None:
        logger.info("cache hit %s (%s)", short, hit.result.backend)
        if meta is not None:
            meta["cache_hit"] = True
        if hit.provenance is not None:
            return hit.model_copy(update={"provenance": with_server(hit.provenance)})
        return hit
    if meta is not None:
        meta["cache_hit"] = False
    logger.info("cache miss %s, solving with %s", short, resolved)
    prep, solved, resolved = run_pipeline(case_in, backend=resolved, details=True)
    validation = _maybe_validate(case_in, prep, solved, resolved)
    diagnostics = build_diagnostics(case_in, prep, solved)
    record = _record_from(case_in, prep, solved, resolved, key, validation, diagnostics)
    store.put(key, record)
    return record


def _write_dir(case: Any, workdir: str | Path | None) -> Path:
    if workdir is not None:
        return Path(workdir)
    from b3_core.cases import CoreCase

    if isinstance(case, (str, Path)):
        return Path(case).resolve().parent
    if isinstance(case, CoreCase) and case.workdir:
        return Path(case.workdir)
    raise ValueError("write=True needs workdir= for non-file cases")


def _write_record(case: Any, record: RunRecord, workdir: str | Path | None) -> Path:
    folder = _write_dir(case, workdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run{record.case_hash[:12]}.json"
    path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    logging.getLogger("b3_core.api").debug("wrote %s", path)
    return path


with_curvature = _with_curvature
write_record = _write_record


def homogenize(
    case: Any,
    *,
    name: str | None = None,
    backend: str | None = None,
    cache: Cache | None = None,
    write: bool = False,
    workdir: str | Path | None = None,
    kx: float | None = None,
    ky: float | None = None,
    allow_pair_periodicity: bool | None = None,
) -> CoreResult:
    """Homogenise one case.

    The default call writes no files. Pass ``write=True`` (and ``workdir``
    when ``case`` is not a path) to also store ``run<hash12>.json``.

    ``kx`` and ``ky`` [1/mm] are mould curvature. They replace the curvature
    on ``case`` and are translated into kerf opening (``hw(z)``) before the
    solve. Positive ``kx`` opens a top-mouth x-groove and pinches a
    bottom-mouth one. ``result.kerfs`` reports each groove's root and mouth
    half-width. Omit both to keep ``case.with_curvature(...)``.

    ``backend="auto"`` on a flat case uses FEniCSx when it is installed,
    otherwise MFEM. A curved case (nonzero ``kx`` or ``ky``) uses FEniCSx
    only. Without that install, ``auto`` raises. ``backend="numpy"`` is the
    explicit curved opt-in and warns about the O(k^2) offset. ``mfem`` on a
    curved case requires ``allow_pair_periodicity=True``. ``ccx`` rejects it.
    """
    record = run_case(
        case,
        backend=backend,
        cache=cache,
        kx=kx,
        ky=ky,
        allow_pair_periodicity=allow_pair_periodicity,
    )
    if write:
        _write_record(case, record, workdir)
    return CoreResult.from_record(record, name=name)


def homogenize_to_disk(
    case: Any,
    *,
    workdir: str | Path | None = None,
    legacy_layout: bool = False,
    **kwargs: Any,
) -> tuple[CoreResult, Path]:
    """``homogenize(..., write=True)``. Returns the result and the written path."""
    record = run_case(
        case,
        backend=kwargs.get("backend"),
        cache=kwargs.get("cache"),
        kx=kwargs.get("kx"),
        ky=kwargs.get("ky"),
    )
    result = CoreResult.from_record(record, name=kwargs.get("name"))
    folder = _write_dir(case, workdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run{record.case_hash[:12]}.json"
    if legacy_layout:
        payload = record.flat()
    else:
        payload = record.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result, path


def stiffness_tensor(case: Any, **kwargs: Any) -> np.ndarray:
    """Effective 6×6 stiffness (Pa, Voigt xx, yy, zz, yz, xz, xy)."""
    return np.asarray(homogenize(case, **kwargs).stiffness, dtype=float)


def sweep(
    cases: Sequence[Any],
    *,
    backend: str | None = None,
    cache: Cache | None = None,
) -> list[RunRecord]:
    """Homogenise each case with one shared cache."""
    store: Cache = cache if cache is not None else NullCache()
    return [run_case(item, backend=backend, cache=store) for item in cases]


def cprop(case_data: Any) -> dict[str, Any]:
    """Deprecated flat writer. Overwrites ``run<hash12>.json`` and never raises on a repeat."""
    warnings.warn(
        "cprop is deprecated; use run_case (pure) or homogenize_to_disk",
        DeprecationWarning,
        stacklevel=2,
    )
    record = run_case(case_data)
    _case, folder = normalize_case(case_data)
    dest = Path(folder or ".")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"run{record.case_hash[:12]}.json"
    path.write_text(json.dumps(record.flat(), indent=4) + "\n", encoding="utf-8")
    return record.flat()
