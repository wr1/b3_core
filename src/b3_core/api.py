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
from b3_core.models import CaseInput
from b3_core.result import CoreResult, ResultBlock, RunRecord
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
        input=case.model_dump(mode="json"),
        geometry=prep.geometry.as_dict(),
        result=ResultBlock(
            backend=resolved,
            stiffness=np.asarray(solved.stiffness, dtype=float).tolist(),
            properties=properties,
            extras={k: float(v) for k, v in dict(solved.extras).items()},
        ),
        validation=validation,
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


def run_case(
    case: Any,
    *,
    backend: str | None = None,
    cache: Cache | None = None,
) -> RunRecord:
    """Solve a case and return a namespaced record. Does not write a run file."""
    from b3_core.pipeline import resolve_backend, run_pipeline

    case_in, _workdir = normalize_case(case)
    resolved = resolve_backend(case_in, backend)
    key = case_hash(case_in, backend=resolved)
    store: Cache = cache if cache is not None else NullCache()
    short = key[:12]
    hit = store.get(key)
    if hit is not None:
        logger.info("cache hit %s (%s)", short, hit.result.backend)
        return hit
    logger.info("cache miss %s, solving with %s", short, resolved)
    prep, solved, resolved = run_pipeline(case_in, backend=resolved)
    validation = _maybe_validate(case_in, prep, solved, resolved)
    record = _record_from(case_in, prep, solved, resolved, key, validation)
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


def homogenize(
    case: Any,
    *,
    name: str | None = None,
    backend: str | None = None,
    cache: Cache | None = None,
    write: bool = False,
    workdir: str | Path | None = None,
) -> CoreResult:
    """Homogenise one case.

    The default call writes no files. Pass ``write=True`` (and ``workdir``
    when ``case`` is not a path) to also store ``run<hash12>.json``.
    """
    record = run_case(case, backend=backend, cache=cache)
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
