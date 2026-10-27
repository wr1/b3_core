"""Design points in unit space, and one true solve per point."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import qmc

from b3_core.fit.params import apply
from b3_core.fit.spec import DesignTable, ParamSet
from b3_core.loaders import normalize_case


def _ccd(dimension: int) -> np.ndarray:
    """Face-centred cube: corners, axial points, and the centre. Duplicates drop."""
    if dimension == 0:
        return np.zeros((0, 0))
    corners = np.array(np.meshgrid(*([[-1.0, 1.0]] * dimension), indexing="ij"))
    corners = corners.reshape(dimension, -1).T
    axial = []
    for axis in range(dimension):
        for sign in (-1.0, 1.0):
            row = np.zeros(dimension)
            row[axis] = sign
            axial.append(row)
    rows = np.vstack([corners, np.asarray(axial), np.zeros((1, dimension))])
    kept: list[np.ndarray] = []
    for row in rows:
        if any(np.allclose(row, old, atol=1e-12) for old in kept):
            continue
        kept.append(row)
    return np.vstack(kept)


def _pad(base: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= len(base):
        return base[:n]
    extra = n - len(base)
    dimension = base.shape[1]
    if dimension == 0:
        return base
    sampler = qmc.LatinHypercube(d=dimension, seed=int(seed))
    unit = sampler.random(extra) * 2.0 - 1.0
    return np.vstack([base, unit])


def design(
    paramset: ParamSet,
    *,
    method: str = "ccd",
    n: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Unit-space points for the free parameters only. Shape ``(n, n_free)``."""
    dimension = len(paramset.free())
    name = method.strip().lower()
    if dimension == 0:
        return np.zeros((0, 0))
    if name == "ccd":
        points = _ccd(dimension)
        if n is not None and int(n) > 0:
            points = _pad(points, int(n), seed)
        return points
    count = int(n) if n else max(2 * dimension, 4)
    if name == "sobol":
        # Sobol warns below a power of two; round up and then trim.
        draw = 1 << (count - 1).bit_length()
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
        unit = sampler.random(draw)[:count] * 2.0 - 1.0
        return unit
    if name == "lhs":
        sampler = qmc.LatinHypercube(d=dimension, seed=int(seed))
        return sampler.random(count) * 2.0 - 1.0
    raise ValueError(f"unknown design method {method!r}; use ccd, sobol, or lhs")


def _properties(record: Any) -> dict[str, float]:
    if isinstance(record, dict):
        source = record
    else:
        result = getattr(record, "result", None)
        source = dict(getattr(result, "properties", {}) or {})
        geometry = getattr(record, "geometry", None) or {}
        for key, value in geometry.items():
            source.setdefault(key, value)
    out: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value)
    return out


def _one_bound(job: tuple) -> dict[str, Any]:
    case_payload, paramset, unit_row, backend, cache_path, runner = job
    from b3_core.cache import DiskCache
    from b3_core.models import CaseInput

    physical = paramset.from_unit(unit_row)
    cache = DiskCache(cache_path) if cache_path else None
    case = CaseInput.model_validate(case_payload)
    try:
        record = runner(apply(case, paramset, physical), backend=backend, cache=cache)
        hit = bool(getattr(record, "meta_cache_hit", False))
        return {
            "x_unit": [float(value) for value in unit_row],
            "x": [float(value) for value in physical],
            "ok": True,
            "error": "",
            "properties": _properties(record),
            "cache_hit": hit,
        }
    except Exception as exc:
        return {
            "x_unit": [float(value) for value in unit_row],
            "x": [float(value) for value in np.atleast_1d(physical)],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "properties": {},
            "cache_hit": False,
        }


def evaluate(
    case: Any,
    paramset: ParamSet,
    points: np.ndarray,
    *,
    backend: str = "fenicsx",
    cache: Any = None,
    workers: int = 1,
    solve: Any = None,
) -> DesignTable:
    """One ``run_case`` per design row. A failed row is recorded and kept."""
    from b3_core.sweep.parallel import map_workers

    case_in, _workdir = normalize_case(case)
    columns = [param.name for param in paramset.free()]
    if solve is None:
        from b3_core.api import run_case

        runner = run_case
    else:
        runner = solve
    cache_path = getattr(cache, "root", None)
    cache_path = str(cache_path) if cache_path else ""
    payload = case_in.model_dump(mode="json")
    array = np.atleast_2d(np.asarray(points, dtype=float))
    if array.size == 0:
        array = np.zeros((0, len(columns)))
    jobs = [(payload, paramset, row, backend, cache_path, runner) for row in array]
    rows = map_workers(_one_bound, jobs, workers)
    return DesignTable(columns=columns, rows=rows)
