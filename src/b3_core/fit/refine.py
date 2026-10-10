"""Bounded fit on true solves. ``jac='fd'`` is a finite-difference Jacobian."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from b3_core import __version__
from b3_core.fit.params import apply
from b3_core.fit.residuals import residuals
from b3_core.fit.spec import FitResult, ParamSet, TargetSet
from b3_core.hashing import case_hash
from b3_core.loaders import normalize_case
from b3_core.solvers.stamps import stamp_for


def _solve(case: Any, *, backend: str, cache: Any) -> Any:
    from b3_core.api import run_case

    return run_case(case, backend=backend, cache=cache)


def _prior_rows(params: ParamSet, x: np.ndarray) -> list[float]:
    rows = []
    for param, value in zip(params.free(), x, strict=True):
        prior = param.prior
        if prior is None or prior.kind == "uniform" or prior.sd in (None, 0.0):
            continue
        if prior.mean is None:
            continue
        if prior.kind == "lognormal":
            rows.append(
                (float(np.log(value)) - float(np.log(prior.mean))) / float(prior.sd)
            )
        else:
            rows.append((float(value) - float(prior.mean)) / float(prior.sd))
    return rows


def _vector(
    record: Any, targets: TargetSet, params: ParamSet, x: np.ndarray
) -> np.ndarray:
    report = residuals(record, targets)
    values = [
        float(row["z"]) for row in report.rows if row["used"] and row["z"] is not None
    ]
    values.extend(_prior_rows(params, x))
    if not values:
        return np.zeros(1)
    return np.asarray(values, dtype=float)


def _mark_calibrated(case: Any) -> Any:
    core = case.core.model_copy(update={"source": "calibrated"})
    resin = case.resin.model_copy(update={"source": "calibrated"})
    return case.model_copy(update={"core": core, "resin": resin})


def _provenance(case: Any, backend: str) -> dict[str, Any]:
    case_in, _workdir = normalize_case(case)
    return {
        "backend": backend,
        "b3_core_version": __version__,
        "case_hash": case_hash(case_in, backend=backend),
        "solver_stamp": stamp_for(backend),
    }


mark_calibrated = _mark_calibrated
fit_provenance = _provenance


def refine(
    case: Any,
    paramset: ParamSet,
    targets: TargetSet,
    x0: Any = None,
    *,
    backend: str = "fenicsx",
    cache: Any = None,
    tol_sigma: float = 1.0,
    max_solves: int = 30,
    jac: str = "fd",
    solve: Any = None,
) -> FitResult:
    """Trust-region fit. Stops at ``tol_sigma`` or ``max_solves`` evaluations."""
    if jac != "fd":
        raise ValueError("refine minimum supports jac='fd' only")
    case_in, _workdir = normalize_case(case)
    free = paramset.free()
    runner = solve or _solve
    solves = {"true": 0, "cache_hits": 0, "wall_s": 0.0}

    def evaluate(vector: np.ndarray) -> Any:
        solves["true"] += 1
        return runner(apply(case_in, paramset, vector), backend=backend, cache=cache)

    if not free:
        record = evaluate(np.zeros(0))
        report = residuals(record, targets)
        calibrated = _mark_calibrated(apply(case_in, paramset))
        return FitResult(
            status="no_free_params",
            params=list(paramset.params),
            residuals=report,
            solves=solves,
            calibrated_case=calibrated.model_dump(mode="json"),
            provenance=_provenance(case_in, backend),
            messages=["no free parameters; the case is predicted, not calibrated"],
        )

    start = np.asarray(
        [param.value for param in free] if x0 is None else x0, dtype=float
    )
    lower = np.asarray(
        [-np.inf if param.lower is None else param.lower for param in free], dtype=float
    )
    upper = np.asarray(
        [np.inf if param.upper is None else param.upper for param in free], dtype=float
    )

    holder: dict[str, Any] = {}

    def fun(vector: np.ndarray) -> np.ndarray:
        record = evaluate(vector)
        holder["record"] = record
        holder["x"] = np.asarray(vector, dtype=float)
        return _vector(record, targets, paramset, vector)

    fitted = least_squares(
        fun,
        start,
        jac="2-point",
        bounds=(lower, upper),
        max_nfev=int(max_solves),
        ftol=1e-10,
        xtol=1e-10,
    )
    if "x" not in holder or not np.allclose(holder["x"], fitted.x):
        holder["record"] = evaluate(fitted.x)
    report = residuals(holder["record"], targets)
    status = "converged" if report.max_abs_z <= float(tol_sigma) else "error"
    sds = _parameter_sd(fitted.jac)
    final_params = []
    cursor = 0
    for param in paramset.params:
        if param.fixed:
            final_params.append(param)
            continue
        sd = None if cursor >= len(sds) else sds[cursor]
        final_params.append(
            param.model_copy(update={"value": float(fitted.x[cursor]), "sd": sd})
        )
        cursor += 1
    calibrated = _mark_calibrated(apply(case_in, paramset, fitted.x))
    messages = []
    if status != "converged":
        messages.append(
            f"max |z| is {report.max_abs_z:.3g}, above tol_sigma {tol_sigma:g}"
        )
    return FitResult(
        status=status,
        params=final_params,
        residuals=report,
        solves=solves,
        calibrated_case=calibrated.model_dump(mode="json"),
        provenance=_provenance(case_in, backend),
        messages=messages,
    )


def _parameter_sd(jac: Any) -> list[float | None]:
    matrix = np.asarray(jac, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return []
    gram = matrix.T @ matrix
    try:
        covariance = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return [None] * matrix.shape[1]
    diag = np.diag(covariance)
    return [None if value < 0 else float(np.sqrt(value)) for value in diag]
