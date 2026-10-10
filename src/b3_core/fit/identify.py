"""Cheap sensitivity: two solves per free parameter."""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_core.fit.params import apply
from b3_core.fit.residuals import residuals
from b3_core.fit.spec import IdentifiabilityReport, ParamSet, TargetSet
from b3_core.loaders import normalize_case

# Earlier names are kept when two parameters are collinear.
_PREFERENCE = (
    "halo_cell_size",
    "foam_Ez_scale",
    "foam_G_scale",
    "foam_E_scale",
    "kerf_width",
    "resin_E",
)


def _z(record: Any, targets: TargetSet) -> np.ndarray:
    report = residuals(record, targets)
    return np.asarray(
        [
            float(row["z"])
            for row in report.rows
            if row["used"] and row["z"] is not None
        ],
        dtype=float,
    )


def _suggest_fix(left: str, right: str) -> str:
    order = {name: index for index, name in enumerate(_PREFERENCE)}
    left_rank = order.get(left, len(order))
    right_rank = order.get(right, len(order))
    return right if right_rank >= left_rank else left


def sensitivity(
    case: Any,
    paramset: ParamSet,
    targets: TargetSet,
    x: Any = None,
    *,
    backend: str = "fenicsx",
    cache: Any = None,
    step: float = 0.02,
    solve: Any = None,
) -> IdentifiabilityReport:
    """Central differences in unit space. ``step`` is the unit-space half-width."""
    case_in, _workdir = normalize_case(case)
    free = paramset.free()
    if x is None:
        current = np.asarray([param.value for param in free], dtype=float)
    else:
        current = np.asarray(x, dtype=float)
    runner = solve
    if runner is None:
        from b3_core.api import run_case

        def runner(point, *, backend, cache):  # noqa: A001
            return run_case(point, backend=backend, cache=cache)

    def at(vector: np.ndarray) -> np.ndarray:
        return _z(
            runner(apply(case_in, paramset, vector), backend=backend, cache=cache),
            targets,
        )

    if not free:
        return IdentifiabilityReport(
            jacobian=[],
            singular_values=[],
            condition=1.0,
            correlations=[],
            weak_params=[],
            collinear=[],
            identifiable=True,
        )
    centre = paramset.to_unit(current)
    columns = []
    for index in range(len(free)):
        plus = centre.copy()
        minus = centre.copy()
        plus[index] = min(1.0, centre[index] + float(step))
        minus[index] = max(-1.0, centre[index] - float(step))
        span = float(plus[index] - minus[index])
        if span == 0.0:
            columns.append(np.zeros_like(at(current)))
            continue
        columns.append(
            (at(paramset.from_unit(plus)) - at(paramset.from_unit(minus))) / span
        )
    jacobian = np.column_stack(columns) if columns else np.zeros((0, 0))
    singular = (
        np.linalg.svd(jacobian, compute_uv=False) if jacobian.size else np.zeros(0)
    )
    condition = (
        float(singular[0] / singular[-1])
        if len(singular) and singular[-1] > 0.0
        else float("inf")
    )
    norms = np.linalg.norm(jacobian, axis=0) if jacobian.size else np.zeros(len(free))
    peak = float(norms.max()) if len(norms) else 0.0
    weak = [
        param.name
        for param, norm in zip(free, norms, strict=True)
        if peak == 0.0 or float(norm) < 0.05 * peak
    ]
    safe = np.where(norms > 0.0, norms, 1.0)
    cosine = (
        (jacobian / safe).T @ (jacobian / safe) if jacobian.size else np.eye(len(free))
    )
    collinear = []
    for left in range(len(free)):
        for right in range(left + 1, len(free)):
            corr = float(cosine[left, right])
            if abs(corr) > 0.95:
                names = (free[left].name, free[right].name)
                collinear.append(
                    {
                        "params": list(names),
                        "corr": corr,
                        "suggest_fix": _suggest_fix(*names),
                    }
                )
    n_used = int(jacobian.shape[0]) if jacobian.ndim == 2 else 0
    identifiable = not weak and not collinear and len(free) <= n_used
    return IdentifiabilityReport(
        jacobian=jacobian.tolist(),
        singular_values=[float(value) for value in singular],
        condition=condition,
        correlations=np.asarray(cosine, dtype=float).tolist(),
        weak_params=weak,
        collinear=collinear,
        identifiable=identifiable,
    )
