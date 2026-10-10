"""Quadratic response surface in unit space, and a solve on that surface."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import least_squares

from b3_core.fit.residuals import model_quantity
from b3_core.fit.spec import DesignTable, ParamSet, TargetSet
from b3_core.physics_surrogate import ridge_lstsq


class ResponseSurface(BaseModel):
    """Full quadratic. ``predict`` returns the property, not the log."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    names: list[str]
    terms: list[str]
    coefs: dict[str, list[float]]
    log: bool = True
    ridge: float = 1e-2
    loo_cv: dict[str, float | None] = {}

    def _phi(self, unit: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(np.asarray(unit, dtype=float))
        cols = [1.0]
        labels_known = len(self.terms) > 0
        for value in u:
            cols.append(float(value))
        for value in u:
            cols.append(float(value) ** 2)
        for i, left in enumerate(u):
            for right in u[i + 1 :]:
                cols.append(float(left) * float(right))
        phi = np.asarray(cols, dtype=float)
        if labels_known and phi.size != len(self.terms):
            raise ValueError(
                f"unit vector length {u.size} does not match terms {self.terms}"
            )
        return phi

    def predict(self, unit: Any) -> dict[str, float]:
        phi = self._phi(np.asarray(unit, dtype=float))
        out: dict[str, float] = {}
        for name, coef in self.coefs.items():
            raw = float(phi @ np.asarray(coef, dtype=float))
            out[name] = float(np.exp(raw)) if self.log else raw
        return out

    def grad(self, unit: Any) -> dict[str, list[float]]:
        """Derivative of the predicted property w.r.t. each unit coordinate."""
        u = np.atleast_1d(np.asarray(unit, dtype=float))
        step = 1e-4
        base = self.predict(u)
        out: dict[str, list[float]] = {name: [] for name in base}
        for axis in range(u.size):
            bumped = u.copy()
            bumped[axis] += step
            plus = self.predict(bumped)
            for name in base:
                out[name].append((plus[name] - base[name]) / step)
        return out


def _terms(dimension: int) -> list[str]:
    names = ["1"]
    names.extend(f"u{i}" for i in range(dimension))
    names.extend(f"u{i}^2" for i in range(dimension))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            names.append(f"u{i}*u{j}")
    return names


def _phi_row(unit: list[float] | np.ndarray) -> np.ndarray:
    u = np.atleast_1d(np.asarray(unit, dtype=float))
    cols = [1.0]
    cols.extend(float(value) for value in u)
    cols.extend(float(value) ** 2 for value in u)
    for i, left in enumerate(u):
        for right in u[i + 1 :]:
            cols.append(float(left) * float(right))
    return np.asarray(cols, dtype=float)


def _measured(targets: TargetSet) -> list[tuple[str, float, float]]:
    rows = []
    for target in targets.targets:
        if target.value is None or target.basis != "infused":
            continue
        name = model_quantity(target)
        if target.sd is not None and target.sd > 0:
            sd = float(target.sd)
        else:
            rel = 0.05 if target.rel_sd is None else float(target.rel_sd)
            sd = abs(float(target.value)) * rel
        if sd <= 0:
            continue
        rows.append((name, float(target.value), sd))
    return rows


def fit_quadratic(
    table: DesignTable,
    targets: TargetSet,
    *,
    log: bool = True,
    ridge: float = 1e-2,
) -> ResponseSurface:
    """Ridge quadratic on the successful design rows."""
    good = [
        row for row in table.rows if row.get("ok") and row.get("x_unit") is not None
    ]
    if not good:
        raise ValueError("response surface needs at least one successful design row")
    dimension = len(good[0]["x_unit"])
    phi = np.vstack([_phi_row(row["x_unit"]) for row in good])
    wanted = [name for name, _value, _sd in _measured(targets)]
    present = set(good[0]["properties"])
    names = [name for name in wanted if name in present] or [
        name for name in ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz") if name in present
    ]
    coefs: dict[str, list[float]] = {}
    loo: dict[str, float | None] = {}
    for name in names:
        observed = np.array(
            [float(row["properties"][name]) for row in good], dtype=float
        )
        target = np.log(np.clip(observed, 1e-30, None)) if log else observed
        coefs[name] = [float(value) for value in ridge_lstsq(phi, target, lam=ridge)]
        loo[name] = _loo(phi, target, observed, log=log, ridge=ridge)
    return ResponseSurface(
        names=names,
        terms=_terms(dimension),
        coefs=coefs,
        log=log,
        ridge=float(ridge),
        loo_cv=loo,
    )


def _loo(
    phi: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    *,
    log: bool,
    ridge: float,
) -> float | None:
    n = len(observed)
    if n < 3:
        return None
    errors = []
    for index in range(n):
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        coef = ridge_lstsq(phi[mask], target[mask], lam=ridge)
        raw = float(phi[index] @ coef)
        pred = float(np.exp(raw)) if log else raw
        scale = abs(float(observed[index])) or 1.0
        errors.append(abs(pred - float(observed[index])) / scale)
    return float(np.mean(errors))


def _prior_rows(paramset: ParamSet, physical: np.ndarray) -> list[float]:
    rows = []
    for param, value in zip(paramset.free(), physical, strict=True):
        prior = param.prior
        if (
            prior is None
            or prior.kind == "uniform"
            or not prior.sd
            or prior.mean is None
        ):
            continue
        if prior.kind == "lognormal":
            rows.append((float(np.log(value)) - float(np.log(prior.mean))) / prior.sd)
        else:
            rows.append((float(value) - float(prior.mean)) / float(prior.sd))
    return rows


def solve_rsm(
    surface: ResponseSurface, targets: TargetSet, paramset: ParamSet
) -> np.ndarray:
    """Least squares on the surface. Returns free-parameter values, not unit space."""
    measured = _measured(targets)
    dimension = len(paramset.free())
    if dimension == 0:
        return np.zeros(0)

    def fun(unit: np.ndarray) -> np.ndarray:
        predicted = surface.predict(unit)
        residual = []
        for name, value, sd in measured:
            if name not in predicted:
                continue
            residual.append((predicted[name] - value) / sd)
        residual.extend(_prior_rows(paramset, paramset.from_unit(unit)))
        if not residual:
            residual = [0.0]
        return np.asarray(residual, dtype=float)

    fitted = least_squares(
        fun,
        np.zeros(dimension),
        bounds=(-np.ones(dimension), np.ones(dimension)),
        ftol=1e-12,
        xtol=1e-12,
    )
    return paramset.from_unit(fitted.x)
