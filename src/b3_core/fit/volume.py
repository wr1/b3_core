"""Halo width from infused density or resin uptake. No finite-element solve."""

from __future__ import annotations

from typing import Any

from scipy.optimize import brentq

from b3_core.fit.params import apply
from b3_core.fit.spec import HaloEstimate, ParamSet, ParamSpec
from b3_core.loaders import normalize_case
from b3_core.pipeline import prepare


def _rho(case: Any, cell_size: float) -> float:
    spec = ParamSet(
        params=[
            ParamSpec(
                name="halo_cell_size",
                path="core.cell_size",
                transform="abs",
                value=float(cell_size),
                fixed=True,
            )
        ]
    )
    prepared = prepare(apply(case, spec))
    return float(prepared.geometry.rho_infused)


def _uptake(case: Any, cell_size: float) -> float:
    case_in, _workdir = normalize_case(case)
    thickness_m = float(case_in.thickness) / 1000.0
    return _rho(case, cell_size) * thickness_m


def estimate_halo(
    case: Any,
    *,
    rho_infused: float | None = None,
    uptake_kg_m2: float | None = None,
    sd: float | None = None,
    param: str = "core.cell_size",
    bounds: tuple[float, float] = (0.0, 2.0),
) -> HaloEstimate:
    """Match density or uptake with ``brentq`` on the halo width.

    ``feasible`` is false when the target lies outside the values at the
    two bounds. The returned value is then the closer endpoint.
    """
    if (rho_infused is None) == (uptake_kg_m2 is None):
        raise ValueError("pass exactly one of rho_infused or uptake_kg_m2")
    if param != "core.cell_size":
        raise ValueError("estimate_halo only adjusts core.cell_size")
    lo, hi = float(bounds[0]), float(bounds[1])
    if uptake_kg_m2 is None:
        target = float(rho_infused)  # type: ignore[arg-type]
        predict = _rho
    else:
        target = float(uptake_kg_m2)
        predict = _uptake

    def gap(cell_size: float) -> float:
        return predict(case, cell_size) - target

    low = gap(lo)
    high = gap(hi)
    if low == 0.0:
        value = lo
        feasible = True
    elif high == 0.0:
        value = hi
        feasible = True
    elif low * high < 0.0:
        value = float(brentq(gap, lo, hi))
        feasible = True
    else:
        value = lo if abs(low) <= abs(high) else hi
        feasible = False
    model = predict(case, value)
    band = 0.0 if sd is None else abs(float(sd))
    return HaloEstimate(
        param=param,
        value=float(value),
        lower=max(lo, float(value) - band),
        upper=min(hi, float(value) + band),
        feasible=feasible,
        target_rho=None if rho_infused is None else float(rho_infused),
        model_rho=float(model) if uptake_kg_m2 is None else None,
    )
