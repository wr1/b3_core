"""Fit schemas. Nothing here decides which parameters are free."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from b3_core.models import Curvature

Quantity = Literal[
    "Ex",
    "Ey",
    "Ez",
    "Gxy",
    "Gxz",
    "Gyz",
    "nuxy",
    "nuxz",
    "nuyz",
    "rho_infused",
    "resin_uptake_kg_m2",
    "resin_vf",
    "effective_resin_vf",
]


class Prior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["normal", "lognormal", "uniform"] = "uniform"
    mean: float | None = None
    sd: float | None = None


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    transform: Literal["abs", "scale", "log"] = "abs"
    value: float
    lower: float | None = None
    upper: float | None = None
    fixed: bool = False
    prior: Prior | None = None
    unit: str = ""
    sd: float | None = None


class ParamSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: list[ParamSpec]

    def free(self) -> list[ParamSpec]:
        return [param for param in self.params if not param.fixed]

    def to_unit(self, x: Any) -> np.ndarray:
        """Free-parameter values → [-1, 1] across each parameter's bounds."""
        mapped = []
        for param, value in zip(self.free(), x, strict=True):
            lo = param.lower
            hi = param.upper
            if lo is None or hi is None or hi == lo:
                mapped.append(0.0)
            else:
                mapped.append(2.0 * (float(value) - lo) / (hi - lo) - 1.0)
        return np.asarray(mapped, dtype=float)

    def from_unit(self, u: Any) -> np.ndarray:
        """[-1, 1] → free-parameter values."""
        mapped = []
        for param, value in zip(self.free(), u, strict=True):
            lo = param.lower
            hi = param.upper
            if lo is None or hi is None:
                mapped.append(float(param.value))
            else:
                mapped.append(lo + 0.5 * (float(value) + 1.0) * (hi - lo))
        return np.asarray(mapped, dtype=float)


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: Quantity
    value: float | None
    sd: float | None = None
    rel_sd: float | None = None
    weight: float = 1.0
    axis_map: dict[str, str] = Field(default_factory=dict)
    basis: Literal["infused", "neat"] = "infused"
    method: str = ""
    source: str = ""


class TargetSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[Target]
    curvature: Curvature = Field(default_factory=Curvature)


class FitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["b3_core.fit/1"] = Field("b3_core.fit/1", alias="schema")
    case: str | dict
    params: ParamSet
    targets: TargetSet
    backend: str = "fenicsx"
    cache: str | None = ".b3cache"
    stages: dict = Field(
        default_factory=lambda: {
            "halo_from_mass": True,
            "design": {"method": "ccd", "n": None, "seed": 0},
            "rsm": {"log": True, "ridge": 1e-2},
            "refine": {"tol_sigma": 1.0, "max_solves": 30, "jac": "fd"},
            "identify": {"step": 0.02},
        }
    )
    workers: int = 1


class ResidualReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict]
    chi2: float
    n_used: int
    max_abs_z: float


class IdentifiabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jacobian: list[list[float]]
    singular_values: list[float]
    condition: float
    correlations: list[list[float]]
    weak_params: list[str]
    collinear: list[dict]
    identifiable: bool


class HaloEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    param: str
    value: float
    lower: float
    upper: float
    feasible: bool
    target_rho: float | None = None
    model_rho: float | None = None


class DesignTable(BaseModel):
    """One row per design point. Failures stay in ``rows`` with ``ok`` false."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[dict]


class FitResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["b3_core.fit-result/1"] = Field(
        "b3_core.fit-result/1", alias="schema"
    )
    status: Literal[
        "converged",
        "rsm_only",
        "not_identifiable",
        "infeasible",
        "no_free_params",
        "error",
    ]
    params: list[ParamSpec]
    residuals: ResidualReport
    identifiability: IdentifiabilityReport | None = None
    rsm: dict | None = None
    solves: dict
    calibrated_case: dict
    provenance: dict
    messages: list[str] = Field(default_factory=list)
