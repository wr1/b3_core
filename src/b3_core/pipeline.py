"""The only homogenisation pipeline.

``prepare`` is the only caller of ``create_grooved_mesh`` outside ``core/mesh.py``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from b3_core.models import CaseInput
from b3_core.solvers.protocol import (
    BackendCapabilityError,
    SolveRequest,
    SolveResult,
    UnknownBackendError,
)

logger = logging.getLogger("b3_core.pipeline")


@dataclass(frozen=True)
class GeometryReport:
    resin_vf: float
    area_increase: float
    halo_vf: float
    effective_resin_vf: float
    rho_infused: float

    def as_dict(self) -> dict[str, float]:
        return {
            "resin_vf": float(self.resin_vf),
            "area_increase": float(self.area_increase),
            "halo_vf": float(self.halo_vf),
            "effective_resin_vf": float(self.effective_resin_vf),
            "rho_infused": float(self.rho_infused),
        }


@dataclass(frozen=True)
class PreparedCase:
    case: CaseInput
    mesh: Any
    geometry: GeometryReport
    score_field: Any | None


@dataclass(frozen=True)
class _Needs:
    orthotropic: bool
    halo: bool
    face_layer: bool

    def reason(self) -> str:
        parts: list[str] = []
        if self.orthotropic:
            parts.append("orthotropic")
        if self.halo:
            parts.append("halo")
        if not parts and self.face_layer:
            parts.append("face")
        return ", ".join(parts) if parts else "isotropic"

    def unmet_by(self, caps: Any) -> list[str]:
        missing: list[str] = []
        if self.orthotropic and not caps.orthotropic:
            missing.append("orthotropic")
        if self.halo and not caps.halo:
            missing.append("halo")
        if self.face_layer and not caps.face_layer:
            missing.append("face_layer")
        return missing


def _needs(case: CaseInput) -> _Needs:
    from b3_core.core.scoring import halo_reach

    face = case.face is not None and float(case.face.thickness) > 0.0
    return _Needs(
        orthotropic=bool(case.is_orthotropic),
        halo=halo_reach(case) > 0.0,
        face_layer=face,
    )


def resolve_backend(case: CaseInput, requested: str | None = None) -> str:
    """Pick a backend name. ``auto`` logs the choice.

    An explicit backend that cannot do the job warns and falls back to numpy
    in 0.3. That fallback becomes :class:`BackendCapabilityError` in 1.0.
    """
    from b3_core.solvers import get_backend

    name = requested or case.backend
    needs = _needs(case)
    if name == "auto":
        choice = "numpy" if (needs.orthotropic or needs.halo) else "mfem"
        logger.info("backend auto → %s (%s)", choice, needs.reason())
        return choice
    try:
        caps = get_backend(name).capabilities
    except UnknownBackendError:
        raise
    missing = needs.unmet_by(caps)
    if missing:
        warnings.warn(
            f"backend {name!r} cannot handle {', '.join(missing)}; "
            "falling back to numpy (BackendCapabilityError in 1.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "backend %s cannot handle %s; falling back to numpy",
            name,
            ", ".join(missing),
        )
        return "numpy"
    return name


def prepare(case: CaseInput) -> PreparedCase:
    """Mesh, geometric report, and the resin-halo field."""
    from b3_core.core.analysis import geom_analysis
    from b3_core.core.mesh import create_grooved_mesh
    from b3_core.core.scoring import ScoreField, effective_resin_vf, halo_reach

    reach = halo_reach(case)
    kwargs = case.mesh_kwargs()
    kwargs["s_halo"] = reach
    mesh = create_grooved_mesh(**kwargs)
    geom = geom_analysis(mesh)
    field = ScoreField(case.score_dict()) if reach > 0.0 else None
    eff, halo_vf = effective_resin_vf(mesh, field, geom["resin_vf"])
    rho = float(case.core.rho) * (1.0 - eff) + float(case.resin.rho) * eff
    report = GeometryReport(
        resin_vf=float(geom["resin_vf"]),
        area_increase=float(geom["area_increase"]),
        halo_vf=float(halo_vf),
        effective_resin_vf=float(eff),
        rho_infused=float(rho),
    )
    return PreparedCase(case=case, mesh=mesh, geometry=report, score_field=field)


def solve(prep: PreparedCase, backend: Any, *, details: bool = False) -> SolveResult:
    """Run one backend on an already prepared case."""
    case = prep.case
    request = SolveRequest(
        mesh=prep.mesh,
        core=case.core,
        resin=case.resin,
        face=case.face,
        score_field=prep.score_field,
        scoring=case.scoring,
        element_type=case.element_type,
        details=details,
    )
    return backend.solve(request)


def run_pipeline(
    case: CaseInput,
    *,
    backend: str | None = None,
    details: bool = False,
) -> tuple[PreparedCase, SolveResult, str]:
    """Prepare, resolve the backend, and solve. Returns the resolved name."""
    from b3_core.solvers import get_backend

    name = resolve_backend(case, backend)
    prep = prepare(case)
    result = solve(prep, get_backend(name), details=details)
    return prep, result, name


def backend_capabilities(name: str) -> Any:
    """Capabilities of a registered backend. Visualisation uses this, not solvers."""
    from b3_core.solvers import get_backend

    return get_backend(name).capabilities


def solve_result_from_record(record: Any) -> SolveResult:
    """Stiffness view of a cached record. Displacements are not stored."""
    stiffness = np.asarray(record.result.stiffness, dtype=float)
    return SolveResult(
        stiffness=stiffness,
        properties=dict(record.result.properties),
        compliance=np.linalg.inv(stiffness),
        extras=dict(record.result.extras),
    )


def mesh_for(case: CaseInput | dict) -> Any:
    """Grooved mesh for a case or a legacy dict. Used by visualisation."""
    from b3_core.loaders import normalize_case

    model, _workdir = normalize_case(case)
    return prepare(model).mesh


__all__ = [
    "BackendCapabilityError",
    "GeometryReport",
    "PreparedCase",
    "backend_capabilities",
    "mesh_for",
    "solve_result_from_record",
    "prepare",
    "resolve_backend",
    "run_pipeline",
    "solve",
]
