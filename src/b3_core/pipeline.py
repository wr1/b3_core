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
    kerfs: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "resin_vf": float(self.resin_vf),
            "area_increase": float(self.area_increase),
            "halo_vf": float(self.halo_vf),
            "effective_resin_vf": float(self.effective_resin_vf),
            "rho_infused": float(self.rho_infused),
            "kerfs": [dict(row) for row in self.kerfs],
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
    curved: bool = False

    def reason(self) -> str:
        parts: list[str] = []
        if self.orthotropic:
            parts.append("orthotropic")
        if self.halo:
            parts.append("halo")
        if self.curved:
            parts.append("curved")
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
    curved = float(case.curvature.kx) != 0.0 or float(case.curvature.ky) != 0.0
    return _Needs(
        orthotropic=bool(case.is_orthotropic),
        halo=halo_reach(case) > 0.0,
        face_layer=face,
        curved=curved,
    )


def _curved_choice(case: CaseInput, name: str) -> str:
    """Route a case with kx or ky. Pair-based solvers do not run silently."""
    from b3_core.solvers import get_backend

    if name == "auto":
        fenicsx = get_backend("fenicsx")
        if fenicsx.is_available():
            logger.info("backend auto → fenicsx (curved)")
            return "fenicsx"
        raise BackendCapabilityError(
            "a curved case needs FEniCSx interpolated periodicity, and "
            "fenicsx is not installed. Run `b3_core doctor`. Pass "
            "backend='numpy' only to accept the documented O(k^2) offset."
        )
    if name == "numpy":
        warnings.warn(
            "backend 'numpy' on a curved case has a documented O(k^2) offset; "
            "see the curvature tolerance table in the backends reference. "
            "fenicsx is the curved reference",
            UserWarning,
            stacklevel=3,
        )
        return "numpy"
    if name == "mfem":
        if not case.allow_pair_periodicity:
            raise BackendCapabilityError(
                "backend 'mfem' has no native interpolated z periodicity. "
                "Use fenicsx, or set allow_pair_periodicity=True for the "
                "z-prolongation cross-check."
            )
        warnings.warn(
            "backend 'mfem' is running with allow_pair_periodicity; "
            "the z face uses the tensor-grid prolongation",
            UserWarning,
            stacklevel=3,
        )
        return "mfem"
    if name == "ccx":
        raise BackendCapabilityError(
            "backend 'ccx' supports node-pair periodicity only. "
            "A tapered z face is not a CalculiX case."
        )
    try:
        caps = get_backend(name).capabilities
    except UnknownBackendError:
        raise
    if not caps.interpolated_periodicity:
        raise BackendCapabilityError(
            f"backend {name!r} cannot tie a tapered z face "
            "(interpolated_periodicity is false)"
        )
    return name


# Preference order for ``auto`` and capability fallback. FEniCSx leads when it
# is installed: its periodic constraint is the finite-element projection of the
# image, including a face that is not a tensor grid. MFEM is next and covers
# the same constitutive maps. Numpy is the always-available last resort.
_PREFERENCE = ("fenicsx", "mfem", "ccx", "numpy")


def _first_capable(needs: _Needs, *, require_available: bool) -> str:
    """First backend in :data:`_PREFERENCE` that can represent ``needs``."""
    from b3_core.solvers import get_backend

    for name in _PREFERENCE:
        try:
            backend = get_backend(name)
            caps = backend.capabilities
            if needs.unmet_by(caps):
                continue
            if require_available and not backend.is_available():
                continue
        except Exception:
            continue
        return name
    return "numpy"


def resolve_backend(case: CaseInput, requested: str | None = None) -> str:
    """Pick a backend name. ``auto`` logs the choice.

    ``auto`` prefers fenicsx when that environment is installed, then mfem, then
    the next capable backend, then numpy. An explicit backend that cannot do the
    job warns and falls back to the preferred capable backend in 0.3; that
    fallback becomes :class:`BackendCapabilityError` in 1.0.
    """
    from b3_core.solvers import get_backend

    name = requested or case.backend
    needs = _needs(case)
    if needs.curved:
        return _curved_choice(case, name)
    if name == "auto":
        choice = _first_capable(needs, require_available=True)
        logger.info("backend auto → %s (%s)", choice, needs.reason())
        return choice
    try:
        caps = get_backend(name).capabilities
    except UnknownBackendError:
        raise
    missing = needs.unmet_by(caps)
    if missing:
        fallback = _first_capable(needs, require_available=True)
        warnings.warn(
            f"backend {name!r} cannot handle {', '.join(missing)}; "
            f"falling back to {fallback} (BackendCapabilityError in 1.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "backend %s cannot handle %s; falling back to %s",
            name,
            ", ".join(missing),
            fallback,
        )
        return fallback
    return name


def fenicsx_installed() -> bool:
    """True when dolfinx imports. Sweep uses this so it does not import solvers."""
    from b3_core.solvers.fenicsx import is_fenicsx_available

    return bool(is_fenicsx_available())


def prepare(case: CaseInput, *, wall_morph: bool = True) -> PreparedCase:
    """Mesh, geometric report, and the resin-halo field.

    ``wall_morph=False`` builds the curved through-thickness stations and
    skips the interval-affine wall warp. That is the morph-off mesh used by
    ``diagnose bisect``.
    """
    from b3_core.core.analysis import geom_analysis
    from b3_core.core.mesh import create_grooved_mesh, kerf_openings
    from b3_core.core.scoring import ScoreField, effective_resin_vf, halo_reach

    reach = halo_reach(case)
    kwargs = case.mesh_kwargs()
    kwargs["s_halo"] = reach
    kwargs["wall_morph"] = wall_morph
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
        kerfs=tuple(kerf_openings(case)),
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
