"""Solver backend protocol. Every backend returns a full 6×6 stiffness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np


class BackendError(ValueError):
    """A backend name or capability is not usable."""


class UnknownBackendError(BackendError):
    """No backend is registered under this name."""


class BackendCapabilityError(BackendError):
    """The requested backend cannot represent this case.

    In 0.3 an incapable explicit backend warns and falls back, except a curved
    case: ``mfem`` and ``ccx`` raise immediately because a coincident-node tie
    on a tapered z face is the wrong number. From 1.0 every incapable explicit
    backend raises.
    """


@dataclass(frozen=True)
class Capabilities:
    orthotropic: bool
    halo: bool
    face_layer: bool
    displacements: bool
    element_types: frozenset[str] = frozenset({"C3D8"})
    interpolated_periodicity: bool = False


@dataclass(frozen=True)
class SolveRequest:
    mesh: Any
    core: Any
    resin: Any
    face: Any | None
    score_field: Any | None
    scoring: Any | None
    element_type: str
    details: bool = False
    workdir: Path | None = None


@dataclass(frozen=True)
class SolveResult:
    stiffness: np.ndarray
    properties: dict[str, float]
    compliance: np.ndarray
    displacements: dict[str, np.ndarray] | None = None
    points: np.ndarray | None = None
    extras: dict[str, float] = field(default_factory=dict)
    # Matrix before ``0.5(C+Cᵀ)``. Fenicsx does not symmetrise, so this matches
    # ``stiffness``. Absent on older result objects.
    raw_stiffness: np.ndarray | None = None
    # Per-face periodic ties. None means the caller builds them from the mesh.
    ties: list[dict[str, Any]] | None = None


def as_solve_result(result: Any, *, details: bool) -> SolveResult:
    """Adapt a legacy ``*Result`` object to :class:`SolveResult`."""
    displacements = getattr(result, "displacements", None)
    points = getattr(result, "points", None)
    raw = getattr(result, "raw_stiffness", None)
    ties = getattr(result, "ties", None)
    return SolveResult(
        stiffness=np.asarray(result.stiffness, dtype=float),
        properties=dict(result.properties),
        compliance=np.asarray(result.compliance, dtype=float),
        displacements=displacements if details else None,
        points=points if details else None,
        raw_stiffness=None if raw is None else np.asarray(raw, dtype=float),
        ties=None if ties is None else [dict(row) for row in ties],
    )


@runtime_checkable
class SolverBackend(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]

    def is_available(self) -> bool: ...

    def solve(self, req: SolveRequest) -> SolveResult: ...
