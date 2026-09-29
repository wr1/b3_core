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

    In 0.3 an incapable explicit backend warns and falls back to numpy.
    From 1.0 ``resolve_backend`` raises this exception instead.
    """


@dataclass(frozen=True)
class Capabilities:
    orthotropic: bool
    halo: bool
    face_layer: bool
    displacements: bool
    element_types: frozenset[str] = frozenset({"C3D8"})


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


def as_solve_result(result: Any, *, details: bool) -> SolveResult:
    """Adapt a legacy ``*Result`` object to :class:`SolveResult`."""
    displacements = getattr(result, "displacements", None)
    points = getattr(result, "points", None)
    return SolveResult(
        stiffness=np.asarray(result.stiffness, dtype=float),
        properties=dict(result.properties),
        compliance=np.asarray(result.compliance, dtype=float),
        displacements=displacements if details else None,
        points=points if details else None,
    )


@runtime_checkable
class SolverBackend(Protocol):
    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]

    def is_available(self) -> bool: ...

    def solve(self, req: SolveRequest) -> SolveResult: ...
