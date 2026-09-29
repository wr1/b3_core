"""Registered stand-ins so API tests do not run a finite-element solve."""

from __future__ import annotations

import numpy as np

from b3_core.solvers import register
from b3_core.solvers.protocol import Capabilities, SolveResult

CANONICAL = {
    "Ex": 1.0e9,
    "Ey": 1.0e9,
    "Ez": 1.0e9,
    "Gxy": 0.4e9,
    "Gxz": 0.4e9,
    "Gyz": 0.4e9,
    "nuxy": 0.3,
    "nuxz": 0.3,
    "nuyz": 0.3,
    "Exx": 1.0e9,
    "Eyy": 1.0e9,
    "Ezz": 1.0e9,
}


def props(modulus: float, shear: float) -> dict[str, float]:
    """Canonical keys plus legacy Exx so flat() and validation both see them."""
    out = {
        "Ex": modulus,
        "Ey": modulus,
        "Ez": modulus,
        "Exx": modulus,
        "Eyy": modulus,
        "Ezz": modulus,
        "Gxy": shear,
        "Gxz": shear,
        "Gyz": shear,
        "nuxy": 0.3,
        "nuxz": 0.3,
        "nuyz": 0.3,
    }
    return out


def solver_double(name: str, properties: dict[str, float], calls: list[str]):
    """Backend whose solve() appends ``name`` and returns a constant tensor."""

    class _Backend:
        capabilities = Capabilities(
            orthotropic=True,
            halo=True,
            face_layer=True,
            displacements=False,
        )

        def is_available(self) -> bool:
            return True

        def solve(self, req):
            calls.append(name)
            stiffness = np.eye(6)
            return SolveResult(
                stiffness=stiffness,
                properties=dict(properties),
                compliance=stiffness,
            )

    _Backend.name = name
    return _Backend()


def fake_backend(name: str = "fake", *, solves: list | None = None):
    """Return ``(register, cls)`` for a capable backend with known properties."""

    class FakeBackend:
        capabilities = Capabilities(
            orthotropic=True,
            halo=True,
            face_layer=True,
            displacements=True,
        )

        def is_available(self) -> bool:
            return True

        def solve(self, req):
            if solves is not None:
                solves.append(name)
            stiffness = np.eye(6) * 1.0e9
            fields = None
            if req.details:
                fields = {"xx": np.zeros((1, 3))}
            return SolveResult(
                stiffness=stiffness,
                properties=dict(CANONICAL),
                compliance=np.eye(6) / 1.0e9,
                displacements=fields,
            )

    FakeBackend.name = name
    return register, FakeBackend


def unregister(name: str) -> None:
    from b3_core import solvers

    solvers._REGISTRY.pop(name, None)
