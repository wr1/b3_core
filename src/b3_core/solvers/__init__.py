"""Backend registry.

Built-ins load on first use so ``import b3_core`` does not import mfem,
dolfinx, or pyvista. Third-party backends register through the entry-point
group ``b3_core.solvers`` or :func:`register`.
"""

from __future__ import annotations

import importlib
from typing import Any

from b3_core.solvers.protocol import (
    BackendError,
    SolverBackend,
    UnknownBackendError,
)

_REGISTRY: dict[str, type] = {}
_ENTRY_POINTS_LOADED = False

_BUILTIN = {
    "mfem": "b3_core.solvers.mfem:MfemBackend",
    "numpy": "b3_core.solvers.numpy_fe.backend:NumpyBackend",
    "fenicsx": "b3_core.solvers.fenicsx:FenicsxBackend",
    "ccx": "b3_core.solvers.calculix.backend:CalculixBackend",
}


def register(cls: type) -> type:
    """Register a :class:`SolverBackend`. Usable as a decorator."""
    name = getattr(cls, "name", None)
    if not name:
        raise BackendError(f"{cls!r} has no backend name")
    _REGISTRY[str(name)] = cls
    return cls


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    from importlib.metadata import entry_points

    eps = entry_points()
    group: Any
    if hasattr(eps, "select"):
        group = eps.select(group="b3_core.solvers")
    else:
        group = eps.get("b3_core.solvers", [])
    for ep in group:
        register(ep.load())


def _load_builtin(name: str) -> None:
    target = _BUILTIN[name]
    module_name, _, qual = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, qual)
    register(cls)


def get_backend(name: str) -> SolverBackend:
    """Return a new instance of the named backend."""
    _load_entry_points()
    if name not in _REGISTRY and name in _BUILTIN:
        _load_builtin(name)
    cls = _REGISTRY.get(name)
    if cls is None:
        known = sorted(set(_REGISTRY) | set(_BUILTIN))
        raise UnknownBackendError(
            f"unknown backend {name!r}; known backends: {', '.join(known)}"
        )
    return cls()


def available_backends() -> dict[str, bool]:
    """Name → :meth:`is_available` for every built-in and registered backend."""
    _load_entry_points()
    names = set(_BUILTIN) | set(_REGISTRY)
    out: dict[str, bool] = {}
    for name in sorted(names):
        try:
            out[name] = bool(get_backend(name).is_available())
        except Exception:
            out[name] = False
    return out


__all__ = [
    "BackendError",
    "UnknownBackendError",
    "available_backends",
    "get_backend",
    "register",
]
