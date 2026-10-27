"""Install check for the FEniCSx stack used on curved cases."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from b3_core.solvers.fenicsx import DIRECT

_STACK = (
    ("dolfinx", ("dolfinx", "fenics-dolfinx")),
    ("dolfinx_mpc", ("dolfinx_mpc", "dolfinx-mpc")),
    ("petsc4py", ("petsc4py",)),
    ("mpi4py", ("mpi4py",)),
    ("ufl", ("fenics-ufl", "ufl")),
    ("basix", ("fenics-basix", "basix")),
)


def _version(module: Any, dist_names: tuple[str, ...]) -> str:
    for name in dist_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    found = getattr(module, "__version__", None)
    return "unknown" if found is None else str(found)


def probe_imports() -> tuple[dict[str, str | None], str | None]:
    """Import each required module. The first failure is the error string."""
    versions: dict[str, str | None] = {}
    error: str | None = None
    for import_name, dist_names in _STACK:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            versions[import_name] = None
            if error is None:
                error = f"{import_name}: {exc}"
            continue
        versions[import_name] = _version(module, dist_names)
    return versions, error


def _mpi_and_mumps() -> tuple[int, bool]:
    from mpi4py import MPI
    from petsc4py import PETSc

    has_mumps = bool(PETSc.Sys.hasExternalPackage("mumps"))
    return int(MPI.COMM_WORLD.size), has_mumps and DIRECT[1] == "mumps"


def _cube_mesh():
    """2×2×2 linear hex cube, all foam, in millimetres."""
    import numpy as np
    import pyvista as pv

    axis = np.linspace(0.0, 2.0, 3)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = pv.StructuredGrid(xx, yy, zz).cast_to_unstructured_grid()
    cells = int(grid.n_cells)
    grid.cell_data["resin"] = np.zeros(cells, dtype=bool)
    grid.cell_data["face"] = np.zeros(cells, dtype=bool)
    return grid


def _periodic_cube() -> dict[str, Any]:
    """One periodic homogenisation of a homogeneous 2×2×2 cube."""
    from b3_core.solvers.fenicsx import runfenicsx

    grid = _cube_mesh()
    young = 4.0e9
    props = runfenicsx(
        grid,
        {"E": young, "nu": 0.3, "rho": 1100.0},
        {"E": young, "nu": 0.3, "rho": 100.0},
    )
    ex = float(props["Ex"])
    rel = abs(ex - young) / young
    return {
        "ok": rel < 1e-4 and int(grid.n_cells) == 8,
        "cells": int(grid.n_cells),
        "Ex": ex,
        "rel_error": rel,
    }


def doctor() -> dict[str, Any]:
    """Report imports, MPI size, the MUMPS factor, and a 2×2×2 solve.

    ``ok`` is true only when every import works, MUMPS is the factorisation,
    and the cube solve recovers the input modulus.
    """
    versions, error = probe_imports()
    factorisation = f"{DIRECT[0]}/{DIRECT[1]}"
    report: dict[str, Any] = {
        "ok": False,
        "imports": versions,
        "mpi_size": None,
        "mumps": False,
        "factorisation": factorisation,
        "solve": None,
        "error": error,
    }
    if error is not None:
        return report
    try:
        mpi_size, mumps = _mpi_and_mumps()
    except Exception as exc:
        report["error"] = f"mpi/mumps: {exc}"
        return report
    report["mpi_size"] = mpi_size
    report["mumps"] = mumps
    if not mumps:
        report["error"] = f"MUMPS is not the factorisation ({factorisation})"
        return report
    try:
        solve = _periodic_cube()
    except Exception as exc:
        report["error"] = f"solve: {exc}"
        report["solve"] = {"ok": False, "error": str(exc)}
        return report
    report["solve"] = solve
    if not solve["ok"]:
        report["error"] = (
            f"2x2x2 periodic solve Ex={solve['Ex']} rel_error={solve['rel_error']}"
        )
        return report
    report["ok"] = True
    report["error"] = None
    return report
