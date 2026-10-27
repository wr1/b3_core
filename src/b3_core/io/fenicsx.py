#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

LOAD_CASES = ("xx", "yy", "zz", "yz", "xz", "xy")
PROPERTY_KEYS = (
    "Exx",
    "Eyy",
    "Ezz",
    "Gxy",
    "Gxz",
    "Gyz",
    "nuxy",
    "nuxz",
    "nuyx",
    "nuyz",
    "nuzx",
    "nuzy",
)


class FenicsxUnavailableError(RuntimeError):
    """Raised when the optional FEniCSx stack is not installed."""


@dataclass(frozen=True)
class FenicsxResult:
    properties: dict[str, float]
    stiffness: np.ndarray
    compliance: np.ndarray


def is_fenicsx_available() -> bool:
    required = ("dolfinx", "dolfinx_mpc", "ufl", "basix", "mpi4py", "petsc4py")
    return all(importlib.util.find_spec(name) is not None for name in required)


def _require_fenicsx():
    if not is_fenicsx_available():
        raise FenicsxUnavailableError(
            "FEniCSx backend requires dolfinx, dolfinx_mpc, ufl, basix, "
            "mpi4py, and petsc4py. "
            "Install a FEniCSx environment and rerun with backend='fenicsx'."
        )

    import basix.ufl
    import dolfinx_mpc
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    return basix, dolfinx_mpc, ufl, fem, mesh, MPI


def _vtk_hexahedra(mesh):
    grid = mesh.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()

    cells = grid.cells.reshape((-1, 9))
    if not np.all(cells[:, 0] == 8):
        raise ValueError(
            "FEniCSx backend currently supports linear hexahedral cells only"
        )

    vtk_to_fenicsx = [0, 1, 3, 2, 4, 5, 7, 6]
    return (
        np.asarray(grid.points, dtype=np.float64),
        np.asarray(cells[:, 1:][:, vtk_to_fenicsx], dtype=np.int64),
        grid,
    )


def _material_field(fem, domain, grid, core, resin, face):
    if "E" not in core:
        raise ValueError(
            "FEniCSx backend currently supports isotropic core material only"
        )

    q = fem.functionspace(domain, ("DG", 0))
    young = fem.Function(q)
    poisson = fem.Function(q)

    resin_cells = np.asarray(grid.cell_data["resin"], dtype=bool)
    face_cells = np.asarray(grid.cell_data["face"], dtype=bool)
    e_values = np.full(grid.n_cells, core["E"], dtype=np.float64)
    nu_values = np.full(grid.n_cells, core["nu"], dtype=np.float64)

    e_values[resin_cells] = resin["E"]
    nu_values[resin_cells] = resin["nu"]

    if face is not None and face_cells.any():
        e_values[face_cells] = face.get("E", 12_000_000_000.0)
        nu_values[face_cells] = face.get("nu", 0.3)

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cell_vertices = domain.topology.connectivity(tdim, 0)
    geometry = domain.geometry.x
    local_cells = domain.topology.index_map(tdim).size_local
    dolfinx_centers = np.array(
        [geometry[cell_vertices.links(i)].mean(axis=0) for i in range(local_cells)]
    )
    pyvista_centers = np.asarray(
        grid.cell_centers().points, dtype=dolfinx_centers.dtype
    )
    center_to_cell = {
        tuple(np.round(center, 12)): index
        for index, center in enumerate(pyvista_centers)
    }
    cell_order = np.array(
        [center_to_cell[tuple(np.round(center, 12))] for center in dolfinx_centers],
        dtype=np.int64,
    )

    young.x.array[:local_cells] = e_values[cell_order]
    poisson.x.array[:local_cells] = nu_values[cell_order]
    return young, poisson


def _affine_values(case, origin):
    def values(x):
        xx = x[0] - origin[0]
        yy = x[1] - origin[1]
        zz = x[2] - origin[2]
        out = np.zeros((3, x.shape[1]), dtype=np.float64)
        if case == "xx":
            out[0] = xx
        elif case == "yy":
            out[1] = yy
        elif case == "zz":
            out[2] = zz
        elif case == "yz":
            out[1] = 0.5 * zz
            out[2] = 0.5 * yy
        elif case == "xz":
            out[0] = 0.5 * zz
            out[2] = 0.5 * xx
        elif case == "xy":
            out[0] = 0.5 * yy
            out[1] = 0.5 * xx
        else:
            raise ValueError(f"unknown load case {case!r}")
        return out

    return values


def _properties_from_stiffness(stiffness):
    compliance = np.linalg.inv(stiffness)
    return {
        "Exx": float(1.0 / compliance[0, 0]),
        "Eyy": float(1.0 / compliance[1, 1]),
        "Ezz": float(1.0 / compliance[2, 2]),
        "Gyz": float(1.0 / compliance[3, 3]),
        "Gxz": float(1.0 / compliance[4, 4]),
        "Gxy": float(1.0 / compliance[5, 5]),
        "nuxy": float(-compliance[1, 0] / compliance[0, 0]),
        "nuxz": float(-compliance[2, 0] / compliance[0, 0]),
        "nuyx": float(-compliance[0, 1] / compliance[1, 1]),
        "nuyz": float(-compliance[2, 1] / compliance[1, 1]),
        "nuzx": float(-compliance[0, 2] / compliance[2, 2]),
        "nuzy": float(-compliance[1, 2] / compliance[2, 2]),
    }, compliance


def runfenicsx(mesh, resin, core, face=None, *, return_details=False):
    """Solve six affine-strain homogenisation cases with FEniCSx.

    The returned property dictionary uses the same keys as the CCX postprocessor.
    This backend currently supports serial, linear-hexahedral, isotropic material
    models. It intentionally imports FEniCSx lazily so the package remains usable
    in environments that only have the CCX workflow installed.
    """

    basix, dolfinx_mpc, ufl, fem, dmesh, MPI = _require_fenicsx()
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("FEniCSx backend currently expects serial execution")

    points, cells, grid = _vtk_hexahedra(mesh)
    coord_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    domain = dmesh.create_mesh(MPI.COMM_WORLD, cells, ufl.Mesh(coord_el), points)

    v_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(domain.geometry.dim,))
    v = fem.functionspace(domain, v_el)
    young, poisson = _material_field(fem, domain, grid, core, resin, face)

    mu = young / (2.0 * (1.0 + poisson))
    lmbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(3)

    u = ufl.TrialFunction(v)
    w = ufl.TestFunction(v)
    a = ufl.inner(sigma(u), eps(w)) * ufl.dx

    bounds = np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 2].min()],
        dtype=np.float64,
    )
    upper = np.array(
        [points[:, 0].max(), points[:, 1].max(), points[:, 2].max()],
        dtype=np.float64,
    )
    tol = 1e-10

    lengths = upper - bounds

    def pin_corner(x):
        return (
            np.isclose(x[0], bounds[0], atol=tol)
            & np.isclose(x[1], bounds[1], atol=tol)
            & np.isclose(x[2], bounds[2], atol=tol)
        )

    pinned_dofs = fem.locate_dofs_geometrical(v, pin_corner)
    zero = fem.Function(v)
    zero.x.array[:] = 0.0
    pin_bc = fem.dirichletbc(zero, pinned_dofs)

    def periodic_mpc():
        mpc = dolfinx_mpc.MultiPointConstraint(v)
        scalar_type = domain.geometry.x.dtype
        constraints = {}
        for point in np.asarray(points, dtype=scalar_type):
            is_upper = np.isclose(point, upper.astype(scalar_type), atol=tol)
            if not np.any(is_upper):
                continue
            master = point.copy()
            master[is_upper] -= lengths.astype(scalar_type)[is_upper]
            constraints[point.tobytes()] = {master.tobytes(): 1.0}
        for component in range(3):
            mpc.create_general_constraint(
                constraints,
                subspace_slave=component,
                subspace_master=component,
            )
        mpc.finalize()
        return mpc

    volume = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain)))
    stress_entries = [
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ]
    stiffness = np.zeros((6, 6), dtype=np.float64)
    mpc = periodic_mpc()

    for col, case in enumerate(LOAD_CASES):
        macro = fem.Function(v)
        macro.interpolate(_affine_values(case, bounds))
        lform = -ufl.inner(sigma(macro), eps(w)) * ufl.dx
        problem = dolfinx_mpc.LinearProblem(
            a,
            lform,
            mpc,
            bcs=[pin_bc],
            petsc_options_prefix=f"b3_core_{case}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        fluctuation = problem.solve()
        mpc.backsubstitution(fluctuation)
        total = fem.Function(v)
        total.x.array[:] = fluctuation.x.array + macro.x.array
        for row, (i, j) in enumerate(stress_entries):
            stiffness[row, col] = (
                fem.assemble_scalar(fem.form(sigma(total)[i, j] * ufl.dx)) / volume
            )

    properties, compliance = _properties_from_stiffness(stiffness)
    if return_details:
        return FenicsxResult(properties, stiffness, compliance)
    return properties


def validate_against_ccx(
    ccx_output, other_output, *, label="fenicsx", rtol=0.05, atol=0.0
):
    comparison = {}
    passed = True
    for key in PROPERTY_KEYS:
        if key not in ccx_output or key not in other_output:
            continue
        ccx_value = float(ccx_output[key])
        other_value = float(other_output[key])
        abs_error = abs(other_value - ccx_value)
        rel_error = abs_error / max(abs(ccx_value), atol, np.finfo(float).eps)
        ok = bool(abs_error <= atol or rel_error <= rtol)
        comparison[key] = {
            "ccx": ccx_value,
            label: other_value,
            "abs_error": float(abs_error),
            "rel_error": float(rel_error),
            "ok": ok,
        }
        passed = passed and ok
    return {"passed": passed, "rtol": rtol, "atol": atol, "properties": comparison}
