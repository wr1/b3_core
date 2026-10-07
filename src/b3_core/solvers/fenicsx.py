#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from b3_core.solvers.elasticity import (
    LOAD_CASES,
    constituent_dict,
    face_material_dict,
    properties_from_stiffness,
    scoring_payload,
)
from b3_core.solvers.protocol import (
    Capabilities,
    SolveRequest,
    SolveResult,
    as_solve_result,
)
from b3_core.solvers.sampling import per_gp_stiffness, phase_attributes


class FenicsxUnavailableError(RuntimeError):
    """Raised when the optional FEniCSx stack is not installed."""


@dataclass(frozen=True)
class FenicsxResult:
    properties: dict[str, float]
    stiffness: np.ndarray
    compliance: np.ndarray
    # u = E·(x − x0) + w on the mesh vertices (metres), one array per load case.
    displacements: dict | None = None
    points: np.ndarray | None = None


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


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _is_orthotropic(mat) -> bool:
    if mat is None:
        return False
    return constituent_dict(mat).get("E1") is not None


def _needs_general(core, resin, face, score_field) -> bool:
    """True when a single λ, μ per phase cannot represent the case."""
    if _is_orthotropic(core) or _is_orthotropic(resin) or _is_orthotropic(face):
        return True
    return score_field is not None and bool(getattr(score_field, "active", False))


def _dolfinx_cell_order(domain, grid) -> np.ndarray:
    """PyVista cell index of each local dolfinx cell, matched by centre.

    The two centre sums round at ULP scale, so a rounded coordinate key can
    flip for a value sitting on a rounding boundary. Match by nearest
    neighbour with a strict gap bound instead of a coordinate hash.
    """
    from scipy.spatial import cKDTree

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cell_vertices = domain.topology.connectivity(tdim, 0)
    geometry = domain.geometry.x
    local_cells = domain.topology.index_map(tdim).size_local
    centers = np.array(
        [geometry[cell_vertices.links(i)].mean(axis=0) for i in range(local_cells)]
    )
    pyvista_centers = np.asarray(grid.cell_centers().points, dtype=centers.dtype)
    scale = float(np.abs(pyvista_centers).max()) or 1.0
    dist, idx = cKDTree(pyvista_centers).query(centers, k=1)
    if float(dist.max()) > 1e-9 * scale:
        raise RuntimeError(
            "dolfinx/pyvista cell centres do not match (max gap "
            f"{float(dist.max()):.3e})"
        )
    return idx.astype(np.int64)


def _material_field(fem, domain, grid, core, resin, face):
    if "E" not in core:
        raise ValueError("FEniCSx isotropic path requires an isotropic core (E, nu)")

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

    cell_order = _dolfinx_cell_order(domain, grid)
    young.x.array[: len(cell_order)] = e_values[cell_order]
    poisson.x.array[: len(cell_order)] = nu_values[cell_order]
    return young, poisson


def _voigt_to_tensor(c6: np.ndarray) -> np.ndarray:
    """Engineering Voigt 6×6 to the tensor contracted as ``ε : C : ε``.

    The Voigt shear strain is ``γ = 2ε``. Storing the Voigt entry in every
    minor-symmetric slot makes the tensor contraction supply that factor.
    """
    tensor = np.zeros(c6.shape[:-2] + (3, 3, 3, 3), dtype=np.float64)
    for a, (row_i, row_j) in enumerate(_VOIGT_PAIRS):
        for b, (col_k, col_l) in enumerate(_VOIGT_PAIRS):
            val = c6[..., a, b]
            tensor[..., row_i, row_j, col_k, col_l] = val
            tensor[..., row_j, row_i, col_k, col_l] = val
            tensor[..., row_i, row_j, col_l, col_k] = val
            tensor[..., row_j, row_i, col_l, col_k] = val
    return tensor


def _physical_quadrature(domain, ref_points: np.ndarray) -> np.ndarray:
    """Map reference quadrature points to physical coordinates, metres."""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn = domain.topology.connectivity(tdim, 0)
    geometry = np.asarray(domain.geometry.x)
    cmap = domain.geometry.cmaps[0]
    pts = np.ascontiguousarray(ref_points, dtype=np.float64)
    n_cells = domain.topology.index_map(tdim).size_local
    out = np.empty((n_cells, len(pts), 3), dtype=np.float64)
    for cell in range(n_cells):
        nodes = np.ascontiguousarray(geometry[conn.links(cell)], dtype=np.float64)
        out[cell] = cmap.push_forward(pts, nodes)
    return out


def _quadrature_stiffness(
    basix_mod, fem, ufl, domain, grid, core, resin, face, score_field, scoring
):
    """Per-Gauss-point stiffness as a quadrature-element coefficient.

    The array is ``per_gp_stiffness``: neat phase 6×6 matrices, with the resin
    halo blended on foam points. Dolfinx contracts it as a 4th-order tensor.
    """
    degree = 2
    ref, _weights = basix_mod.make_quadrature(basix_mod.CellType.hexahedron, degree)
    gp_dx = _physical_quadrature(domain, ref)
    order = _dolfinx_cell_order(domain, grid)
    gp_pv = np.empty_like(gp_dx)
    gp_pv[order] = gp_dx

    cells = np.asarray(grid.cells).reshape(-1, 9)[:, 1:]
    c_pv = per_gp_stiffness(
        phase_attributes(grid, face),
        gp_pv * 1000.0,
        core=core,
        resin=resin,
        face=face,
        score_field=score_field,
        scoring=scoring,
        points_m=np.asarray(grid.points, dtype=np.float64),
        cells=cells,
    )
    tensor = np.ascontiguousarray(_voigt_to_tensor(c_pv[order]), dtype=np.float64)
    element = basix_mod.ufl.quadrature_element(
        "hexahedron", value_shape=(3, 3, 3, 3), degree=degree
    )
    stiffness = fem.Function(fem.functionspace(domain, element))
    flat = tensor.reshape(-1)
    if flat.size != stiffness.x.array.size:
        raise RuntimeError(
            f"quadrature stiffness has {flat.size} entries, the element expects "
            f"{stiffness.x.array.size}"
        )
    stiffness.x.array[:] = flat
    measure = ufl.Measure(
        "dx",
        domain=domain,
        metadata={"quadrature_degree": degree, "quadrature_rule": "default"},
    )
    return stiffness, measure


def _nodal_vectors(function_space, values, points: np.ndarray) -> np.ndarray:
    """Gather a block-3 vector function onto ``points``.

    ``tabulate_dof_coordinates`` reproduces the vertex coordinates up to a
    ULP-scale round-trip error, so a rounded coordinate key can flip for a
    value sitting on a rounding boundary. Match by nearest neighbour with a
    strict gap bound instead of a coordinate hash.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(function_space.tabulate_dof_coordinates())
    nodal = np.asarray(values, dtype=np.float64).reshape(len(coords), 3)
    if len(coords) != len(points):
        raise RuntimeError(
            f"function space has {len(coords)} nodes, the mesh has {len(points)}"
        )
    points = np.asarray(points, dtype=np.float64)
    scale = float(np.abs(points).max()) or 1.0
    dist, idx = cKDTree(points).query(coords, k=1)
    if float(dist.max()) > 1e-12 * scale:
        raise RuntimeError(
            "dof coordinates do not match mesh vertices (max gap "
            f"{float(dist.max()):.3e})"
        )
    out = np.empty_like(points, dtype=np.float64)
    out[idx] = nodal
    return out


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


def _hex_space(mesh):
    """Linear vector space on the mesh, scaled to metres. Serial only."""
    basix, _dolfinx_mpc, ufl, fem, dmesh, MPI = _require_fenicsx()
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("FEniCSx backend currently expects serial execution")

    points, cells, grid = _vtk_hexahedra(mesh)
    coord_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    domain = dmesh.create_mesh(MPI.COMM_WORLD, cells, ufl.Mesh(coord_el), points)
    v_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(domain.geometry.dim,))
    return domain, fem.functionspace(domain, v_el), points, grid


def _periodic_constraint(function_space, points):
    """Periodicity via the finite-element interpolant of the periodic image.

    ``create_periodic_constraint_topological`` evaluates the master cell that
    contains ``x - L``. A node image is weight 1. A kerf-tapered ``z`` face,
    whose nodes no longer coincide, gets that cell's bilinear weights. Edges
    and the high corner subtract every high-face period, so the image lies on
    the low faces. The low corner is the Dirichlet pin and is not a slave.
    """
    _basix, dolfinx_mpc, _ufl, fem, dmesh, _MPI = _require_fenicsx()
    bounds = np.min(points, axis=0).astype(np.float64, copy=False)
    upper = np.max(points, axis=0).astype(np.float64, copy=False)
    lengths = upper - bounds
    tol = 1e-10

    def pin_corner(x):
        return (
            np.isclose(x[0], bounds[0], atol=tol)
            & np.isclose(x[1], bounds[1], atol=tol)
            & np.isclose(x[2], bounds[2], atol=tol)
        )

    zero = fem.Function(function_space)
    zero.x.array[:] = 0.0
    pin_bc = fem.dirichletbc(
        zero, fem.locate_dofs_geometrical(function_space, pin_corner)
    )

    # One tag covers every high face. An edge or corner dof is unique, and
    # the relation reduces every high coordinate.
    def on_high_face(x):
        return (
            np.isclose(x[0], upper[0], atol=tol)
            | np.isclose(x[1], upper[1], atol=tol)
            | np.isclose(x[2], upper[2], atol=tol)
        )

    domain = function_space.mesh
    fdim = domain.topology.dim - 1
    facets = np.sort(dmesh.locate_entities_boundary(domain, fdim, on_high_face))
    tags = dmesh.meshtags(domain, fdim, facets, np.full(len(facets), 1, dtype=np.int32))

    def relation(x):
        out = np.array(x, copy=True)
        for axis in range(3):
            on_face = np.isclose(x[axis], upper[axis], atol=tol)
            out[axis, on_face] -= lengths[axis]
        return out

    mpc = dolfinx_mpc.MultiPointConstraint(function_space)
    # dolfinx_mpc rejects the vector element itself. One call per component
    # applies the same map, so each direction is the same interpolant.
    for component in range(3):
        mpc.create_periodic_constraint_topological(
            function_space.sub(component),
            tags,
            1,
            relation,
            [pin_bc],
            scale=1.0,
            tol=tol,
        )
    mpc.finalize()
    return mpc, pin_bc, bounds


# One direct factor of the condensed system, reused for the six load cases.
# On the curved grid-scored mesh (~19k dofs) MUMPS LU took 5.9 s, UMFPACK 13 s,
# and PETSc's own LU 63 s. The three agreed to 1e-12. CHOLMOD rejects the
# matrix: the MPC system is not positive definite.
_DIRECT = ("lu", "mumps")


def _solve_fluctuations(
    fem, dolfinx_mpc, function_space, a_form, load_form, macro, mpc, pin_bc, bounds
):
    """Factor ``K`` once and solve the six macro-strain loads.

    ``load_form`` is the compiled linear form of ``-σ(macro) : ε(v)``.
    ``macro`` is interpolated per case, so the form is not rebuilt.
    Returns ``u = E·(x − x0) + w`` as a separate function per load case.
    """
    from dolfinx.fem.petsc import assign, set_bc
    from dolfinx.la.petsc import _ghost_update, create_vector
    from petsc4py import PETSc

    pc_type, package = _DIRECT
    bcs = [pin_bc]
    matrix = dolfinx_mpc.assemble_matrix(a_form, mpc, bcs=bcs)
    ksp = PETSc.KSP().create(function_space.mesh.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType(pc_type)
    pc.setFactorSolverType(package)
    try:
        ksp.setUp()
    except Exception as exc:
        ksp.destroy()
        matrix.destroy()
        raise RuntimeError(
            f"FEniCSx could not factor the periodic system with {package}"
        ) from exc

    index_map = (
        function_space.dofmap.index_map,
        function_space.dofmap.index_map_bs,
    )
    rhs = create_vector([index_map])
    unknown = create_vector([index_map])
    totals = {}
    try:
        for case in LOAD_CASES:
            macro.interpolate(_affine_values(case, bounds))
            dolfinx_mpc.assemble_vector(load_form, mpc, rhs)
            dolfinx_mpc.apply_lifting(rhs, [a_form], bcs=[bcs], constraint=mpc)
            _ghost_update(rhs, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            set_bc(rhs, bcs)
            _ghost_update(rhs, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            ksp.solve(rhs, unknown)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError(
                    f"FEniCSx direct solve failed for load {case}: "
                    f"{ksp.getConvergedReason()}"
                )
            _ghost_update(unknown, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            fluctuation = fem.Function(function_space)
            assign(unknown, fluctuation)
            mpc.homogenize(fluctuation)
            mpc.backsubstitution(fluctuation)
            total = fem.Function(function_space)
            total.x.array[:] = fluctuation.x.array + macro.x.array
            totals[case] = total
    finally:
        rhs.destroy()
        unknown.destroy()
        ksp.destroy()
        matrix.destroy()
    return totals


def runfenicsx(
    mesh,
    resin,
    core,
    face=None,
    *,
    score_field=None,
    scoring=None,
    return_details=False,
):
    """Solve six affine-strain homogenisation cases with FEniCSx.

    The returned property dictionary uses the same keys as the CCX postprocessor.
    Serial, linear hexahedra. Isotropic phases use ``λ`` and ``μ``. Orthotropic
    constituents and the graded resin halo use the shared per-Gauss-point 6×6
    stiffness. The six load cases share one direct factor of ``K``. FEniCSx is
    imported lazily so the package stays usable without it.

    Periodicity is ``dolfinx_mpc``'s topological constraint: each high-face
    degree of freedom equals the finite-element interpolant of its image after
    subtracting the periods of every high face it lies on. A node that lands
    on a node stays a weight-1 tie. A kerf-tapered ``z`` face, whose nodes no
    longer match, is the bilinear interpolant of the bottom face.
    """

    basix, dolfinx_mpc, ufl, fem, _dmesh, _MPI = _require_fenicsx()
    resin = constituent_dict(resin)
    core = constituent_dict(core)
    face = face_material_dict(face)
    scoring = scoring_payload(scoring)
    domain, v, points, grid = _hex_space(mesh)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    if _needs_general(core, resin, face, score_field):
        stiffness_c, dx = _quadrature_stiffness(
            basix,
            fem,
            ufl,
            domain,
            grid,
            core,
            resin,
            face,
            score_field,
            scoring,
        )

        def sigma(u):
            ii, jj, kk, ll = ufl.indices(4)
            strain = eps(u)
            return ufl.as_tensor(stiffness_c[ii, jj, kk, ll] * strain[kk, ll], (ii, jj))

    else:
        young, poisson = _material_field(fem, domain, grid, core, resin, face)
        mu = young / (2.0 * (1.0 + poisson))
        lmbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        dx = ufl.dx

        def sigma(u):
            return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(3)

    u = ufl.TrialFunction(v)
    w = ufl.TestFunction(v)
    a = ufl.inner(sigma(u), eps(w)) * dx

    mpc, pin_bc, bounds = _periodic_constraint(v, points)
    # One matrix for all six columns. Only the macro-strain load changes.
    macro = fem.Function(v)
    a_form = fem.form(a)
    load_form = fem.form(-ufl.inner(sigma(macro), eps(w)) * dx)
    fluctuations = _solve_fluctuations(
        fem, dolfinx_mpc, v, a_form, load_form, macro, mpc, pin_bc, bounds
    )

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
    displacements = {} if return_details else None

    for col, case in enumerate(LOAD_CASES):
        total = fluctuations[case]
        if displacements is not None:
            displacements[case] = _nodal_vectors(v, total.x.array, points)
        for row, (i, j) in enumerate(stress_entries):
            stiffness[row, col] = (
                fem.assemble_scalar(fem.form(sigma(total)[i, j] * dx)) / volume
            )

    properties, compliance = properties_from_stiffness(stiffness)
    if return_details:
        return FenicsxResult(properties, stiffness, compliance, displacements, points)
    return properties


class FenicsxBackend:
    name = "fenicsx"
    capabilities = Capabilities(
        orthotropic=True,
        halo=True,
        face_layer=True,
        displacements=True,
        element_types=frozenset({"C3D8"}),
    )

    def is_available(self) -> bool:
        return is_fenicsx_available()

    def solve(self, req: SolveRequest) -> SolveResult:
        result = runfenicsx(
            req.mesh,
            req.resin,
            req.core,
            req.face,
            score_field=req.score_field,
            scoring=req.scoring,
            return_details=True,
        )
        return as_solve_result(result, details=req.details)
