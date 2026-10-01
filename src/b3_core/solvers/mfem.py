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

# The pyvista StructuredGrid hexahedra are wound so that the raw VTK
# connectivity yields a negative Jacobian in MFEM; swapping the bottom and top
# quads restores a positive-orientation element.
_VTK_TO_MFEM_HEX = [4, 5, 6, 7, 0, 1, 2, 3]

# Hard-coded skin material, matching the fenicsx backend defaults.
_FACE_E = 12_000_000_000.0
_FACE_NU = 0.3


class MfemUnavailableError(RuntimeError):
    """Raised when the optional PyMFEM stack is not installed."""


@dataclass(frozen=True)
class MfemResult:
    properties: dict[str, float]
    stiffness: np.ndarray
    compliance: np.ndarray
    # Per-load-case total displacement u = E.x + w on the original grid points
    # (only populated when return_details=True), for deformed-shape / periodicity
    # visualisation. points are the base-grid coordinates (metres).
    displacements: dict | None = None
    points: np.ndarray | None = None


def is_mfem_available() -> bool:
    return importlib.util.find_spec("mfem") is not None


def _require_mfem():
    if not is_mfem_available():
        raise MfemUnavailableError(
            "MFEM backend requires PyMFEM. It is a project dependency "
            "(`pip install mfem`, or `uv sync`). On Python 3.12 see the "
            "README note on numba. Then rerun with backend='mfem'."
        )

    import mfem.ser as mfem

    return mfem


def _lame(young, poisson):
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lam, mu


def _macro_strain(case):
    """Unit macroscopic strain tensor for a load case (engineering shear = 1)."""
    strain = np.zeros((3, 3), dtype=np.float64)
    diag = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2)}
    if case in diag:
        strain[diag[case]] = 1.0
    elif case == "yz":
        strain[1, 2] = strain[2, 1] = 0.5
    elif case == "xz":
        strain[0, 2] = strain[2, 0] = 0.5
    elif case == "xy":
        strain[0, 1] = strain[1, 0] = 0.5
    else:
        raise ValueError(f"unknown load case {case!r}")
    return strain


def _strain_voigt(strain):
    """Engineering Voigt strain from a symmetric 3x3 tensor (shear doubled)."""
    return np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            2.0 * strain[1, 2],
            2.0 * strain[0, 2],
            2.0 * strain[0, 1],
        ]
    )


def _iso_stress(strain, lam, mu):
    return lam * np.trace(strain) * np.eye(3) + 2.0 * mu * strain


def _vtk_hexahedra(mesh):
    grid = mesh.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()

    cells = grid.cells.reshape((-1, 9))
    if not np.all(cells[:, 0] == 8):
        raise ValueError("MFEM backend currently supports linear hexahedral cells only")

    return (
        np.asarray(grid.points, dtype=np.float64),
        np.asarray(cells[:, 1:], dtype=np.int64),
        grid,
    )


def _material_arrays(core, resin, face, max_attr):
    """Per-attribute Lame parameters, index i == attribute i + 1."""
    if "E" not in core:
        raise ValueError("MFEM backend currently supports isotropic core material only")

    face_e = (face or {}).get("E", _FACE_E)
    face_nu = (face or {}).get("nu", _FACE_NU)
    materials = [
        (core["E"], core["nu"]),
        (resin["E"], resin["nu"]),
        (face_e, face_nu),
    ]

    lam = np.zeros(max_attr, dtype=np.float64)
    mu = np.zeros(max_attr, dtype=np.float64)
    for i in range(max_attr):
        young, poisson = materials[i] if i < len(materials) else materials[0]
        lam[i], mu[i] = _lame(young, poisson)
    return lam, mu


# --------------------------------------------------------------------------- #
# anisotropic / graded per-Gauss-point path
# --------------------------------------------------------------------------- #
def _is_orthotropic(mat) -> bool:
    return constituent_dict(mat).get("E1") is not None


def _needs_general_path(core, resin, face, score_field) -> bool:
    """True when isotropic per-attribute Lame parameters cannot represent the case."""
    if _is_orthotropic(core) or _is_orthotropic(resin):
        return True
    if face is not None and _is_orthotropic(face):
        return True
    return score_field is not None and bool(getattr(score_field, "active", False))


def _voigt_b(dshape):
    """Engineering-Voigt B (6 x nd*3) for MFEM's ``byVDIM`` element dof order.

    ``dshape`` is ``(nd, 3)`` physical-space shape-function gradients. Columns
    are component-major: ``[ux_0..ux_{nd-1}, uy_.., uz_..]`` — the order
    ``GetElementVDofs`` returns for a vector space built with ``byVDIM``.
    """
    d = np.asarray(dshape, dtype=np.float64)
    nd = d.shape[0]
    cx = np.arange(nd)
    cy = cx + nd
    cz = cx + 2 * nd
    B = np.zeros((6, nd * 3))
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    B[0, cx] = dx
    B[1, cy] = dy
    B[2, cz] = dz
    B[3, cy] = dz
    B[3, cz] = dy
    B[4, cx] = dz
    B[4, cz] = dx
    B[5, cx] = dy
    B[5, cy] = dx
    return B


def _collect_gp_data(mfem, mesh, fes):
    """Physical Gauss-point coordinates, derivatives and weights for every cell.

    Mirrors ``b3_tex``'s one-pass collector: ``T.Transform`` for the physical
    point, ``CalcDShape`` + ``CalcInverse(J)`` for physical derivatives, and
    ``ip.weight * T.Weight()`` for the integration weight.
    """
    n_elem = mesh.GetNE()
    fe0 = fes.GetFE(0)
    nd, dim = fe0.GetDof(), fe0.GetDim()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    coords = np.empty((n_elem * nq, 3), dtype=np.float64)
    dshapes = np.empty((n_elem * nq, nd, 3), dtype=np.float64)
    weights = np.empty(n_elem * nq, dtype=np.float64)

    d_ref = mfem.DenseMatrix(nd, dim)
    j_inv = mfem.DenseMatrix(dim, dim)
    d_phys = mfem.DenseMatrix(nd, dim)
    for e in range(n_elem):
        trans = mesh.GetElementTransformation(e)
        fe = fes.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        for q in range(nq):
            ip = ir.IntPoint(q)
            trans.SetIntPoint(ip)
            idx = e * nq + q
            coords[idx] = np.asarray(trans.Transform(ip))
            fe.CalcDShape(ip, d_ref)
            mfem.CalcInverse(trans.Jacobian(), j_inv)
            mfem.Mult(d_ref, j_inv, d_phys)
            dshapes[idx] = np.asarray(d_phys.GetDataArray())
            weights[idx] = ip.weight * trans.Weight()

    return coords, dshapes, weights, n_elem, nq, nd


def _precomputed_integrator(mfem, c_gp, dshapes, weights, n_elem, nq, nd):
    c_view = c_gp.reshape(n_elem, nq, 6, 6)
    d_view = dshapes.reshape(n_elem, nq, nd, 3)
    w_view = weights.reshape(n_elem, nq)

    class _PrecomputedIntegrator(mfem.PyBilinearFormIntegrator):
        def __init__(self):
            super().__init__()

        def AssembleElementMatrix(self, fe, trans, elmat):
            e = trans.ElementNo
            elmat.SetSize(nd * 3)
            local = np.zeros((nd * 3, nd * 3))
            for q in range(nq):
                b = _voigt_b(d_view[e, q])
                local += b.T @ c_view[e, q] @ b * w_view[e, q]
            elmat.GetDataArray()[:] = local

    return _PrecomputedIntegrator()


def _precomputed_rhs(mfem, sigma_gp, dshapes, weights, n_elem, nq, nd):
    s_view = sigma_gp.reshape(n_elem, nq, 6)
    d_view = dshapes.reshape(n_elem, nq, nd, 3)
    w_view = weights.reshape(n_elem, nq)

    class _PrecomputedRHS(mfem.PyLinearFormIntegrator):
        def __init__(self):
            super().__init__()

        def AssembleRHSElementVect(self, el, trans, elvect):
            e = trans.ElementNo
            elvect.SetSize(nd * 3)
            local = np.zeros(nd * 3)
            for q in range(nq):
                b = _voigt_b(d_view[e, q])
                local -= b.T @ s_view[e, q] * w_view[e, q]
            elvect.GetDataArray()[:] = local

    return _PrecomputedRHS()


def _solve_general(mfem, fes, pinned, c_gp, dshapes, weights, n_elem, nq, nd):
    """Assemble and solve the six periodic cases with per-GP stiffness.

    Returns ``(stiffness, correctors)`` where ``correctors[k]`` is the periodic
    fluctuation ``w`` (global dof array) for load case ``k``.
    """
    c_flat = c_gp.reshape(n_elem * nq, 6, 6)
    bilinear = mfem.BilinearForm(fes)
    bilinear.AddDomainIntegrator(
        _precomputed_integrator(mfem, c_gp, dshapes, weights, n_elem, nq, nd)
    )
    bilinear.Assemble()

    total_volume = float(weights.sum())
    t1 = np.einsum("n,nij->ij", weights, c_flat)  # integral of C(x) dV

    load = np.zeros((fes.GetVSize(), len(LOAD_CASES)))
    correctors: list[np.ndarray] = []
    for k, case in enumerate(LOAD_CASES):
        eps0 = _strain_voigt(_macro_strain(case))
        sigma = np.einsum("nij,j->ni", c_flat, eps0)
        lform = mfem.LinearForm(fes)
        lform.AddDomainIntegrator(
            _precomputed_rhs(mfem, sigma, dshapes, weights, n_elem, nq, nd)
        )
        lform.Assemble()
        load[:, k] = -lform.GetDataArray()  # positive macro load L_k

        corrector = mfem.GridFunction(fes)
        corrector.Assign(0.0)
        operator = mfem.OperatorPtr()
        rhs = mfem.Vector()
        sol = mfem.Vector()
        bilinear.FormLinearSystem(pinned, corrector, lform, operator, sol, rhs)
        matrix = mfem.OperatorHandle2SparseMatrix(operator)
        smoother = mfem.GSSmoother(matrix)
        mfem.PCG(matrix, smoother, rhs, sol, 0, 5000, 1e-12, 0.0)
        bilinear.RecoverFEMSolution(sol, lform, corrector)
        correctors.append(corrector.GetDataArray().copy())

    w = np.column_stack(correctors)
    stiffness = (t1 + load.T @ w) / total_volume
    stiffness = 0.5 * (stiffness + stiffness.T)
    return stiffness, correctors


def runmfem(
    mesh,
    resin,
    core,
    face=None,
    *,
    score_field=None,
    scoring=None,
    return_details=False,
):
    """Solve six periodic-homogenisation cases with MFEM (PyMFEM).

    Uses true periodic boundary conditions via an MFEM periodic mesh and an
    energy-based reduction of the corrector solves, returning the same property
    keys as the CCX postprocessor. Isotropic constituents use MFEM's native
    ``ElasticityIntegrator``; orthotropic constituents and the graded resin halo
    use a per-Gauss-point stiffness with a custom integrator, matching the numpy
    backend's constitutive map. The backend is serial and linear-hexahedral, and
    imports PyMFEM lazily so the package stays usable in CCX-only environments.
    """

    mfem = _require_mfem()
    core = constituent_dict(core)
    resin = constituent_dict(resin)

    points, cells, grid = _vtk_hexahedra(mesh)
    attr = phase_attributes(grid, face)

    lengths = points.max(axis=0) - points.min(axis=0)
    for axis, name in enumerate("xyz"):
        if len(np.unique(np.round(points[:, axis], 12))) < 4:
            raise ValueError(
                f"MFEM backend needs at least three elements along {name} for "
                "periodic boundary conditions; refine the mesh (e.g. add madd "
                "layers)."
            )

    base = mfem.Mesh(3, len(points), len(cells), 0, 3)
    for point in points:
        base.AddVertex(float(point[0]), float(point[1]), float(point[2]))
    for cell, cell_attr in zip(cells[:, _VTK_TO_MFEM_HEX], attr, strict=True):
        base.AddHex(*[int(v) for v in cell], int(cell_attr))
    base.FinalizeHexMesh(1, 0, False)

    translations = (
        mfem.Vector([float(lengths[0]), 0.0, 0.0]),
        mfem.Vector([0.0, float(lengths[1]), 0.0]),
        mfem.Vector([0.0, 0.0, float(lengths[2])]),
    )
    periodic = mfem.Mesh.MakePeriodic(
        base, base.CreatePeriodicVertexMapping(translations)
    )

    fec = mfem.H1_FECollection(1, 3)
    fes = mfem.FiniteElementSpace(periodic, fec, 3, mfem.Ordering.byVDIM)

    # For deformed-shape viz: map each original (base) grid vertex to the vdofs of
    # its periodic image, so the periodic fluctuation w can be read back onto the
    # full grid. MakePeriodic merges identified vertices, and H1 DOF order is not
    # vertex order, hence the CreatePeriodicVertexMapping + DofToVDof round-trip.
    base_to_vdof = None
    if return_details:
        vmap = base.CreatePeriodicVertexMapping(translations)
        v2v = np.array([vmap[i] for i in range(len(points))], dtype=np.int64)
        pverts = np.array([periodic.GetVertexArray(i) for i in range(periodic.GetNV())])
        pindex = {
            (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9)): j
            for j, p in enumerate(pverts)
        }
        peridx = np.array(
            [
                pindex[
                    (
                        round(float(points[v2v[i], 0]), 9),
                        round(float(points[v2v[i], 1]), 9),
                        round(float(points[v2v[i], 2]), 9),
                    )
                ]
                for i in range(len(points))
            ],
            dtype=np.int64,
        )
        base_to_vdof = np.array(
            [[fes.DofToVDof(int(pv), comp) for comp in range(3)] for pv in peridx],
            dtype=np.int64,
        )

    # Pin the three DOFs of one node to remove the rigid-body translations of
    # the fully periodic cell (a 3-torus has no periodic rotation modes).
    pinned = mfem.intArray([fes.DofToVDof(0, comp) for comp in range(3)])

    general = _needs_general_path(core, resin, face, score_field)
    if general:
        gp_coords, dshapes, weights, n_elem, nq, nd = _collect_gp_data(
            mfem, periodic, fes
        )
        gp_mm = gp_coords.reshape(n_elem, nq, 3) * 1000.0
        c_gp = per_gp_stiffness(
            attr,
            gp_mm,
            core=core,
            resin=resin,
            face=face,
            score_field=score_field,
            scoring=scoring_payload(scoring),
            points_m=points,
            cells=cells,
        )
        stiffness, correctors = _solve_general(
            mfem, fes, pinned, c_gp, dshapes, weights, n_elem, nq, nd
        )
    else:
        max_attr = periodic.attributes.Max()
        lam_by_attr, mu_by_attr = _material_arrays(core, resin, face, max_attr)
        lam_coeff = mfem.PWConstCoefficient(mfem.Vector(lam_by_attr.tolist()))
        mu_coeff = mfem.PWConstCoefficient(mfem.Vector(mu_by_attr.tolist()))

        bilinear = mfem.BilinearForm(fes)
        bilinear.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coeff, mu_coeff))
        bilinear.Assemble()

        element_volume = np.array(
            [base.GetElementVolume(i) for i in range(base.GetNE())], dtype=np.float64
        )
        total_volume = float(element_volume.sum())
        present = sorted({int(a) for a in attr})
        volume_by_attr = {p: float(element_volume[attr == p].sum()) for p in present}
        lame_by_attr = {p: (lam_by_attr[p - 1], mu_by_attr[p - 1]) for p in present}

        load_vectors = []
        correctors = []
        for case in LOAD_CASES:
            strain = _macro_strain(case)
            lform = mfem.LinearForm(fes)
            keep_alive = []  # PyMFEM holds raw pointers; keep Python refs alive
            for p in present:
                lam_p, mu_p = lame_by_attr[p]
                stress = _iso_stress(strain, lam_p, mu_p).flatten()
                marker = mfem.intArray([0] * max_attr)
                marker[p - 1] = 1
                coeff = mfem.VectorConstantCoefficient(
                    mfem.Vector([float(v) for v in stress])
                )
                integrator = mfem.VectorDomainLFGradIntegrator(coeff)
                lform.AddDomainIntegrator(integrator, marker)
                keep_alive.extend((marker, coeff, integrator))
            lform.Assemble()
            load = lform.GetDataArray().copy()

            # Solve K w = -L for the periodic fluctuation w.
            lform *= -1.0
            corrector = mfem.GridFunction(fes)
            corrector.Assign(0.0)
            operator = mfem.OperatorPtr()
            rhs = mfem.Vector()
            sol = mfem.Vector()
            bilinear.FormLinearSystem(pinned, corrector, lform, operator, sol, rhs)
            matrix = mfem.OperatorHandle2SparseMatrix(operator)
            smoother = mfem.GSSmoother(matrix)
            mfem.PCG(matrix, smoother, rhs, sol, 0, 5000, 1e-12, 0.0)
            bilinear.RecoverFEMSolution(sol, lform, corrector)

            load_vectors.append(load)
            correctors.append(corrector.GetDataArray().copy())

        stiffness = np.zeros((6, 6), dtype=np.float64)
        for k, case_k in enumerate(LOAD_CASES):
            strain_k = _macro_strain(case_k)
            for j, case_l in enumerate(LOAD_CASES):
                strain_l = _macro_strain(case_l)
                energy = sum(
                    volume_by_attr[p]
                    * (
                        lame_by_attr[p][0] * np.trace(strain_k) * np.trace(strain_l)
                        + 2.0 * lame_by_attr[p][1] * np.sum(strain_k * strain_l)
                    )
                    for p in present
                )
                stiffness[k, j] = (
                    energy + load_vectors[k].dot(correctors[j])
                ) / total_volume

        stiffness = 0.5 * (stiffness + stiffness.T)

    properties, compliance = properties_from_stiffness(stiffness)
    if return_details:
        displacements = {}
        for k, case in enumerate(LOAD_CASES):
            displacements[case] = (
                points @ _macro_strain(case) + correctors[k][base_to_vdof]
            )
        return MfemResult(properties, stiffness, compliance, displacements, points)
    return properties


class MfemBackend:
    name = "mfem"
    capabilities = Capabilities(
        orthotropic=True,
        halo=True,
        face_layer=True,
        displacements=True,
        element_types=frozenset({"C3D8"}),
    )

    def is_available(self) -> bool:
        return is_mfem_available()

    def solve(self, req: SolveRequest) -> SolveResult:
        result = runmfem(
            req.mesh,
            constituent_dict(req.resin),
            constituent_dict(req.core),
            face_material_dict(req.face),
            score_field=req.score_field,
            scoring=scoring_payload(req.scoring),
            return_details=True,
        )
        return as_solve_result(result, details=req.details)
