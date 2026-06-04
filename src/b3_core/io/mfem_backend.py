#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from .fenicsx import _properties_from_stiffness, LOAD_CASES


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
            "MFEM backend requires PyMFEM. Install it with "
            "`pip install mfem` (or `uv sync --extra mfem`) and rerun with "
            "backend='mfem'."
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


def _element_attributes(grid, face):
    """Tag cells: foam = 1, resin = 2, face skin = 3."""
    resin_cells = np.asarray(grid.cell_data["resin"], dtype=bool)
    face_cells = np.asarray(grid.cell_data["face"], dtype=bool)
    attr = np.ones(grid.n_cells, dtype=np.int64)
    attr[resin_cells] = 2
    if face is not None and face_cells.any():
        attr[face_cells] = 3
    return attr


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


def runmfem(mesh, resin, core, face=None, *, return_details=False):
    """Solve six periodic-homogenisation cases with MFEM (PyMFEM).

    Uses true periodic boundary conditions via an MFEM periodic mesh and an
    energy-based reduction of the corrector solves, returning the same property
    keys as the CCX postprocessor. The backend is serial, linear-hexahedral and
    isotropic per material, and imports PyMFEM lazily so the package stays
    usable in CCX-only environments.
    """

    mfem = _require_mfem()

    points, cells, grid = _vtk_hexahedra(mesh)
    attr = _element_attributes(grid, face)

    lengths = points.max(axis=0) - points.min(axis=0)
    for axis, name in enumerate("xyz"):
        if len(np.unique(np.round(points[:, axis], 12))) < 3:
            raise ValueError(
                f"MFEM backend needs at least two elements along {name} for "
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
    displacements = {} if return_details else None
    if return_details:
        vmap = base.CreatePeriodicVertexMapping(translations)
        v2v = np.array([vmap[i] for i in range(len(points))], dtype=np.int64)
        pverts = np.array(
            [periodic.GetVertexArray(i) for i in range(periodic.GetNV())]
        )
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

    max_attr = periodic.attributes.Max()
    lam_by_attr, mu_by_attr = _material_arrays(core, resin, face, max_attr)
    lam_coeff = mfem.PWConstCoefficient(mfem.Vector(lam_by_attr.tolist()))
    mu_coeff = mfem.PWConstCoefficient(mfem.Vector(mu_by_attr.tolist()))

    bilinear = mfem.BilinearForm(fes)
    bilinear.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coeff, mu_coeff))
    bilinear.Assemble()

    # Pin the three DOFs of one node to remove the rigid-body translations of
    # the fully periodic cell (a 3-torus has no periodic rotation modes).
    pinned = mfem.intArray([fes.DofToVDof(0, comp) for comp in range(3)])

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
        corrector_data = corrector.GetDataArray().copy()
        correctors.append(corrector_data)
        if return_details:
            # total displacement u = E.x + w (the periodic fluctuation) on the grid
            displacements[case] = points @ strain + corrector_data[base_to_vdof]

    stiffness = np.zeros((6, 6), dtype=np.float64)
    for k, case_k in enumerate(LOAD_CASES):
        strain_k = _macro_strain(case_k)
        for l, case_l in enumerate(LOAD_CASES):
            strain_l = _macro_strain(case_l)
            energy = sum(
                volume_by_attr[p]
                * (
                    lame_by_attr[p][0] * np.trace(strain_k) * np.trace(strain_l)
                    + 2.0 * lame_by_attr[p][1] * np.sum(strain_k * strain_l)
                )
                for p in present
            )
            stiffness[k, l] = (
                energy + load_vectors[k].dot(correctors[l])
            ) / total_volume

    stiffness = 0.5 * (stiffness + stiffness.T)
    properties, compliance = _properties_from_stiffness(stiffness)
    if return_details:
        return MfemResult(properties, stiffness, compliance, displacements, points)
    return properties
