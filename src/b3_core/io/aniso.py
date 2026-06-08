#!/usr/bin/env python3
"""Anisotropic periodic-homogenisation backend (numpy + scipy).

A self-contained trilinear-hexahedral periodic homogeniser that accepts a full
6x6 stiffness per material phase, so the core/resin/face can be **orthotropic**
(unlike the isotropic-only MFEM/CCX integrators). It also returns the per-element
strain field for each unit load case, which the strain-based resin-grid failure
check builds on.

Voigt order matches the rest of the package: (xx, yy, zz, yz, xz, xy), engineering
shear (gamma = 2*eps). Geometry is scaled mm -> m so results are SI, exactly as
the MFEM backend does.

It also applies the kerf-damage halo: when the mesh carries a graded
``halo_fraction`` (foam cells opened by grid-scoring), each such cell gets a
distance-graded blend ``phi*C_resin + (1-phi)*C_foam`` with ``phi = porosity *
halo_fraction`` — a per-cell stiffness, which the periodic solver takes directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fenicsx import _properties_from_stiffness, LOAD_CASES

# VTK hexahedron corner parametric coordinates in [-1, 1] (matches grid.cells order).
_HEX_NODES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # 2-point Gauss
# Unit macroscopic strain (engineering Voigt) for each load case.
_UNIT = np.eye(6)


@dataclass(frozen=True)
class AnisoResult:
    properties: dict
    stiffness: np.ndarray
    compliance: np.ndarray
    displacements: dict | None = None
    points: np.ndarray | None = None
    # Per-element total strain under each unit load case (n_elem, 6 cases, 6 Voigt),
    # element attribute and volume — inputs to the resin failure check.
    elem_strain: np.ndarray | None = None
    elem_attr: np.ndarray | None = None
    elem_volume: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# constitutive matrices
# --------------------------------------------------------------------------- #
def isotropic_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def orthotropic_C(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    """6x6 stiffness from orthotropic engineering constants (axes 1=x,2=y,3=z).

    Voigt order (xx,yy,zz,yz,xz,xy): shear diagonals are G23, G13, G12.
    """
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1.0 / G23, 1.0 / G13, 1.0 / G12
    return np.linalg.inv(S)


def material_C(mat: dict) -> np.ndarray:
    """6x6 stiffness from a material dict (isotropic E/nu or orthotropic E1.. )."""
    if mat.get("E1") is not None:
        return orthotropic_C(
            mat["E1"], mat["E2"], mat["E3"],
            mat["G12"], mat["G13"], mat["G23"],
            mat["nu12"], mat["nu13"], mat["nu23"],
        )
    return isotropic_C(mat["E"], mat["nu"])


# --------------------------------------------------------------------------- #
# element kinematics
# --------------------------------------------------------------------------- #
def _shape_grads():
    """Reference dN/dxi at the 8 Gauss points -> (8 gp, 8 node, 3)."""
    out = []
    for zk in _GP:
        for ej in _GP:
            for xi in _GP:
                g = np.zeros((8, 3))
                for a, (xa, ya, za) in enumerate(_HEX_NODES):
                    g[a, 0] = 0.125 * xa * (1 + ya * ej) * (1 + za * zk)
                    g[a, 1] = 0.125 * ya * (1 + xa * xi) * (1 + za * zk)
                    g[a, 2] = 0.125 * za * (1 + xa * xi) * (1 + ya * ej)
                out.append(g)
    return np.array(out)


def _shape_grads_center():
    g = np.zeros((8, 3))
    for a, (xa, ya, za) in enumerate(_HEX_NODES):
        g[a, 0] = 0.125 * xa
        g[a, 1] = 0.125 * ya
        g[a, 2] = 0.125 * za
    return g


_DN = _shape_grads()       # (8, 8, 3)
_DN_C = _shape_grads_center()

# (bx + 2by + 4bz) corner sign-pattern -> local slot matching _HEX_NODES order.
_CANON_LUT = np.array([0, 1, 3, 2, 4, 5, 7, 6])


def _canonicalize(points, cells):
    """Reorder each axis-aligned hex's nodes into canonical VTK order.

    pyvista's `indexing='xy'` structured grids wind hexes inconsistently
    (negative Jacobian); for box-shaped cells we can recover the standard order
    from the node coordinates so the trilinear shape functions are valid.
    """
    out = cells.copy()
    for e, conn in enumerate(cells):
        c = points[conn]
        mid = c.mean(axis=0)
        key = ((c[:, 0] > mid[0]).astype(int)
               + 2 * (c[:, 1] > mid[1]).astype(int)
               + 4 * (c[:, 2] > mid[2]).astype(int))
        out[e, _CANON_LUT[key]] = conn
    return out


def _bmat(dN_xyz):
    """Engineering-Voigt B (6 x 24) from physical shape-function gradients (8x3)."""
    B = np.zeros((6, 24))
    for a in range(8):
        bx, by, bz = dN_xyz[a]
        c = 3 * a
        B[0, c] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz
        B[3, c + 1] = bz   # gamma_yz
        B[3, c + 2] = by
        B[4, c] = bz       # gamma_xz
        B[4, c + 2] = bx
        B[5, c] = by       # gamma_xy
        B[5, c + 1] = bx
    return B


# --------------------------------------------------------------------------- #
# periodic node identification
# --------------------------------------------------------------------------- #
def _periodic_masters(points: np.ndarray):
    """Map every node to a master (opposite faces tied); return (master_of, n_masters)."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = 1e-9 + 1e-6 * (hi - lo).max()
    folded = points.copy()
    for ax in range(3):
        folded[np.abs(points[:, ax] - hi[ax]) < tol, ax] = lo[ax]
    keys = np.round((folded - lo) / max((hi - lo).max(), 1.0), 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


# --------------------------------------------------------------------------- #
# core homogenisation
# --------------------------------------------------------------------------- #
def homogenize_aniso(points_m, cells, elem_C):
    """Periodic homogenisation with a per-element 6x6 stiffness.

    ``elem_C`` is ``(n_elem, 6, 6)`` — one stiffness per cell, so graded phases
    (e.g. the kerf-damage halo) are handled directly. Returns (stiffness, info).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import factorized

    cells = _canonicalize(points_m, cells)
    elem_C = np.asarray(elem_C, dtype=float)
    n_elem = len(cells)
    master_of, n_master = _periodic_masters(points_m)
    ndof = 3 * n_master

    # Per-element B, detJ*w at each Gauss point, plus centre B and volume.
    Ke = np.zeros((n_elem, 24, 24))
    fe = np.zeros((n_elem, 24, 6))          # macro-strain load: integral B^T C eps0
    Bc = np.zeros((n_elem, 6, 24))          # B at element centre
    vol = np.zeros(n_elem)
    edofs = np.zeros((n_elem, 24), dtype=np.int64)
    for e, conn in enumerate(cells):
        X = points_m[conn]                  # (8,3)
        C = elem_C[e]
        for gp in range(8):
            J = _DN[gp].T @ X               # (3,3)
            dN_xyz = _DN[gp] @ np.linalg.inv(J)
            B = _bmat(dN_xyz)
            w = abs(np.linalg.det(J))       # |detJ| * Gauss weight(=1)
            Ke[e] += (B.T @ C @ B) * w
            fe[e] += (B.T @ C) * w
            vol[e] += w
        Jc = _DN_C.T @ X
        Bc[e] = _bmat(_DN_C @ np.linalg.inv(Jc))
        mdofs = master_of[conn]
        edofs[e] = (3 * np.repeat(mdofs, 3) + np.tile([0, 1, 2], 8))

    # Assemble global K (periodic nodes accumulate via shared master dofs).
    rows = np.repeat(edofs, 24, axis=1).reshape(n_elem, 24, 24)
    cols = np.tile(edofs, (1, 24)).reshape(n_elem, 24, 24)
    K = csr_matrix(
        (Ke.ravel(), (rows.ravel(), cols.ravel())), shape=(ndof, ndof)
    )

    # Macro-strain load vectors L_k = sum_e fe @ eps0_k, assembled to master dofs.
    L = np.zeros((ndof, 6))
    for e in range(n_elem):
        np.add.at(L, edofs[e], fe[e])       # fe[e] is (24,6)

    # Pin master 0's 3 dofs to kill rigid translation; solve K w = -L on free dofs.
    free = np.ones(ndof, dtype=bool)
    free[:3] = False
    Kff = K[free][:, free]
    solve = factorized(Kff.tocsc())
    W = np.zeros((ndof, 6))
    rhs = -L[free]
    for k in range(6):
        W[free, k] = solve(rhs[:, k])

    # Total element strain at centre for each unit case: eps0_k + Bc @ w_k.
    elem_strain = np.zeros((n_elem, 6, 6))  # (elem, case, voigt)
    for e in range(n_elem):
        we = W[edofs[e]]                    # (24, 6)
        elem_strain[e] = _UNIT.T + (Bc[e] @ we).T   # row k = eps0_k + Bc w_k

    # Exact energy reduction (matches the MFEM backend): the volume-averaged
    # constituent stiffness plus the corrector coupling L^T W. Element-centre
    # strains above are kept only for the failure post-processing.
    total_vol = vol.sum()
    T1 = np.zeros((6, 6))
    for e in range(n_elem):
        T1 += vol[e] * elem_C[e]
    stiffness = (T1 + L.T @ W) / total_vol
    stiffness = 0.5 * (stiffness + stiffness.T)

    info = {
        "master_of": master_of, "W": W, "vol": vol,
        "elem_strain": elem_strain,
    }
    return stiffness, info


# --------------------------------------------------------------------------- #
# resin-grid failure check (Laustsen et al. 2014)
# --------------------------------------------------------------------------- #
# Allowable in-situ resin failure strain from uniaxial grid-scored tension tests
# (Laustsen et al. 2014, Table 3); the resin grid fails brittle far below bulk.
ALLOWABLE_RESIN_STRAIN = {"H60_resinA": 8443e-6, "H130_resinA": 13120e-6, "resinB": 5194e-6}


def _max_principal_strain(eps_voigt: np.ndarray) -> np.ndarray:
    """Max (most tensile) principal strain per row of engineering-Voigt strains."""
    e = np.asarray(eps_voigt, dtype=float).reshape(-1, 6)
    T = np.zeros((len(e), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = e[:, 0], e[:, 1], e[:, 2]
    T[:, 1, 2] = T[:, 2, 1] = 0.5 * e[:, 3]   # gamma/2
    T[:, 0, 2] = T[:, 2, 0] = 0.5 * e[:, 4]
    T[:, 0, 1] = T[:, 1, 0] = 0.5 * e[:, 5]
    return np.linalg.eigvalsh(T)[:, -1]


def resin_failure_index(
    result: "AnisoResult",
    *,
    macro_strain=None,
    macro_stress=None,
    allowable: float = ALLOWABLE_RESIN_STRAIN["H60_resinA"],
) -> dict:
    """Strain-based resin-grid failure check (Laustsen et al. 2014).

    Applies a macroscopic strain (engineering Voigt, order xx,yy,zz,yz,xz,xy) or
    stress to the homogenised RVE, reconstructs the per-element strain in the
    resin cells, and compares the maximum tensile principal strain to the
    allowable in-situ resin strain. `failure_index >= 1` predicts resin fracture.
    """
    if result.elem_strain is None:
        raise ValueError("need a return_details=True result for the failure check")
    if macro_strain is None:
        if macro_stress is None:
            raise ValueError("provide macro_strain or macro_stress")
        macro_strain = result.compliance @ np.asarray(macro_stress, dtype=float)
    eps0 = np.asarray(macro_strain, dtype=float).reshape(6)

    # Linear superposition of the six unit-case element strains.
    elem_eps = np.einsum("i,eiv->ev", eps0, result.elem_strain)
    resin = result.elem_attr == 2
    if not resin.any():
        return {"failure_index": 0.0, "max_principal_strain": 0.0, "n_resin": 0}
    maxp = _max_principal_strain(elem_eps[resin])
    worst = int(np.argmax(maxp))
    return {
        "failure_index": float(maxp.max() / allowable),
        "max_principal_strain": float(maxp.max()),
        "allowable": float(allowable),
        "n_resin": int(resin.sum()),
        "worst_resin_element": int(np.flatnonzero(resin)[worst]),
    }


def foam_porosity(core: dict, scoring: dict | None) -> float:
    """Resin-filled fraction of an opened foam cell = 1 - rho_foam/rho_solid."""
    solid = (scoring or {}).get("solid_density", 1400.0)
    return float(np.clip(1.0 - core["rho"] / solid, 0.0, 1.0))


def runnumpy(mesh, resin, core, face=None, *, scoring=None, return_details=False):
    """Drop-in for runmfem using the numpy anisotropic homogeniser.

    Accepts isotropic or orthotropic material dicts. When the mesh carries a
    kerf-damage ``halo_fraction`` (from ``scoring``), each halo cell gets a graded
    stiffness ``phi*C_resin + (1-phi)*C_foam`` with ``phi = porosity * fraction``.
    Mirrors the MFEM result and adds the per-element strain field for the failure
    check.
    """
    grid = mesh.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()
    cell_block = grid.cells.reshape((-1, 9))
    if not np.all(cell_block[:, 0] == 8):
        raise ValueError("numpy backend supports linear hexahedral cells only")
    points = np.asarray(grid.points, dtype=np.float64)
    cells = np.asarray(cell_block[:, 1:], dtype=np.int64)

    resin_cells = np.asarray(grid.cell_data["resin"], dtype=bool)
    face_cells = np.asarray(grid.cell_data["face"], dtype=bool)
    attr = np.ones(grid.n_cells, dtype=np.int64)
    attr[resin_cells] = 2
    if face is not None and face_cells.any():
        attr[face_cells] = 3

    C_core, C_resin = material_C(core), material_C(resin)
    elem_C = np.broadcast_to(C_core, (grid.n_cells, 6, 6)).copy()
    elem_C[attr == 2] = C_resin
    if (attr == 3).any():
        face_mat = dict(face) if face else {}
        face_mat.setdefault("E", 12_000_000_000.0)
        face_mat.setdefault("nu", 0.3)
        elem_C[attr == 3] = material_C(face_mat)

    # Graded kerf-damage halo: opened cells filled with resin (porosity-scaled).
    if "halo_fraction" in grid.cell_data:
        phi = foam_porosity(core, scoring) * np.asarray(grid.cell_data["halo_fraction"])
        m = phi > 0
        p = phi[m][:, None, None]
        elem_C[m] = p * C_resin + (1.0 - p) * C_core

    stiffness, info = homogenize_aniso(points, cells, elem_C)
    properties, compliance = _properties_from_stiffness(stiffness)

    if not return_details:
        return AnisoResult(properties, stiffness, compliance)

    displacements = {}
    master_of, W = info["master_of"], info["W"]
    for k, case in enumerate(LOAD_CASES):
        e0 = np.zeros((3, 3))
        v = _UNIT[k]
        e0[0, 0], e0[1, 1], e0[2, 2] = v[0], v[1], v[2]
        e0[1, 2] = e0[2, 1] = 0.5 * v[3]
        e0[0, 2] = e0[2, 0] = 0.5 * v[4]
        e0[0, 1] = e0[1, 0] = 0.5 * v[5]
        w_node = W[3 * master_of[:, None] + np.array([0, 1, 2]), k]
        displacements[case] = points @ e0 + w_node
    return AnisoResult(
        properties, stiffness, compliance, displacements, points,
        info["elem_strain"], attr, info["vol"],
    )
