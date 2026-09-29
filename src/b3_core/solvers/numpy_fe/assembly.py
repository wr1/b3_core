"""Hex kinematics and periodic homogenisation (numpy)."""

from __future__ import annotations

import numpy as np

# VTK hexahedron corner parametric coordinates in [-1, 1] (matches grid.cells order).
_HEX_NODES = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # 2-point Gauss
# Unit macroscopic strain (engineering Voigt) for each load case.
_UNIT = np.eye(6)


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


def _shape_values():
    """Trilinear shape-function values at the 8 Gauss points -> (8 gp, 8 node)."""
    out = []
    for zk in _GP:
        for ej in _GP:
            for xi in _GP:
                N = np.array(
                    [
                        0.125 * (1 + xa * xi) * (1 + ya * ej) * (1 + za * zk)
                        for xa, ya, za in _HEX_NODES
                    ]
                )
                out.append(N)
    return np.array(out)


_DN = _shape_grads()  # (8, 8, 3)
_DN_C = _shape_grads_center()
_N = _shape_values()  # (8 gp, 8 node)
SHAPE_N = _N
UNIT = _UNIT

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
        key = (
            (c[:, 0] > mid[0]).astype(int)
            + 2 * (c[:, 1] > mid[1]).astype(int)
            + 4 * (c[:, 2] > mid[2]).astype(int)
        )
        out[e, _CANON_LUT[key]] = conn
    return out


canonicalize = _canonicalize


def _bmat(dN_xyz):
    """Engineering-Voigt B (6 x 24) from physical shape-function gradients (8x3)."""
    B = np.zeros((6, 24))
    for a in range(8):
        bx, by, bz = dN_xyz[a]
        c = 3 * a
        B[0, c] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz
        B[3, c + 1] = bz  # gamma_yz
        B[3, c + 2] = by
        B[4, c] = bz  # gamma_xz
        B[4, c + 2] = bx
        B[5, c] = by  # gamma_xy
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
def homogenize_aniso(points_m, cells, gp_C):
    """Periodic homogenisation with a per-Gauss-point 6x6 stiffness.

    ``gp_C`` is ``(n_elem, 8, 6, 6)`` — one stiffness per Gauss point, so a graded
    material field (e.g. the stochastic resin halo) integrates exactly. A
    ``(n_elem, 6, 6)`` array (constant per element) is broadcast. Returns
    (stiffness 6x6, info).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import factorized

    cells = _canonicalize(points_m, cells)
    gp_C = np.asarray(gp_C, dtype=float)
    if gp_C.ndim == 3:
        gp_C = np.broadcast_to(gp_C[:, None], (len(cells), 8, 6, 6))
    n_elem = len(cells)
    master_of, n_master = _periodic_masters(points_m)
    ndof = 3 * n_master

    # Per-element B, detJ*w at each Gauss point, plus centre B and volume.
    Ke = np.zeros((n_elem, 24, 24))
    fe = np.zeros((n_elem, 24, 6))  # macro-strain load: integral B^T C eps0
    Bc = np.zeros((n_elem, 6, 24))  # B at element centre
    vol = np.zeros(n_elem)
    T1 = np.zeros((6, 6))  # integral of C (energy-reduction term)
    edofs = np.zeros((n_elem, 24), dtype=np.int64)
    for e, conn in enumerate(cells):
        X = points_m[conn]  # (8,3)
        for gp in range(8):
            J = _DN[gp].T @ X  # (3,3)
            dN_xyz = _DN[gp] @ np.linalg.inv(J)
            B = _bmat(dN_xyz)
            w = abs(np.linalg.det(J))  # |detJ| * Gauss weight(=1)
            C = gp_C[e, gp]
            Ke[e] += (B.T @ C @ B) * w
            fe[e] += (B.T @ C) * w
            vol[e] += w
            T1 += C * w
        Jc = _DN_C.T @ X
        Bc[e] = _bmat(_DN_C @ np.linalg.inv(Jc))
        mdofs = master_of[conn]
        edofs[e] = 3 * np.repeat(mdofs, 3) + np.tile([0, 1, 2], 8)

    # Assemble global K (periodic nodes accumulate via shared master dofs).
    rows = np.repeat(edofs, 24, axis=1).reshape(n_elem, 24, 24)
    cols = np.tile(edofs, (1, 24)).reshape(n_elem, 24, 24)
    K = csr_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())), shape=(ndof, ndof))

    # Macro-strain load vectors L_k = sum_e fe @ eps0_k, assembled to master dofs.
    L = np.zeros((ndof, 6))
    for e in range(n_elem):
        np.add.at(L, edofs[e], fe[e])  # fe[e] is (24,6)

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
        we = W[edofs[e]]  # (24, 6)
        elem_strain[e] = _UNIT.T + (Bc[e] @ we).T  # row k = eps0_k + Bc w_k

    # Exact energy reduction (matches the MFEM backend): the integral of the
    # constituent stiffness (T1, accumulated over Gauss points above) plus the
    # corrector coupling L^T W. Element-centre strains feed the failure check.
    total_vol = vol.sum()
    stiffness = (T1 + L.T @ W) / total_vol
    stiffness = 0.5 * (stiffness + stiffness.T)

    info = {
        "master_of": master_of,
        "W": W,
        "vol": vol,
        "elem_strain": elem_strain,
    }
    return stiffness, info
