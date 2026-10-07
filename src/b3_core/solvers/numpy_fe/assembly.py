"""Hex kinematics and periodic homogenisation (numpy)."""

from __future__ import annotations

import numpy as np

from b3_core.solvers.periodic import z_face_ties

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

# pyvista StructuredGrid (indexing='xy') reverses both z-faces relative to the
# VTK hex order in ``_HEX_NODES``. A coordinate octant sort recovered that on
# boxes and collided once a kerf taper put two corners in one octant.
_PYVISTA_TO_VTK = np.array([0, 3, 2, 1, 4, 7, 6, 5])


def _canonicalize(points, cells):
    """Put each structured-grid hex into the VTK order used by ``_HEX_NODES``.

    PyVista's structured-grid winding reverses both z-faces. Cells that already
    have a positive centre Jacobian are left alone, so a second call is a
    no-op and a tapered kerf is not reordered by corner octants.
    """
    cells = np.asarray(cells, dtype=np.int64)
    X = points[cells]
    jac = np.einsum("ab,ebc->eac", _DN_C.T, X)
    flip = np.linalg.det(jac) <= 0.0
    out = np.array(cells, copy=True)
    if np.any(flip):
        out[flip] = cells[flip][:, _PYVISTA_TO_VTK]
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
def _periodic_masters(points: np.ndarray, axes=(0, 1, 2)):
    """Map every node to a master (opposite faces tied); return (master_of, n_masters).

    ``axes`` selects which high faces are folded onto the low face before the
    coordinate hash. The z face is omitted when the kerf taper has moved it
    off the lattice; those nodes are interpolated instead.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = 1e-9 + 1e-6 * (hi - lo).max()
    folded = points.copy()
    for ax in axes:
        folded[np.abs(points[:, ax] - hi[ax]) < tol, ax] = lo[ax]
    keys = np.round((folded - lo) / max((hi - lo).max(), 1.0), 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def _xy_representatives(points: np.ndarray):
    """Node index of the low-x, low-y image of each node. z is not folded."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = 1e-9 + 1e-6 * (hi - lo).max()
    folded = np.array(points, dtype=float, copy=True)
    folded[np.abs(points[:, 0] - hi[0]) < tol, 0] = lo[0]
    folded[np.abs(points[:, 1] - hi[1]) < tol, 1] = lo[1]
    span = max(float((hi - lo).max()), 1.0)
    keys = np.round((folded - lo) / span, 6)
    rec = np.empty(len(points), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    rec["x"], rec["y"], rec["z"] = keys[:, 0], keys[:, 1], keys[:, 2]
    _, inv = np.unique(rec, return_inverse=True)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], inv))
    groups, first = np.unique(inv[order], return_index=True)
    rep_of = np.empty(int(groups.max()) + 1, dtype=np.int64)
    rep_of[groups] = order[first]
    return rep_of[inv], tol, lo, hi


def _z_prolongation(points: np.ndarray, ties):
    """Independent-dof weights for every node when z is interpolated.

    Returns ``masters`` ``(n, 4)`` compact independent-node ids, ``weights``
    ``(n, 4)``, and the compact id of the low corner that is pinned.
    x and y stay node-to-node. A ``z = H`` representative is the bilinear
    combination from ``ties``, reduced through the x/y ties.
    """
    rep, tol, lo, hi = _xy_representatives(points)
    n = len(points)
    on_top = np.abs(points[:, 2] - hi[2]) < tol
    is_rep = rep == np.arange(n)
    indep = np.flatnonzero(is_rep & ~on_top)
    compact = -np.ones(n, dtype=np.int64)
    compact[indep] = np.arange(len(indep))
    slave_row = {int(node): i for i, node in enumerate(ties.slave)}

    rep_terms: dict[int, list[tuple[int, float]]] = {
        int(node): [(int(compact[node]), 1.0)] for node in indep
    }
    for node in np.flatnonzero(is_rep & on_top):
        acc: dict[int, float] = {}
        row = slave_row[int(node)]
        for master, weight in zip(ties.masters[row], ties.weights[row], strict=True):
            if weight == 0.0:
                continue
            reduced = int(compact[rep[int(master)]])
            if reduced < 0:
                raise RuntimeError("z-face master is not an independent node")
            acc[reduced] = acc.get(reduced, 0.0) + float(weight)
        items = list(acc.items())
        if len(items) > 4:
            raise RuntimeError("z-face constraint needs more than 4 masters")
        rep_terms[int(node)] = items

    masters = np.zeros((n, 4), dtype=np.int64)
    weights = np.zeros((n, 4), dtype=float)
    for node in range(n):
        for term, (master, weight) in enumerate(rep_terms[int(rep[node])]):
            masters[node, term] = master
            weights[node, term] = weight
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise RuntimeError("periodic weights do not sum to 1")

    at_corner = np.flatnonzero(np.max(np.abs(points - lo), axis=1) < tol)
    if len(at_corner) != 1:
        raise RuntimeError("expected one node at the low corner")
    pin = int(compact[rep[int(at_corner[0])]])
    if pin < 0:
        raise RuntimeError("low corner is not an independent node")
    return masters, weights, pin


# --------------------------------------------------------------------------- #
# core homogenisation
# --------------------------------------------------------------------------- #
def _assemble_tied(cells, Ke, fe, master_of, n_master):
    """Scatter element matrices onto node-to-node periodic masters."""
    from scipy.sparse import csr_matrix

    n_elem = len(cells)
    edofs = np.empty((n_elem, 24), dtype=np.int64)
    for e, conn in enumerate(cells):
        mdofs = master_of[conn]
        edofs[e] = 3 * np.repeat(mdofs, 3) + np.tile([0, 1, 2], 8)
    ndof = 3 * n_master
    rows = np.repeat(edofs, 24, axis=1).reshape(n_elem, 24, 24)
    cols = np.tile(edofs, (1, 24)).reshape(n_elem, 24, 24)
    K = csr_matrix((Ke.ravel(), (rows.ravel(), cols.ravel())), shape=(ndof, ndof))
    L = np.zeros((ndof, 6))
    for e in range(n_elem):
        np.add.at(L, edofs[e], fe[e])
    return K, L, edofs


def _element_terms(conn, masters, weights):
    """Nonzero prolongation terms of one element: local dof, master dof, weight."""
    m = masters[conn]
    w = weights[conn]
    mdof = 3 * m[:, :, None] + np.arange(3)
    ww = np.broadcast_to(w[:, :, None], (8, 4, 3))
    local = np.broadcast_to(np.arange(24).reshape(8, 1, 3), (8, 4, 3))
    mask = ww.ravel() != 0.0
    return local.ravel()[mask], mdof.ravel()[mask], ww.ravel()[mask]


def _assemble_interpolated(cells, Ke, fe, masters, weights):
    """Scatter ``Cᵀ Ke C`` where z-face nodes are bilinear combinations."""
    from scipy.sparse import csr_matrix

    nterm = (weights > 0.0).sum(axis=1)
    term = np.argmax(weights, axis=1)
    one = masters[np.arange(len(masters)), term]
    simple = np.all(nterm[cells] == 1, axis=1)
    ndof = 3 * int(masters.max()) + 3
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    L = np.zeros((ndof, 6))
    if np.any(simple):
        md = one[cells[simple]]
        ed = 3 * np.repeat(md, 3, axis=1) + np.tile(np.arange(3), 8)
        rows.append(np.repeat(ed, 24, axis=1).ravel())
        cols.append(np.tile(ed, (1, 24)).ravel())
        data.append(Ke[simple].ravel())
        flat = fe[simple].reshape(-1, 6)
        np.add.at(L, ed.ravel(), flat)
    for e in np.flatnonzero(~simple):
        local, mdof, ww = _element_terms(cells[e], masters, weights)
        block = (ww[:, None] * ww[None, :]) * Ke[e][np.ix_(local, local)]
        nt = local.size
        rows.append(np.repeat(mdof, nt))
        cols.append(np.tile(mdof, nt))
        data.append(block.ravel())
        np.add.at(L, mdof, ww[:, None] * fe[e][local])
    K = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    )
    return K, L


def _nodal_fluctuation(W, masters, weights):
    """Prolong independent dofs ``W`` ``(ndof, 6)`` to ``(n_nodes, 3, 6)``."""
    n_ind = W.shape[0] // 3
    W3 = W.reshape(n_ind, 3, 6)
    nodal = np.zeros((len(masters), 3, 6))
    for term in range(masters.shape[1]):
        sel = weights[:, term] != 0.0
        nodal[sel] += weights[sel, term, None, None] * W3[masters[sel, term]]
    return nodal


def homogenize_aniso(points_m, cells, gp_C, *, _force_z_interp=False):
    """Periodic homogenisation with a per-Gauss-point 6x6 stiffness.

    ``gp_C`` is ``(n_elem, 8, 6, 6)`` — one stiffness per Gauss point, so a graded
    material field (e.g. the stochastic resin halo) integrates exactly. A
    ``(n_elem, 6, 6)`` array (constant per element) is broadcast. Returns
    (stiffness 6x6, info).

    x and y faces are tied node to node. z faces that the kerf taper has moved
    apart are tied by bilinear interpolation on the box face. A flat mesh keeps
    the coordinate tie.
    """
    from scipy.sparse.linalg import factorized

    cells = _canonicalize(points_m, cells)
    gp_C = np.asarray(gp_C, dtype=float)
    if gp_C.ndim == 3:
        gp_C = np.broadcast_to(gp_C[:, None], (len(cells), 8, 6, 6))
    n_elem = len(cells)
    ties = z_face_ties(points_m)
    use_interp = _force_z_interp or not ties.identity

    # Per-element B, detJ*w at each Gauss point, plus centre B and volume.
    Ke = np.zeros((n_elem, 24, 24))
    fe = np.zeros((n_elem, 24, 6))  # macro-strain load: integral B^T C eps0
    Bc = np.zeros((n_elem, 6, 24))  # B at element centre
    vol = np.zeros(n_elem)
    T1 = np.zeros((6, 6))  # integral of C (energy-reduction term)
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

    w_nodal = None
    if use_interp:
        masters, weights, pin = _z_prolongation(points_m, ties)
        K, L = _assemble_interpolated(cells, Ke, fe, masters, weights)
        ndof = K.shape[0]
        free = np.ones(ndof, dtype=bool)
        free[3 * pin : 3 * pin + 3] = False
    else:
        master_of, n_master = _periodic_masters(points_m)
        K, L, edofs = _assemble_tied(cells, Ke, fe, master_of, n_master)
        # Pin master 0's 3 dofs to kill rigid translation.
        free = np.ones(K.shape[0], dtype=bool)
        free[:3] = False

    Kff = K[free][:, free]
    solve = factorized(Kff.tocsc())
    W = np.zeros((K.shape[0], 6))
    rhs = -L[free]
    for k in range(6):
        W[free, k] = solve(rhs[:, k])

    # Total element strain at centre for each unit case: eps0_k + Bc @ w_k.
    elem_strain = np.zeros((n_elem, 6, 6))  # (elem, case, voigt)
    if use_interp:
        nodal = _nodal_fluctuation(W, masters, weights)
        w_nodal = nodal
        gathered = nodal[cells].reshape(n_elem, 24, 6)
        corr = np.einsum("eij,ejk->eik", Bc, gathered)
        elem_strain = _UNIT.T[None, :, :] + corr.transpose(0, 2, 1)
        master_of = None
    else:
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
    if w_nodal is not None:
        info["w_nodal"] = w_nodal
    return stiffness, info
