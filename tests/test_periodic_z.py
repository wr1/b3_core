"""Z-face periodicity after the kerf taper moves nodes off the lattice."""

import numpy as np
import pytest
import pyvista as pv

from b3_core.core.mesh import create_grooved_mesh
from b3_core.solvers.elasticity import isotropic_C
from b3_core.solvers.fenicsx import (
    _hex_space,
    _periodic_constraint,
    is_fenicsx_available,
    mpc_tie_report,
    runfenicsx,
)
from b3_core.solvers.mfem import is_mfem_available, runmfem
from b3_core.solvers.numpy_fe.assembly import homogenize_aniso
from b3_core.solvers.numpy_fe.backend import runnumpy
from b3_core.solvers.periodic import face_tol, z_face_ties

CORE = {"E": 80e6, "nu": 0.3, "rho": 100.0}
RESIN = {"E": 3e9, "nu": 0.3, "rho": 1200.0}


def _mesh(kx: float):
    return create_grooved_mesh(
        thickness=10.0,
        dx=20.0,
        dy=20.0,
        xcuts=[[0.0, 20.0, 6.0, 1.0]],
        ycuts=[],
        madd=(0.0,),
        tface=0.0,
        kx=kx,
    )


def _top_ids(points: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.abs(points[:, 2] - points[:, 2].max()) < face_tol(points))


def test_flat_z_ties_are_identity():
    points = np.asarray(_mesh(0.0).points)
    ties = z_face_ties(points)
    assert ties.identity
    assert len(ties.slave) == len(_top_ids(points))
    assert np.allclose(ties.weights.sum(axis=1), 1.0)
    assert np.all(ties.weights.max(axis=1) > 1.0 - 1e-8)


def test_taper_interpolates_every_z_node():
    points = np.asarray(_mesh(5e-5).points)
    ties = z_face_ties(points)
    assert not ties.identity
    assert len(ties.slave) == len(_top_ids(points))
    assert np.allclose(ties.weights.sum(axis=1), 1.0)
    # The mouth shift is larger than the coordinate-hash bin, and it is not a
    # node-to-node tie: some rows use more than one bottom node.
    span = (points.max(axis=0) - points.min(axis=0)).max()
    bot = points[np.abs(points[:, 2] - points[:, 2].min()) < face_tol(points)][:, :2]
    dist = np.min(
        np.linalg.norm(points[ties.slave, :2, None] - bot.T[None, :, :], axis=1),
        axis=1,
    )
    assert dist.max() > 1e-6 * span
    assert np.any((ties.weights > 1e-6).sum(axis=1) >= 2)

    a, b, c = 1.2, -0.3, 0.7
    masters = points[ties.masters]
    predicted = (ties.weights * (a + b * masters[:, :, 0] + c * masters[:, :, 1])).sum(
        axis=1
    )
    exact = a + b * points[ties.slave, 0] + c * points[ties.slave, 1]
    assert np.allclose(predicted, exact)


def test_weighted_path_matches_node_ties_on_a_flat_cube():
    x = np.linspace(0, 2, 3)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    grid = pv.StructuredGrid(X, Y, Z).cast_to_unstructured_grid()
    points = np.asarray(grid.points)
    cells = grid.cells.reshape(-1, 9)[:, 1:]
    C0 = isotropic_C(4e9, 0.3)
    elem_C = np.broadcast_to(C0, (len(cells), 6, 6))
    tied, _ = homogenize_aniso(points, cells, elem_C)
    interpolated, _ = homogenize_aniso(points, cells, elem_C, _force_z_interp=True)
    assert np.allclose(interpolated, tied, rtol=1e-8, atol=1.0)
    assert np.allclose(tied, C0, rtol=1e-9, atol=1.0)


def test_small_taper_does_not_collapse_ezz():
    flat = runnumpy(_mesh(0.0), RESIN, CORE)
    slight = runnumpy(_mesh(5e-5), RESIN, CORE)
    assert slight.properties["Ezz"] == pytest.approx(flat.properties["Ezz"], rel=0.05)
    # Large taper still factorises, and the z face stays fully constrained.
    runnumpy(_mesh(2e-3), RESIN, CORE)
    wide = z_face_ties(np.asarray(_mesh(2e-3).points))
    assert not wide.identity
    assert np.allclose(wide.weights.sum(axis=1), 1.0)
    assert len(wide.slave) == len(_top_ids(np.asarray(_mesh(2e-3).points)))


@pytest.mark.skipif(not is_mfem_available(), reason="MFEM not installed")
def test_taper_numpy_matches_mfem():
    mesh = _mesh(5e-5)
    numpy_C = runnumpy(mesh, RESIN, CORE).stiffness
    mfem_C = runmfem(mesh, RESIN, CORE, None, return_details=True).stiffness
    assert np.abs(numpy_C - mfem_C).max() / np.abs(mfem_C).max() < 1e-5


@pytest.mark.fenicsx
@pytest.mark.skipif(not is_fenicsx_available(), reason="FEniCSx is not installed")
def test_taper_fenicsx_matches_numpy():
    # κ = 5e-5 moves z-face nodes off the lattice. A dropped tie collapses Ezz
    # by tens of percent; the remaining gap is quadrature on the hexes.
    mesh = _mesh(5e-5)
    numpy_C = runnumpy(mesh, RESIN, CORE).stiffness
    fenicsx_C = runfenicsx(mesh, RESIN, CORE, return_details=True).stiffness
    rel = np.abs(numpy_C - fenicsx_C).max() / np.abs(numpy_C).max()
    assert rel < 5e-3
    assert np.all(np.linalg.eigvalsh(fenicsx_C) > 0.0)
    wide = _mesh(2e-3)
    numpy_wide = runnumpy(wide, RESIN, CORE).stiffness
    fenicsx_wide = runfenicsx(wide, RESIN, CORE, return_details=True).stiffness
    wide_rel = np.abs(numpy_wide - fenicsx_wide).max() / np.abs(numpy_wide).max()
    assert wide_rel < 5e-3
    flat = runfenicsx(_mesh(0.0), RESIN, CORE, return_details=True)
    assert flat.properties["Ezz"] == pytest.approx(
        runnumpy(_mesh(0.0), RESIN, CORE).properties["Ezz"], rel=5e-3
    )


def _mpc_rows(mpc):
    """One component-0 slave: weights and master block ids.

    The other two components carry the same weights. A coordinate tie would
    be a single weight of 1; an in-cell image has several.
    """
    slaves = np.asarray(mpc.slaves)
    master_graph = mpc.masters
    coeffs, offsets = mpc.coefficients()
    coeffs = np.asarray(coeffs, dtype=float)
    offsets = np.asarray(offsets)
    rows = []
    for dof in slaves[slaves % 3 == 0]:
        weights = coeffs[offsets[dof] : offsets[dof + 1]]
        masters = np.asarray(master_graph.links(int(dof))) // 3
        for comp in (1, 2):
            other = int(dof) + comp
            same = coeffs[offsets[other] : offsets[other + 1]]
            assert np.allclose(same, weights)
        rows.append((int(dof) // 3, weights, masters))
    return rows


def _periodic_image(coords, node, points):
    image = coords[node, :2].copy()
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    for axis in (0, 1):
        if np.isclose(coords[node, axis], upper[axis], atol=1e-12):
            image[axis] -= upper[axis] - lower[axis]
    return image


def _engineering_macro(case: str) -> np.ndarray:
    strain = np.zeros((3, 3))
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
        raise ValueError(case)
    return strain


@pytest.mark.fenicsx
@pytest.mark.skipif(not is_fenicsx_available(), reason="FEniCSx is not installed")
def test_fenicsx_returns_periodic_displacement_field():
    """u = E·(x − x0) + w on the mesh vertices, and w matches across x."""
    mesh = _mesh(0.0)
    result = runfenicsx(mesh, RESIN, CORE, return_details=True)
    pts = result.points
    assert np.allclose(pts, np.asarray(mesh.points) * 1e-3)
    assert set(result.displacements) == {"xx", "yy", "zz", "yz", "xz", "xy"}
    assert result.displacements["xy"].shape == pts.shape

    origin = pts.min(axis=0)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    fmin = np.flatnonzero(np.isclose(pts[:, 0], xmin))
    fmax = np.flatnonzero(np.isclose(pts[:, 0], xmax))

    def key(i):
        return (round(float(pts[i, 1]), 9), round(float(pts[i, 2]), 9))

    by_yz = {key(i): i for i in fmax}
    shifted = pts - origin
    for case in ("xx", "xy"):
        w = result.displacements[case] - shifted @ _engineering_macro(case)
        paired = [i for i in fmin if key(i) in by_yz]
        assert paired
        worst = max(np.abs(w[i] - w[by_yz[key(i)]]).max() for i in paired)
        assert worst < 1e-8


@pytest.mark.fenicsx
@pytest.mark.skipif(not is_fenicsx_available(), reason="FEniCSx is not installed")
def test_taper_fenicsx_z_face_uses_the_interpolant():
    # The constraint the solve uses, not a coordinate match. A dropped or
    # snapped z node fails the linear reproduction below: the image sits
    # off the lattice by the mouth shift.
    _domain, space, points, _grid = _hex_space(_mesh(5e-5))
    mpc, _pin, _bounds = _periodic_constraint(space, points)
    coords = np.asarray(space.tabulate_dof_coordinates())
    by_node = {node: (weights, masters) for node, weights, masters in _mpc_rows(mpc)}

    zmax = points[:, 2].max()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    top = np.flatnonzero(np.isclose(coords[:, 2], zmax, atol=1e-12))
    assert set(top).issubset(by_node)

    multi = 0
    for node in top:
        weights, masters = by_node[node]
        assert np.allclose(weights.sum(), 1.0)
        image = _periodic_image(coords, node, points)
        predicted = (weights[:, None] * coords[masters, :2]).sum(axis=0)
        assert np.allclose(predicted, image, atol=1e-8)
        if np.count_nonzero(np.abs(weights) > 1e-8) >= 2:
            multi += 1
            gap = np.min(np.linalg.norm(coords[masters, :2] - image, axis=1))
            assert gap > 1e-7
    assert multi >= 1
    tapered = {row["face"]: row for row in mpc_tie_report(mpc, coords)}
    assert tapered["z"]["slaves"] == tapered["z"]["top_nodes"]
    assert tapered["z"]["ok"]
    assert tapered["z"]["multi_node_rows"] >= 1
    assert tapered["z"]["identity"] is False

    # x = dx, off the other high faces, is still an exact node copy.
    side = [
        node
        for node in by_node
        if np.isclose(coords[node, 0], xmax, atol=1e-12)
        and not np.isclose(coords[node, 2], zmax, atol=1e-12)
        and not np.isclose(coords[node, 1], ymax, atol=1e-12)
    ]
    assert side
    for node in side:
        weights, masters = by_node[node]
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)
        assert np.allclose(coords[masters[0], 1:], coords[node, 1:])

    _domain, flat_space, flat_points, _grid = _hex_space(_mesh(0.0))
    flat_mpc, _pin, _bounds = _periodic_constraint(flat_space, flat_points)
    flat_coords = np.asarray(flat_space.tabulate_dof_coordinates())
    flat_rows = {node: (w, m) for node, w, m in _mpc_rows(flat_mpc)}
    flat_top = np.flatnonzero(
        np.isclose(flat_coords[:, 2], flat_points[:, 2].max(), atol=1e-12)
    )
    for node in flat_top:
        weights, masters = flat_rows[node]
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)
        image = _periodic_image(flat_coords, node, flat_points)
        assert np.allclose(flat_coords[masters[0], :2], image, atol=1e-8)
    flat_ties = {row["face"]: row for row in mpc_tie_report(flat_mpc, flat_coords)}
    assert flat_ties["z"]["identity"]
    assert flat_ties["z"]["multi_node_rows"] == 0
    assert flat_ties["z"]["slaves"] == flat_ties["z"]["top_nodes"]
