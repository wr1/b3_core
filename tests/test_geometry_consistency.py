"""Mesh-consistency regressions for the grooved RVE.

1. k -> 0 on the curved path recovers the flat result: flat and curved meshes
   share the same z stations (``kerf_z_elems`` over each groove depth).
2. A pure groove-offset shift of a periodic pattern does not change C.
3. The halo response is smooth in the reach (the thin outer-face band is
   resolved in z, so Gauss points do not jump in or out of it).
"""

import numpy as np
import pytest

from b3_core import homogenize
from b3_core.core.mesh import create_grooved_mesh

CORE = {"E": 85e6, "nu": 0.3, "rho": 80.0}
RESIN = {"E": 3.2e9, "nu": 0.35, "rho": 1150.0}
KEYS = ("Ex", "Ey", "Ez", "Gxz", "Gyz", "Gxy")


def _case(offset=10.0, kx=0.0, ky=0.0, cell_size=None):
    core = dict(CORE)
    if cell_size is not None:
        core["cell_size"] = cell_size
    return {
        "dx": 20.0,
        "dy": 20.0,
        "thickness": 10.0,
        "xgr": [[offset, 20.0, -8.0, 1.0]],
        "ygr": [[offset, 20.0, -8.0, 1.0]],
        "core": core,
        "resin": RESIN,
        "curvature": {"kx": kx, "ky": ky},
    }


def _props(case, backend):
    res = homogenize(case, backend=backend)
    return np.array([float(res.properties[k]) for k in KEYS])


def _maxrel(a, b):
    return float(np.max(np.abs(a / b - 1.0)))


def _require(backend):
    from b3_core.solvers import get_backend

    if not get_backend(backend).is_available():
        pytest.skip(f"{backend} is not installed")


def test_flat_and_near_zero_curved_mesh_share_z_stations():
    kw = dict(
        thickness=10.0,
        dx=20.0,
        dy=20.0,
        xcuts=[[10.0, 20.0, -8.0, 1.0]],
        ycuts=[[10.0, 20.0, -8.0, 1.0]],
        madd=(0.0,),
        tface=0.0,
    )
    flat = create_grooved_mesh(**kw)
    bent = create_grooved_mesh(**kw, kx=1e-9, ky=1e-9)
    assert flat.n_cells == bent.n_cells
    np.testing.assert_allclose(
        np.unique(flat.points[:, 2]), np.unique(bent.points[:, 2]), atol=1e-9
    )


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("fenicsx", marks=pytest.mark.fenicsx)]
)
def test_near_zero_curvature_recovers_flat(backend):
    _require(backend)
    flat = _props(_case(cell_size=0.5), backend)
    near = _props(_case(cell_size=0.5, kx=1e-9, ky=1e-9), backend)
    assert _maxrel(near, flat) < 0.005


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("fenicsx", marks=pytest.mark.fenicsx)]
)
def test_groove_offset_is_a_phase_shift(backend):
    _require(backend)
    centred = _props(_case(offset=10.0, cell_size=0.5), backend)
    for offset in (0.0, 5.0):
        shifted = _props(_case(offset=offset, cell_size=0.5), backend)
        # Was up to 18 % on the legacy coarse grid; now element noise only.
        assert _maxrel(shifted, centred) < 0.01, offset


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("fenicsx", marks=pytest.mark.fenicsx)]
)
def test_halo_response_is_smooth_and_monotone(backend):
    _require(backend)
    reach = np.linspace(0.6, 1.2, 7)
    ez = np.array([_props(_case(cell_size=r), backend)[2] for r in reach])
    steps = np.diff(ez)
    assert np.all(steps > 0)
    # Consecutive steps grow gently; the old Gauss-point jump was ~7x.
    assert np.all(steps[1:] / steps[:-1] < 1.5)


def test_kerf_z_elems_sets_stations_and_pinch_adds_one():
    kw = dict(
        thickness=10.0,
        dx=20.0,
        dy=20.0,
        xcuts=[[10.0, 20.0, -8.0, 1.0]],
        ycuts=[],
        madd=(0.0,),
        tface=0.0,
    )
    z4 = np.unique(create_grooved_mesh(**kw, kerf_z_elems=4).points[:, 2])
    # Root at z = 2, mouth at z = 10: four equal elements over the depth.
    np.testing.assert_allclose(z4, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], atol=1e-9)
    # Closing past full pinch: hw(z) kinks at MIN_HW (z = 6.99), one extra station.
    pinch = create_grooved_mesh(**kw, kerf_z_elems=4, kx=-0.01)
    assert len(np.unique(np.round(pinch.points[:, 2], 9))) == len(z4) + 1
    # Opening (no pinch) keeps the flat stations.
    opening = create_grooved_mesh(**kw, kerf_z_elems=4, kx=0.01)
    assert len(np.unique(np.round(opening.points[:, 2], 9))) == len(z4)


@pytest.mark.fenicsx
def test_fenicsx_opening_curvature_matches_numpy():
    """Opening kerfs (top mouth, kx > 0): FEniCSx and numpy must agree."""
    _require("fenicsx")
    case = _case(cell_size=0.5, kx=1e-3)
    fx = _props(case, "fenicsx")
    npy = _props(case, "numpy")
    assert _maxrel(fx, npy) < 0.01


@pytest.mark.fenicsx
def test_fenicsx_opening_without_halo_matches_numpy():
    """wr1/b3_core#3: no grid lines near the periodic boundary."""
    _require("fenicsx")
    case = _case(kx=1e-3)
    assert _maxrel(_props(case, "fenicsx"), _props(case, "numpy")) < 0.01


@pytest.mark.fenicsx
@pytest.mark.parametrize("cell_size", [None, 0.5], ids=["no-halo", "halo"])
@pytest.mark.parametrize(
    "kx,ky",
    [(3e-3, 0.0), (3e-3, 3e-3), (-3e-3, 0.0), (-3e-3, -3e-3)],
    ids=["open-x", "open-xy", "close-x", "close-xy"],
)
def test_fenicsx_curved_matches_numpy(kx, ky, cell_size):
    _require("fenicsx")
    case = _case(kx=kx, ky=ky, cell_size=cell_size)
    assert _maxrel(_props(case, "fenicsx"), _props(case, "numpy")) < 0.01


@pytest.mark.fenicsx
@pytest.mark.parametrize("kx", [3e-3, -3e-3], ids=["open", "close"])
def test_fenicsx_curved_matches_numpy_on_legacy_coarse_grid(kx):
    _require("fenicsx")
    case = _case(kx=kx)
    case.update(mesh_h_max=None, kerf_z_elems=1)
    assert _maxrel(_props(case, "fenicsx"), _props(case, "numpy")) < 0.01


@pytest.mark.fenicsx
def test_fenicsx_periodic_masters_are_never_slaves():
    """Chained ties (a master that is itself a slave) must be flattened."""
    _require("fenicsx")
    from b3_core.loaders import normalize_case
    from b3_core.pipeline import prepare
    from b3_core.solvers.fenicsx import _hex_space, _periodic_constraint

    case = _case(kx=3e-3, ky=3e-3)
    case.update(mesh_h_max=None, kerf_z_elems=1)
    case_in, _ = normalize_case(case)
    _domain, space, points, _grid = _hex_space(prepare(case_in).mesh)
    mpc, _pin, _bounds = _periodic_constraint(space, points)
    slaves = set(np.asarray(mpc.slaves).tolist())
    masters = set(np.asarray(mpc.masters.array).tolist())
    assert slaves
    assert not slaves & masters
