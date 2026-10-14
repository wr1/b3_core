"""Mesh-consistency regressions for the grooved RVE.

1. k -> 0 on the curved path recovers the flat result (same z stations).
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
