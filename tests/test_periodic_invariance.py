"""Periodicity regressions: tiling, offset, mirror, and an inert ky.

The 0ef6d56 failure mode drops the tapered z face and moves Ezz by tens of
percent. These tolerances sit under that and above element noise.
"""

import logging

import numpy as np
import pytest

from b3_core.core.mesh import create_grooved_mesh
from b3_core.solvers.numpy_fe.backend import runnumpy

CORE = {"E": 80e6, "nu": 0.3, "rho": 100.0}
RESIN = {"E": 3e9, "nu": 0.3, "rho": 1200.0}


def _mesh(*, dx=20.0, dy=20.0, xcuts=None, ycuts=None, kx=0.0, ky=0.0):
    return create_grooved_mesh(
        thickness=10.0,
        dx=dx,
        dy=dy,
        xcuts=[[0.0, 20.0, 6.0, 1.0]] if xcuts is None else xcuts,
        ycuts=[] if ycuts is None else ycuts,
        madd=(0.0,),
        tface=0.0,
        kx=kx,
        ky=ky,
    )


def _C(**kwargs) -> np.ndarray:
    return np.asarray(runnumpy(_mesh(**kwargs), RESIN, CORE).stiffness, dtype=float)


def _rel(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.abs(left).max()), float(np.abs(right).max()), 1.0)
    return float(np.abs(left - right).max() / scale)


def _close(left: np.ndarray, right: np.ndarray, tol: float, label: str) -> None:
    rel = _rel(left, right)
    assert rel < tol, f"{label} rel={rel:.3e}"


def test_one_pitch_matches_two_pitches_flat_and_curved():
    # Interior offset: a groove centered on x = 0 is a split kerf, and the
    # coarse background grid is not the same number of elements per pitch.
    for kx in (0.0, 0.008, -0.008):
        one = _C(dx=20.0, xcuts=[[5.0, 20.0, 6.0, 1.0]], kx=kx)
        two = _C(dx=40.0, xcuts=[[5.0, 20.0, 6.0, 1.0]], kx=kx)
        _close(one, two, 0.12, f"tiling kx={kx}")


def test_offset_is_a_phase_shift_flat_and_curved():
    for kx in (0.0, 0.008, -0.008):
        origin = _C(xcuts=[[5.0, 20.0, 6.0, 1.0]], kx=kx)
        shifted = _C(xcuts=[[12.0, 20.0, 6.0, 1.0]], kx=kx)
        _close(origin, shifted, 0.05, f"offset kx={kx}")


def test_bottom_mouth_positive_kx_matches_top_mouth_negative_kx():
    bottom = _C(xcuts=[[0.0, 20.0, 6.0, 1.0]], kx=0.008)
    top = _C(xcuts=[[0.0, 20.0, -6.0, 1.0]], kx=-0.008)
    _close(bottom, top, 1e-6, "mirror")


def test_ky_does_nothing_without_y_grooves():
    flat = _mesh(ky=0.0)
    bent = _mesh(ky=0.008)
    assert flat.n_cells == bent.n_cells
    assert set(np.round(flat.points[:, 0], 8)) == set(np.round(bent.points[:, 0], 8))
    assert set(np.round(flat.points[:, 1], 8)) == set(np.round(bent.points[:, 1], 8))
    left = np.asarray(runnumpy(flat, RESIN, CORE).stiffness, dtype=float)
    right = np.asarray(runnumpy(bent, RESIN, CORE).stiffness, dtype=float)
    _close(left, right, 1e-12, "ky inert")


def _small_case(xgr, ygr, kx=0.0, ky=0.0) -> dict:
    return {
        "dx": 20.0,
        "dy": 20.0,
        "thickness": 10.0,
        "xgr": xgr,
        "ygr": ygr,
        "core": CORE,
        "resin": RESIN,
        "madd": [0.0],
        "curvature": {"kx": kx, "ky": ky},
    }


@pytest.mark.fenicsx
def test_fenicsx_ezz_stays_near_flat_at_the_sentinel(caplog):
    """κ = 5e-5 must not collapse Ezz. That was the dropped-tie signature."""
    from b3_core import homogenize
    from b3_core.solvers import get_backend

    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    grooves = [[0.0, 20.0, 6.0, 1.0]]
    with caplog.at_level(logging.WARNING, logger="b3_core.pipeline"):
        flat = homogenize(_small_case(grooves, [], kx=0.0), backend="fenicsx")
        slight = homogenize(_small_case(grooves, [], kx=5e-5), backend="fenicsx")
    assert "falling back" not in caplog.text
    assert slight.properties["Ez"] == pytest.approx(flat.properties["Ez"], rel=0.05)


@pytest.mark.fenicsx
def test_biaxial_fenicsx_matches_mfem_with_pair_periodicity(caplog):
    from b3_core import homogenize
    from b3_core.solvers import get_backend

    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    if not get_backend("mfem").is_available():
        pytest.skip("MFEM is not installed")
    grooves = [[0.0, 20.0, 6.0, 1.0]]
    case = _small_case(grooves, grooves, kx=1e-3, ky=1e-3)
    with caplog.at_level(logging.WARNING, logger="b3_core.pipeline"):
        fenicsx = homogenize(case, backend="fenicsx")
        mfem = homogenize(case, backend="mfem", allow_pair_periodicity=True)
    assert "falling back" not in caplog.text
    _close(
        np.asarray(fenicsx.stiffness, dtype=float),
        np.asarray(mfem.stiffness, dtype=float),
        1e-4,
        "biaxial fenicsx vs mfem",
    )
