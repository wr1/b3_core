"""Optional solver jobs. The default suite skips these when the solver is absent."""

import logging

import numpy as np
import pytest

from b3_core.api import homogenize
from b3_core.cases import grid_scored, plain, uniaxial
from b3_core.pipeline import resolve_backend
from b3_core.solvers import get_backend

_KEYS = ("Exx", "Eyy", "Ezz", "Gxy", "Gxz", "Gyz")


def _compare(factory, backend: str):
    reference = homogenize(factory(), backend="mfem")
    other = homogenize(factory(), backend=backend)
    for key in _KEYS:
        assert other.engineering_constants[key] == pytest.approx(
            reference.engineering_constants[key], rel=2e-2
        )
    assert other.stiffness.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(other.stiffness) > 0.0)


@pytest.mark.fenicsx
def test_auto_leads_with_fenicsx():
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    assert resolve_backend(plain().input) == "fenicsx"
    assert resolve_backend(grid_scored().input) == "fenicsx"


@pytest.mark.fenicsx
def test_fenicsx_plain_matches_mfem():
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    _compare(plain, "fenicsx")


@pytest.mark.fenicsx
def test_fenicsx_uniaxial_matches_mfem():
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    _compare(uniaxial, "fenicsx")


def _fenicsx_stiffness(case, caplog):
    """Stiffness from an explicit fenicsx solve. A capability fallback is a failure."""
    with caplog.at_level(logging.WARNING, logger="b3_core.pipeline"):
        got = np.asarray(homogenize(case, backend="fenicsx").stiffness, dtype=float)
    assert "falling back" not in caplog.text
    return got


@pytest.mark.fenicsx
def test_fenicsx_orthotropic_matches_numpy(caplog):
    """Sharp orthotropic kerfs: the quadrature 6×6 path reproduces numpy."""
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    case = grid_scored(cell_size=None, with_halo=False)
    ref = np.asarray(homogenize(case, backend="numpy").stiffness, dtype=float)
    got = _fenicsx_stiffness(case, caplog)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-4


@pytest.mark.fenicsx
def test_fenicsx_halo_matches_numpy(caplog):
    """Graded resin halo on an orthotropic core reproduces numpy."""
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    case = grid_scored()
    ref = np.asarray(homogenize(case, backend="numpy").stiffness, dtype=float)
    got = _fenicsx_stiffness(case, caplog)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-4


@pytest.mark.fenicsx
def test_fenicsx_curved_halo_matches_mfem(caplog):
    """A tapered orthotropic halo matches MFEM. Both sit off numpy by O(kappa^2)."""
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    case = grid_scored().with_curvature(kx=2e-3)
    ref = np.asarray(homogenize(case, backend="mfem").stiffness, dtype=float)
    numpy_C = np.asarray(homogenize(case, backend="numpy").stiffness, dtype=float)
    got = _fenicsx_stiffness(case, caplog)
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() / scale < 1e-6
    assert np.abs(got - numpy_C).max() / scale < 5e-3


@pytest.mark.ccx
def test_ccx_plain_matches_mfem():
    if not get_backend("ccx").is_available():
        pytest.skip("CalculiX ccx is not on PATH")
    _compare(plain, "ccx")


@pytest.mark.ccx
def test_ccx_uniaxial_matches_mfem():
    if not get_backend("ccx").is_available():
        pytest.skip("CalculiX ccx is not on PATH")
    _compare(uniaxial, "ccx")
