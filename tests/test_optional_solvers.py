"""Optional solver jobs. The default suite skips these when the solver is absent."""

import numpy as np
import pytest

from b3_core.api import homogenize
from b3_core.cases import plain, uniaxial
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
def test_fenicsx_plain_matches_mfem():
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    _compare(plain, "fenicsx")


@pytest.mark.fenicsx
def test_fenicsx_uniaxial_matches_mfem():
    if not get_backend("fenicsx").is_available():
        pytest.skip("FEniCSx is not installed")
    _compare(uniaxial, "fenicsx")


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
