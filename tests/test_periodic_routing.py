"""Curved cases do not silently pick a pair-based backend."""

import pytest

from b3_core.cases import plain, uniaxial
from b3_core.core.mesh import create_grooved_mesh
from b3_core.pipeline import resolve_backend
from b3_core.solvers.calculix.writer import periodic_bcs
from b3_core.solvers.fenicsx import FenicsxBackend
from b3_core.solvers.protocol import BackendCapabilityError


def _curved():
    return uniaxial().with_curvature(kx=1.0e-3).input


def test_curved_auto_uses_fenicsx_when_it_imports(monkeypatch):
    monkeypatch.setattr(FenicsxBackend, "is_available", lambda self: True)
    assert resolve_backend(_curved()) == "fenicsx"


def test_curved_auto_errors_without_fenicsx(monkeypatch):
    monkeypatch.setattr(FenicsxBackend, "is_available", lambda self: False)
    with pytest.raises(BackendCapabilityError, match="doctor"):
        resolve_backend(_curved())


def test_flat_auto_still_resolves_without_fenicsx(monkeypatch):
    monkeypatch.setattr(FenicsxBackend, "is_available", lambda self: False)
    assert resolve_backend(plain().input) == "mfem"


def test_explicit_mfem_and_ccx_reject_curvature():
    case = _curved()
    with pytest.raises(BackendCapabilityError, match="mfem"):
        resolve_backend(case, "mfem")
    with pytest.raises(BackendCapabilityError, match="ccx"):
        resolve_backend(case, "ccx")


def test_allow_pair_periodicity_keeps_mfem():
    case = _curved().model_copy(update={"allow_pair_periodicity": True})
    with pytest.warns(UserWarning, match="allow_pair_periodicity"):
        assert resolve_backend(case, "mfem") == "mfem"


def test_explicit_numpy_warns_and_is_allowed():
    with pytest.warns(UserWarning, match="O\\(k\\^2\\)"):
        assert resolve_backend(_curved(), "numpy") == "numpy"


def test_ccx_writer_rejects_a_tapered_mesh():
    common = dict(
        thickness=10.0,
        dx=20.0,
        dy=20.0,
        xcuts=[[0.0, 20.0, 6.0, 1.0]],
        ycuts=[],
        madd=(0.0,),
        tface=0.0,
    )
    periodic_bcs(create_grooved_mesh(**common, kx=0.0), [10, 11, 12])
    with pytest.raises(ValueError, match="node-pair periodicity"):
        periodic_bcs(create_grooved_mesh(**common, kx=2.0e-3), [10, 11, 12])
