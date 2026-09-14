import numpy as np
import pytest

import b3_core.core.cprop as cprop_module
from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import mfem_backend


def test_mfem_reports_missing_dependency():
    if mfem_backend.is_mfem_available():
        pytest.skip("MFEM is installed in this environment")

    with pytest.raises(mfem_backend.MfemUnavailableError):
        mfem_backend.runmfem(None, {"E": 4e9, "nu": 0.3}, {"E": 4e9, "nu": 0.3})


def test_cprop_mfem_validation_dispatch(monkeypatch, tmp_path):
    calls = []

    def fake_mfem(mesh, dct, status=None):
        calls.append("mfem")
        return {"Exx": 100.0, "Gxy": 50.0}

    def fake_ccx(mesh, name, dct, status=None):
        calls.append("ccx")
        return {"Exx": 101.0, "Gxy": 49.0}

    monkeypatch.setattr(cprop_module, "create_grooved_mesh", lambda *a, **k: object())
    monkeypatch.setattr(
        cprop_module,
        "geom_analysis",
        lambda mesh: {"area_increase": 1.0, "resin_vf": 0.0},
    )
    monkeypatch.setattr(cprop_module, "_run_mfem_backend", fake_mfem)
    monkeypatch.setattr(cprop_module, "_run_ccx_backend", fake_ccx)

    cfg = tmp_path / "case.json"
    cfg.write_text(
        """{
            "dx": 50.0,
            "dy": 50.0,
            "thickness": 30.0,
            "xgr": [],
            "ygr": [],
            "core": {"E": 4000000000.0, "nu": 0.3, "rho": 100.0},
            "resin": {"E": 4000000000.0, "nu": 0.3, "rho": 1100.0},
            "backend": "mfem",
            "validate_with_ccx": true
        }"""
    )

    out = cprop_module.cprop(str(cfg))

    assert calls == ["mfem", "ccx"]
    assert out["ccx_validation"]["passed"] is True
    # the comparison labels the alternate backend by name
    assert "mfem" in out["ccx_validation"]["properties"]["Exx"]


@pytest.mark.skipif(not mfem_backend.is_mfem_available(), reason="MFEM not installed")
def test_mfem_recovers_isotropic_tensor():
    """A homogeneous RVE (no grooves, no skin, core == resin) must recover the
    input isotropic tensor under true periodic BCs."""
    e, nu = 4.0e9, 0.3
    g = e / (2.0 * (1.0 + nu))
    mat = {"E": e, "nu": nu, "rho": 100.0}
    mesh = create_grooved_mesh(
        thickness=30.0,
        dx=50.0,
        dy=50.0,
        xcuts=[],
        ycuts=[],
        madd=[-0.3, 0, 0.3],
        tface=0.0,
    )

    props = mfem_backend.runmfem(mesh, mat, mat, None)

    for key in ("Exx", "Eyy", "Ezz"):
        assert props[key] == pytest.approx(e, rel=1e-3)
    for key in ("Gyz", "Gxz", "Gxy"):
        assert props[key] == pytest.approx(g, rel=1e-3)
    for key in ("nuxy", "nuyz", "nuzx", "nuyx", "nuxz", "nuzy"):
        assert props[key] == pytest.approx(nu, rel=1e-3)


@pytest.mark.skipif(not mfem_backend.is_mfem_available(), reason="MFEM not installed")
def test_mfem_grooved_is_orthotropic_and_positive_definite():
    """Symmetric crossed grooves filled with stiffer resin give a symmetric,
    positive-definite stiffness with Exx == Eyy and stiffening over the core."""
    core = {"E": 4e9, "nu": 0.3, "rho": 100.0}
    resin = {"E": 40e9, "nu": 0.3, "rho": 1100.0}
    mesh = create_grooved_mesh(
        thickness=30.0,
        dx=50.0,
        dy=50.0,
        xcuts=[[10, 5, 2, 1]],
        ycuts=[[10, 5, 2, 1]],
        madd=[-0.3, 0, 0.3],
        tface=0.0,
    )

    result = mfem_backend.runmfem(mesh, resin, core, None, return_details=True)

    assert np.allclose(result.stiffness, result.stiffness.T, rtol=1e-6, atol=1.0)
    eigenvalues = np.linalg.eigvalsh(result.stiffness)
    assert np.all(eigenvalues > 0.0)
    assert result.properties["Exx"] == pytest.approx(result.properties["Eyy"], rel=1e-3)
    assert result.properties["Exx"] >= core["E"]


@pytest.mark.skipif(not mfem_backend.is_mfem_available(), reason="MFEM not installed")
def test_mfem_returns_periodic_displacement_field():
    """return_details exposes u = E.x + w on the grid, and the recovered
    fluctuation w must match on opposite faces (the periodic-BC invariant)."""
    mat = {"E": 4e9, "nu": 0.3, "rho": 100.0}
    resin = {"E": 40e9, "nu": 0.3, "rho": 1100.0}
    mesh = create_grooved_mesh(
        thickness=30.0,
        dx=50.0,
        dy=50.0,
        xcuts=[[10, 10, 8, 3]],
        ycuts=[[10, 10, 8, 3]],
        madd=[-0.3, 0, 0.3],
        tface=0.0,
    )
    result = mfem_backend.runmfem(mesh, resin, mat, None, return_details=True)

    pts = result.points
    assert set(result.displacements) == {"xx", "yy", "zz", "yz", "xz", "xy"}
    assert result.displacements["xy"].shape == pts.shape

    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    fmin = np.where(np.isclose(pts[:, 0], xmin))[0]
    fmax = np.where(np.isclose(pts[:, 0], xmax))[0]

    def key(i):
        return (round(float(pts[i, 1]), 9), round(float(pts[i, 2]), 9))

    by_yz = {key(i): i for i in fmax}
    for case in ("xx", "xy"):
        strain = mfem_backend._macro_strain(case)
        w = result.displacements[case] - pts @ strain  # subtract the macro part
        worst = max(
            np.abs(w[i] - w[by_yz[key(i)]]).max() for i in fmin if key(i) in by_yz
        )
        assert worst < 1e-9  # w is periodic across x=0 / x=L
