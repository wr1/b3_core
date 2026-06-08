import json

import numpy as np
import pyvista as pv
import pytest

from b3_core.core.cprop import Material
from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import aniso, mfem_backend

needs_mfem = pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)


def _mesh(path):
    i = json.loads(open(path).read())
    m = create_grooved_mesh(
        thickness=i["thickness"], dx=i["dx"], dy=i["dy"],
        xcuts=i["xgr"], ycuts=i["ygr"], madd=tuple(i.get("madd", [0])),
        tface=(i.get("face") or {}).get("thickness", 0.0),
    )
    return m, i


# -- correctness gates ------------------------------------------------------
def test_uniform_cube_recovers_input_stiffness():
    x = np.linspace(0, 2, 3)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    g = pv.StructuredGrid(X, Y, Z).cast_to_unstructured_grid()
    pts = np.asarray(g.points)
    cells = g.cells.reshape(-1, 9)[:, 1:]
    C0 = aniso.isotropic_C(4e9, 0.3)
    elem_C = np.broadcast_to(C0, (len(cells), 6, 6))
    K, _ = aniso.homogenize_aniso(pts, cells, elem_C)
    assert np.allclose(K, C0, rtol=1e-9, atol=1.0)


@needs_mfem
@pytest.mark.parametrize("case", [
    "examples/with_grooves.json",
    "examples/mfem_patterns/two_sided.json",
    "examples/complex.json",
])
def test_numpy_matches_mfem(case):
    m, i = _mesh(case)
    Cm = np.asarray(
        mfem_backend.runmfem(m, i["resin"], i["core"], i.get("face"),
                             return_details=True).stiffness
    )
    Ca = aniso.runnumpy(m, i["resin"], i["core"], i.get("face")).stiffness
    assert np.abs(Ca - Cm).max() / np.abs(Cm).max() < 1e-5


# -- orthotropic GS30 benchmark (Laustsen et al. 2014, Table 2) -------------
def test_orthotropic_gs30_benchmark():
    m, i = _mesh("examples/diab_gs30.json")
    assert i["core"].get("E1") is not None      # the example uses orthotropic foam
    C = aniso.runnumpy(m, i["resin"], i["core"]).stiffness
    ec = aniso._properties_from_stiffness(C)[0]
    assert ec["Exx"] == pytest.approx(ec["Eyy"], rel=1e-6)   # square grid symmetry
    assert ec["Gxz"] == pytest.approx(ec["Gyz"], rel=1e-6)
    assert ec["Ezz"] > ec["Exx"]                            # stiffer through-thickness
    assert 200e6 < ec["Ezz"] < 320e6                        # paper rule-of-mixtures 262 MPa
    assert 15e6 < ec["Gxy"] < 30e6                          # paper 20 MPa


# -- resin-grid failure check ----------------------------------------------
def test_resin_failure_index_scales_linearly():
    m, i = _mesh("examples/diab_gs30.json")
    det = aniso.runnumpy(m, i["resin"], i["core"], return_details=True)
    base = aniso.resin_failure_index(det, macro_strain=[0.01, 0, 0, 0, 0, 0])
    assert base["n_resin"] > 0
    assert base["failure_index"] > 0
    dbl = aniso.resin_failure_index(det, macro_strain=[0.02, 0, 0, 0, 0, 0])
    assert dbl["failure_index"] == pytest.approx(2 * base["failure_index"], rel=1e-6)
    # a stress-driven query agrees with the equivalent strain
    sig = det.stiffness @ np.array([0.01, 0, 0, 0, 0, 0.0])
    by_stress = aniso.resin_failure_index(det, macro_stress=sig)
    assert by_stress["failure_index"] == pytest.approx(base["failure_index"], rel=1e-6)


# -- material schema --------------------------------------------------------
def test_material_isotropic_or_orthotropic():
    Material(E=1e9, nu=0.3, rho=60)
    ortho = Material(
        E1=32e6, E2=32e6, E3=70e6, G12=19e6, G13=19e6, G23=19e6,
        nu12=0.3, nu13=0.3, nu23=0.3, rho=60,
    )
    assert ortho.is_orthotropic
    with pytest.raises(ValueError):
        Material(rho=60)   # neither complete set
