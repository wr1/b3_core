"""CalculiX TYPE=ORTHO card from Voigt stiffness (no FEA)."""

import numpy as np
import pytest

from b3_core.io.ccx_card import ccx_ortho_card, ortho_dijkl


def test_ortho_dijkl_maps_voigt_shear_to_calculix():
    C = np.zeros((6, 6))
    C[0, 0], C[1, 1], C[2, 2] = 11.0, 22.0, 33.0
    C[0, 1] = C[1, 0] = 12.0
    C[0, 2] = C[2, 0] = 13.0
    C[1, 2] = C[2, 1] = 23.0
    C[3, 3] = 44.0  # yz = Gyz
    C[4, 4] = 55.0  # xz = Gxz
    C[5, 5] = 66.0  # xy = Gxy
    d1111, d1122, d2222, d1133, d2233, d3333, d1212, d1313, d2323 = ortho_dijkl(C)
    assert (d1111, d1122, d2222, d1133, d2233, d3333) == (
        11.0,
        12.0,
        22.0,
        13.0,
        23.0,
        33.0,
    )
    assert d1212 == 66.0
    assert d1313 == 55.0
    assert d2323 == 44.0


def test_ccx_ortho_card_layout():
    C = np.eye(6)
    C[0, 0] = 1.5e9
    text = ccx_ortho_card(C, name="core_hom", rho=180.0, temperature=293.0)
    assert text.startswith("*material,name=core_hom\n*elastic,type=ortho\n")
    assert "*density\n180" in text
    with pytest.raises(ValueError, match="6x6"):
        ortho_dijkl(np.eye(3))
