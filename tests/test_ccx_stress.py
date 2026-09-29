"""CalculiX stress assembly, without a ccx binary."""

import numpy as np

from b3_core.solvers.calculix.stress import (
    ccx_stress_to_voigt,
    stiffness_from_responses,
    volume_average_from_dat,
)


def test_one_element_dat_maps_to_voigt():
    text = "1 1 1.0 2.0 3.0 4.0 5.0 6.0\n"
    mean = volume_average_from_dat(text, np.array([2.0]))
    # Columns are SXX SYY SZZ SXY SXZ SYZ → Voigt xx yy zz yz xz xy.
    assert mean.tolist() == [1.0, 2.0, 3.0, 6.0, 5.0, 4.0]
    assert ccx_stress_to_voigt([1, 2, 3, 4, 5, 6]).tolist() == mean.tolist()


def test_identity_strains_recover_stiffness():
    stiffness = np.arange(36, dtype=float).reshape(6, 6)
    stiffness = 0.5 * (stiffness + stiffness.T)
    strains = [np.eye(6)[:, column] for column in range(6)]
    stresses = [stiffness[:, column] for column in range(6)]
    recovered = stiffness_from_responses(strains, stresses)
    assert np.allclose(recovered, stiffness)
