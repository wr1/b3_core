"""Phase-A hardening: MPC flattening, Gauss-point halo density, mesh guards."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from b3_core.core.scoring import hex_gauss_points
from b3_core.loaders import normalize_case
from b3_core.pipeline import prepare
from b3_core.solvers.fenicsx import flatten_rows

CASE = {
    "dx": 30.0,
    "dy": 30.0,
    "thickness": 20.0,
    "xgr": [[15.0, 30.0, -18.0, 2.0]],
    "ygr": [[15.0, 30.0, -18.0, 2.0]],
    "core": {"E": 70e6, "nu": 0.3, "rho": 80.0},
    "resin": {"E": 3.2e9, "nu": 0.35, "rho": 1150.0},
}


def _case(**over):
    data = json.loads(json.dumps(CASE))
    core = over.pop("core", {})
    data["core"].update(core)
    data.update(over)
    case, _ = normalize_case(data)
    return case


def test_flatten_rows_substitutes_chains():
    rows = {1: {2: 0.5, 3: 0.5}, 2: {4: 1.0}, 5: {1: 1.0}}
    flat = flatten_rows(rows)
    assert flat[1] == {4: 0.5, 3: 0.5}
    assert flat[5] == {4: 0.5, 3: 0.5}
    assert all(m not in rows for row in flat.values() for m in row)


def test_flatten_rows_rejects_cycles():
    with pytest.raises(RuntimeError, match="cyclic"):
        flatten_rows({1: {2: 1.0}, 2: {1: 1.0}})


def test_gauss_weights_sum_to_volume():
    mesh = prepare(_case()).mesh
    _pts, w = hex_gauss_points(mesh)
    assert w.sum() == pytest.approx(30.0 * 30.0 * 20.0, rel=1e-10)


@pytest.mark.parametrize("h", [2.0, 1.0])
def test_halo_density_is_mesh_independent(h):
    ref = prepare(_case(core={"cell_size": 0.81}, mesh_h_max=1.5)).geometry
    got = prepare(_case(core={"cell_size": 0.81}, mesh_h_max=h)).geometry
    assert got.halo_vf == pytest.approx(ref.halo_vf, rel=2e-3)


def test_legacy_coarse_grid_warns():
    with pytest.warns(DeprecationWarning, match="legacy coarse grid"):
        prepare(_case(mesh_h_max=None))


def test_default_grid_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        prepare(_case())


@pytest.mark.parametrize("h", [1.5, 2.0, 3.0])
def test_kerf_has_two_elements_across(h):
    mesh = prepare(_case(mesh_h_max=h)).mesh
    xs = np.unique(np.round(np.asarray(mesh.points)[:, 0], 6))
    inside = xs[(xs > 14.0 + 1e-6) & (xs < 16.0 - 1e-6)]
    assert len(inside) >= 1
