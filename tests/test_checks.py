"""Diagnostics and coordinate ties. Synthetic matrices, plus one small numpy solve."""

import json

import numpy as np
import pytest
import pyvista as pv
from tests.fakes import fake_backend, unregister

from b3_core.cache import MemoryCache
from b3_core.cases import plain, uniaxial
from b3_core.core.mesh import MIN_HW, create_grooved_mesh
from b3_core.core.run import cmd_report_ties, cmd_run
from b3_core.models import Curvature
from b3_core.solvers.checks import (
    _MACRO,
    build_diagnostics,
    kerf_pinch,
    periodic_mismatch,
    positive_definite,
    raw_asymmetry,
    strict_failures,
    tie_report,
    unmatched_nodes,
    voigt_reuss,
)
from b3_core.solvers.elasticity import isotropic_C
from b3_core.solvers.numpy_fe.assembly import homogenize_aniso
from b3_core.solvers.numpy_fe.backend import runnumpy
from b3_core.solvers.protocol import as_solve_result

CORE = {"E": 80e6, "nu": 0.3, "rho": 100.0}
RESIN = {"E": 3e9, "nu": 0.3, "rho": 1200.0}


def _asymmetric(ratio: float) -> np.ndarray:
    matrix = np.eye(6)
    matrix[0, 1] = ratio * np.sqrt(6.0) / np.sqrt(2.0)
    return matrix


def _mesh(kx: float, ky: float = 0.0):
    return create_grooved_mesh(
        thickness=10.0,
        dx=20.0,
        dy=20.0,
        xcuts=[[0.0, 20.0, 6.0, 1.0]],
        ycuts=[[0.0, 20.0, 6.0, 1.0]] if ky else [],
        madd=(0.0,),
        tface=0.0,
        kx=kx,
        ky=ky,
    )


def _face(name: str, ties: list[dict]) -> dict:
    return next(row for row in ties if row["face"] == name)


def test_raw_asymmetry_thresholds():
    assert raw_asymmetry(np.zeros((6, 6)))["value"] == 0.0
    assert raw_asymmetry(np.eye(6))["status"] == "ok"
    assert raw_asymmetry(_asymmetric(1e-8))["status"] == "ok"
    warned = raw_asymmetry(_asymmetric(1e-4))
    assert warned["status"] == "warn"
    assert warned["value"] == pytest.approx(1e-4, rel=1e-6)
    failed = raw_asymmetry(_asymmetric(1e-2))
    assert failed["status"] == "fail"


def test_positive_definite_uses_the_symmetrised_matrix():
    assert positive_definite(np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))["status"] == "ok"
    negative = np.eye(6)
    negative[0, 0] = -1.0
    row = positive_definite(negative)
    assert row["status"] == "fail"
    assert row["value"] < 0.0


def test_voigt_reuss_accepts_a_mixture_and_rejects_an_overshoot():
    phi = 0.2
    core_c = isotropic_C(CORE["E"], CORE["nu"])
    resin_c = isotropic_C(RESIN["E"], RESIN["nu"])
    voigt = (1.0 - phi) * core_c + phi * resin_c
    reuss = np.linalg.inv(
        (1.0 - phi) * np.linalg.inv(core_c) + phi * np.linalg.inv(resin_c)
    )
    mid = 0.5 * (voigt + reuss)
    assert voigt_reuss(mid, CORE, RESIN, phi)["status"] == "ok"
    assert voigt_reuss(voigt, CORE, RESIN, phi)["status"] == "ok"
    over = voigt_reuss(2.0 * voigt, CORE, RESIN, phi)
    assert over["status"] == "fail"
    assert over["value"] > 0.02
    skipped = voigt_reuss(
        voigt, CORE, RESIN, phi, face={"E": 12e9, "nu": 0.3, "thickness": 1.0}
    )
    assert skipped["status"] == "ok"
    assert skipped["value"] is None
    assert voigt_reuss(voigt, CORE, RESIN, 1.5)["status"] == "fail"
    scale = max(float(np.linalg.norm(voigt)), 1.0)
    nudged = voigt_reuss(voigt + 1e-2 * scale * np.eye(6), CORE, RESIN, phi)
    assert nudged["status"] == "warn"


def test_unmatched_nodes_counts_the_gap():
    assert unmatched_nodes([{"top_nodes": 4, "slaves": 4}])["status"] == "ok"
    row = unmatched_nodes(
        [{"top_nodes": 4, "slaves": 4}, {"top_nodes": 5, "slaves": 3}]
    )
    assert row["status"] == "fail"
    assert row["value"] == 2


def test_kerf_pinch_warns_when_the_mouth_is_floored_and_fails_when_it_crosses():
    flat = uniaxial().input
    opened = kerf_pinch(flat)
    assert opened["status"] == "ok"
    assert opened["value"] == pytest.approx(1.5)
    assert kerf_pinch(plain().input)["value"] is None
    warn = flat.model_copy(update={"curvature": Curvature(kx=0.0374875, ky=0.0)})
    warned = kerf_pinch(warn)
    assert warned["status"] == "warn"
    assert 0.0 < warned["value"] < MIN_HW
    crossed = flat.model_copy(update={"curvature": Curvature(kx=0.04, ky=0.0)})
    assert kerf_pinch(crossed)["status"] == "fail"


def test_tie_invariants_at_the_three_curvature_points():
    flat = tie_report(np.asarray(_mesh(0.0).points))
    assert _face("z", flat)["identity"] is True
    assert _face("z", flat)["multi_node_rows"] == 0
    for row in flat:
        assert row["slaves"] == row["top_nodes"]
        assert row["ok"] is True

    slight = tie_report(np.asarray(_mesh(5e-5).points))
    assert _face("z", slight)["identity"] is False
    assert _face("z", slight)["multi_node_rows"] > 0
    assert _face("z", slight)["slaves"] == _face("z", slight)["top_nodes"]
    assert _face("x", slight)["identity"] is True
    assert _face("y", slight)["identity"] is True

    wide = tie_report(np.asarray(_mesh(2e-3, 2e-3).points))
    assert _face("z", wide)["multi_node_rows"] > 0
    assert _face("z", wide)["slaves"] == _face("z", wide)["top_nodes"]
    assert _face("z", wide)["ok"] is True
    assert _face("x", wide)["identity"] is True
    assert _face("y", wide)["identity"] is True


def test_periodic_mismatch_is_zero_for_an_affine_field_and_fails_on_a_jump():
    axis = np.linspace(0.0, 1.0, 3)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    zero = periodic_mismatch(grid, None)
    assert zero["status"] == "ok"
    assert zero["value"] is None
    fields = {name: grid @ _MACRO[name] for name in _MACRO}
    clean = periodic_mismatch(grid, fields)
    assert clean["status"] == "ok"
    assert clean["value"] == pytest.approx(0.0, abs=1e-12)
    jumped = {name: value.copy() for name, value in fields.items()}
    jumped["zz"][grid[:, 2] > 0.9] += 1e-3
    failed = periodic_mismatch(grid, jumped)
    assert failed["status"] == "fail"
    assert failed["value"] == pytest.approx(1e-3, abs=1e-9)
    warned = periodic_mismatch(grid, {"xx": np.zeros((1, 3))})
    assert warned["status"] == "warn"
    assert warned["value"] is None
    mild = {name: value.copy() for name, value in fields.items()}
    mild["zz"][grid[:, 2] > 0.9] += 1e-5
    assert periodic_mismatch(grid, mild)["status"] == "warn"
    assert periodic_mismatch(grid, {"nope": np.zeros_like(grid)})["value"] is None
    broken = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)
    assert _face("z", tie_report(broken))["ok"] is False
    assert periodic_mismatch(broken, {"xx": np.zeros_like(broken)})["status"] == "warn"


def test_numpy_cube_keeps_the_raw_matrix_and_the_published_symmetrisation():
    axis = np.linspace(0.0, 2.0, 3)
    nodes = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = pv.StructuredGrid(*nodes).cast_to_unstructured_grid()
    points = np.asarray(grid.points)
    cells = grid.cells.reshape(-1, 9)[:, 1:]
    stiffness_c = isotropic_C(4e9, 0.3)
    elem_c = np.broadcast_to(stiffness_c, (len(cells), 6, 6))
    published, info = homogenize_aniso(points, cells, elem_c)
    raw = info["raw_stiffness"]
    assert np.allclose(published, 0.5 * (raw + raw.T))
    assert np.allclose(published, stiffness_c, rtol=1e-9, atol=1.0)
    assert raw_asymmetry(raw)["status"] == "ok"


def test_numpy_flat_solve_exposes_raw_and_a_periodic_field():
    result = runnumpy(_mesh(0.0), RESIN, CORE, return_details=True)
    assert np.allclose(
        result.stiffness, 0.5 * (result.raw_stiffness + result.raw_stiffness.T)
    )
    adapted = as_solve_result(result, details=True)
    assert adapted.raw_stiffness is not None
    assert np.allclose(adapted.raw_stiffness, result.raw_stiffness)
    row = periodic_mismatch(result.points, result.displacements)
    assert row["status"] == "ok"
    assert row["value"] < 1e-6
    assert positive_definite(result.stiffness)["status"] == "ok"


def test_split_keeps_an_asymmetric_calculix_response():
    from b3_core.solvers.calculix.stress import (
        split_stiffness_from_responses,
        stiffness_from_responses,
    )

    stress = np.arange(36, dtype=float).reshape(6, 6)
    strains = [np.eye(6)[:, column] for column in range(6)]
    stresses = [stress[:, column] for column in range(6)]
    published, raw = split_stiffness_from_responses(strains, stresses)
    assert np.allclose(raw, stress)
    assert np.allclose(published, 0.5 * (stress + stress.T))
    assert np.allclose(stiffness_from_responses(strains, stresses), published)
    assert raw_asymmetry(raw)["status"] == "fail"
    assert raw_asymmetry(published)["status"] == "ok"


def test_build_diagnostics_uses_backend_ties_when_the_solver_returns_them():
    from types import SimpleNamespace

    case = plain().input
    prep = SimpleNamespace(
        mesh=_mesh(0.0),
        geometry=SimpleNamespace(effective_resin_vf=0.0),
    )
    solved = SimpleNamespace(
        stiffness=np.eye(6) * 1e6,
        raw_stiffness=None,
        ties=[
            {
                "face": "z",
                "top_nodes": 2,
                "slaves": 2,
                "identity": True,
                "multi_node_rows": 0,
                "ok": True,
            }
        ],
        points=None,
        displacements=None,
    )
    block = build_diagnostics(case, prep, solved)
    assert block["ties"] == solved.ties
    assert block["checks"][0]["id"] == "raw_asymmetry"
    assert block["checks"][0]["status"] == "ok"


def test_run_case_stores_diagnostics_and_a_hit_returns_them(tmp_path):
    register, cls = fake_backend("fake")
    register(cls)
    cache = MemoryCache()
    try:
        first = build_case_record(cache)
        second = build_case_record(cache)
    finally:
        unregister("fake")
    assert second.diagnostics == first.diagnostics
    assert strict_failures(first.diagnostics)
    assert {row["face"] for row in first.diagnostics["ties"]} == {"x", "y", "z"}
    del tmp_path


def build_case_record(cache):
    from b3_core.api import run_case

    return run_case(plain(), backend="fake", cache=cache)


def test_cmd_run_strict_exits_after_printing_json(tmp_path, capsys):
    register, cls = fake_backend("fake")
    register(cls)
    case = tmp_path / "case.json"
    try:
        plain(backend="fake").to_json(case)
        with pytest.raises(SystemExit) as caught:
            cmd_run(str(case), backend="fake", as_json=True, no_write=True, strict=True)
    finally:
        unregister("fake")
    assert caught.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(row["status"] == "fail" for row in payload["diagnostics"]["checks"])


def test_cmd_report_ties_json_uses_the_passed_curvature(tmp_path, capsys):
    case = tmp_path / "case.json"
    uniaxial().to_json(case)
    cmd_report_ties(str(case), kx="5e-5", as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["kx"] == pytest.approx(5e-5)
    z_row = _face("z", payload["ties"])
    assert z_row["slaves"] == z_row["top_nodes"]
    assert z_row["multi_node_rows"] > 0
    assert z_row["identity"] is False
