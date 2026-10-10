"""Fit primitives on a stand-in solve. No finite-element run."""

from types import SimpleNamespace

import pytest

from b3_core.cases import uniaxial
from b3_core.fit.identify import sensitivity
from b3_core.fit.materials import estimate_foam, resin_typical, split_basis
from b3_core.fit.params import apply, default_bounds
from b3_core.fit.refine import refine
from b3_core.fit.residuals import residuals
from b3_core.fit.spec import Prior, Target, TargetSet
from b3_core.fit.volume import _rho, estimate_halo
from b3_core.hashing import canonical_json


def test_source_and_reference_stay_out_of_the_cache_json():
    case = uniaxial()
    labelled = case.input.model_copy(
        update={
            "core": case.input.core.model_copy(
                update={"source": "calibrated", "reference": "coupon"}
            )
        }
    )
    assert "calibrated" not in canonical_json(labelled)
    assert "coupon" not in canonical_json(labelled)
    assert canonical_json(labelled) == canonical_json(case.input)


def test_split_basis_and_foam_estimate():
    materials, targets = split_basis(
        [
            {"quantity": "Ex", "value": 80e6, "basis": "neat", "source": "core"},
            {"quantity": "rho_infused", "value": 100.0, "basis": "neat"},
            {"quantity": "Ez", "value": 90e6, "basis": "infused", "rel_sd": 0.05},
        ]
    )
    assert materials["core"].E == pytest.approx(80e6)
    assert materials["core"].source == "neat"
    assert [row.quantity for row in targets.targets] == ["Ez"]
    foam = estimate_foam(100.0, family="PVC")
    assert foam.source == "estimated"
    assert foam.E > 0
    assert "±20%" in foam.reference
    assert resin_typical().source == "neat"


def test_residuals_remap_a_test_axis_and_skip_a_missing_value():
    report = residuals(
        {"Ey": 110.0},
        TargetSet(
            targets=[
                Target(quantity="Ex", value=100.0, sd=10.0, axis_map={"x": "y"}),
                Target(quantity="Ez", value=None),
            ]
        ),
    )
    used = [row for row in report.rows if row["used"]]
    assert used[0]["model_quantity"] == "Ey"
    assert used[0]["z"] == pytest.approx(1.0)
    assert report.n_used == 1


def test_refine_matches_a_scale_on_a_stand_in_solve():
    case = uniaxial()
    base = float(case.input.core.E)
    params = default_bounds(case, rel=0.5)
    params.params = [
        param.model_copy(
            update={
                "fixed": param.name != "foam_E_scale",
                "prior": Prior(kind="uniform"),
            }
        )
        if param.name == "foam_E_scale"
        else param.model_copy(update={"fixed": True})
        for param in params.params
    ]
    targets = TargetSet(targets=[Target(quantity="Ex", value=1.2 * base, rel_sd=0.01)])

    def solve(point, *, backend, cache):
        del backend, cache
        return SimpleNamespace(
            result=SimpleNamespace(properties={"Ex": float(point.core.E)}),
            geometry={},
        )

    result = refine(case, params, targets, backend="numpy", solve=solve, max_solves=40)
    assert result.status == "converged"
    found = next(param for param in result.params if param.name == "foam_E_scale")
    assert found.value == pytest.approx(1.2, rel=1e-3)
    assert result.calibrated_case["core"]["source"] == "calibrated"
    assert result.provenance["backend"] == "numpy"
    assert result.solves["true"] >= 1


def test_sensitivity_flags_a_parameter_the_stand_in_ignores():
    case = uniaxial()
    params = default_bounds(case, rel=0.5)
    params.params = [
        param.model_copy(
            update={"fixed": param.name not in {"foam_E_scale", "kerf_width"}}
        )
        for param in params.params
    ]
    targets = TargetSet(
        targets=[Target(quantity="Ex", value=float(case.input.core.E), rel_sd=0.05)]
    )

    def solve(point, *, backend, cache):
        del backend, cache
        return SimpleNamespace(
            result=SimpleNamespace(properties={"Ex": float(point.core.E)}),
            geometry={},
        )

    report = sensitivity(case, params, targets, solve=solve, step=0.05)
    assert "kerf_width" in report.weak_params
    assert "foam_E_scale" not in report.weak_params
    assert report.identifiable is False


def test_estimate_halo_brackets_density_without_a_solve():
    case = uniaxial()
    low = _rho(case, 0.0)
    high = _rho(case, 1.5)
    if high == pytest.approx(low):
        estimate = estimate_halo(case, rho_infused=low + 50.0)
        assert estimate.feasible is False
        return
    target = 0.5 * (low + high)
    estimate = estimate_halo(case, rho_infused=target, sd=0.05)
    assert estimate.feasible is True
    assert 0.0 < estimate.value < 1.5
    assert estimate.lower <= estimate.value <= estimate.upper


def test_apply_scales_the_isotropic_modulus_and_leaves_a_missing_field():
    case = uniaxial().input
    params = default_bounds(case)
    params.params = [
        param.model_copy(update={"fixed": False, "value": 1.1})
        if param.name == "foam_E_scale"
        else param
        for param in params.params
    ]
    updated = apply(case, params, [1.1])
    assert updated.core.E == pytest.approx(case.core.E * 1.1)
    assert updated.core.Ez is None
