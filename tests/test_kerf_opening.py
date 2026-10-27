"""Curvature on the public API is a kerf taper, reported as mouth vs root width."""

import pytest

from b3_core.cases import curved_panel, uniaxial


def test_zero_curvature_leaves_mouth_equal_to_root():
    row = uniaxial().input.kerf_openings()[0]
    assert row["mouth"] == "bottom"
    assert row["hw_mouth_mm"] == pytest.approx(row["hw_root_mm"])


def test_positive_kx_opens_top_mouth_and_pinches_bottom_mouth():
    top = curved_panel(
        kx=0.008, thickness=20, ligament=2, pitch=10, width=2
    ).input.kerf_openings()[0]
    assert top["mouth"] == "top"
    assert top["hw_mouth_mm"] > top["hw_root_mm"]

    bottom = uniaxial(pitch=10, depth=8, width=2).with_curvature(kx=0.008)
    row = bottom.input.kerf_openings()[0]
    assert row["mouth"] == "bottom"
    assert row["hw_mouth_mm"] < row["hw_root_mm"]


def test_negative_kx_opens_a_bottom_mouth_groove():
    row = uniaxial(pitch=10, depth=8, width=2).with_curvature(kx=-0.008)
    opening = row.input.kerf_openings()[0]
    assert opening["hw_mouth_mm"] > opening["hw_root_mm"]
