"""Signed radius and curvature unit parsing."""

import pytest

from b3_core.cases import curved_panel, uniaxial
from b3_core.units import (
    axis_to_per_mm,
    curvature_tick,
    points_to_per_mm,
    radius_m,
    to_per_mm,
)


def test_units_store_per_mm_and_keep_the_sign():
    assert to_per_mm(0.002, "1/mm") == pytest.approx(0.002)
    assert to_per_mm(2.0, "1/m") == pytest.approx(0.002)
    assert to_per_mm(500.0, "R-mm") == pytest.approx(0.002)
    assert to_per_mm(0.5, "R-m") == pytest.approx(0.002)
    assert to_per_mm(-0.5, "R-m") == pytest.approx(-0.002)
    assert to_per_mm(0.0, "R-m") == 0.0
    with pytest.raises(ValueError, match="1/mm"):
        to_per_mm(1.0, "degrees")


def test_signed_radius_and_flat_tick():
    assert radius_m(0.0) is None
    assert radius_m(0.002) == pytest.approx(0.5)
    assert radius_m(-0.002) == pytest.approx(-0.5)
    assert curvature_tick(0.0) == "0 (flat)"
    assert "R = 0.5 m" in curvature_tick(0.002)
    assert "R = -0.5 m" in curvature_tick(-0.002)


def test_axis_and_points_convert_before_sampling_counts():
    assert axis_to_per_mm("2:4:2", "1/m") == pytest.approx([0.002, 0.004])
    assert axis_to_per_mm("500:1000:2", "R-mm") == pytest.approx([0.002, 0.001])
    first = points_to_per_mm("2,0 -0.5,0", "1/m")[0]
    assert first[0] == pytest.approx(0.002)
    assert first[1] == 0.0
    radius_point = points_to_per_mm("0.5,0", "R-m")[0]
    assert radius_point[0] == pytest.approx(0.002)
    assert radius_point[1] == 0.0


def test_sign_schematic_titles_follow_the_mouth():
    import matplotlib.pyplot as plt

    from b3_core.viz.halo.curvature_figs import plot_sign_schematic

    fig = plot_sign_schematic(kx=0.004)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("bottom" in title and "k<0" in title for title in titles)
    assert any("top" in title and "k>0" in title for title in titles)
    plt.close(fig)


def test_opens_for_follows_the_mouth():
    bottom = uniaxial().input.kerf_openings()[0]
    assert bottom["mouth"] == "bottom"
    assert bottom["opens_for"] == "k<0"
    top = curved_panel(kx=0.0).input.kerf_openings()[0]
    assert top["mouth"] == "top"
    assert top["opens_for"] == "k>0"
