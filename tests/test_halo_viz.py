import matplotlib
import pytest

matplotlib.use("Agg")

import json
from pathlib import Path

from b3_core.viz.halo import (
    effective_modulus_ratio,
    plot_halo_degradation,
    plot_halo_intuitive_board,
    plot_halo_side_cut,
    resin_probability_vs_distance,
)

CASE = json.loads(
    Path(__file__).resolve().parents[1].joinpath("examples/diab_gs30_scored.json").read_text()
)


def test_uniform_survival_linear():
    d, p, reach = resin_probability_vs_distance(0.6)
    assert reach == 0.6
    assert p[0] == 1.0
    assert p[-1] == pytest.approx(0.0, abs=0.05)


def test_effective_modulus_at_cut_and_far():
    d, p, _ = resin_probability_vs_distance(0.6)
    e = effective_modulus_ratio(p, e_foam=70e6, e_resin=3e9)
    assert e[0] == pytest.approx(3e9 / 70e6, rel=1e-6)
    assert e[-1] == pytest.approx(1.0, rel=0.05)


def test_plot_halo_degradation_runs():
    fig, _ = plot_halo_degradation([0.3, 0.6], e_foam=70e6, e_resin=3e9)
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_halo_side_cut_runs():
    fig, ax = plot_halo_side_cut(CASE)
    assert ax.images
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_halo_intuitive_board_runs():
    fig = plot_halo_intuitive_board(CASE)
    assert len(fig.axes) >= 3
    import matplotlib.pyplot as plt
    plt.close(fig)