import importlib.util

import numpy as np
import pytest

from b3_core.io import mfem_backend
from b3_core.viz import animate

_HAVE_IMAGEIO = importlib.util.find_spec("imageio") is not None

pytestmark = pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)


def test_bend_points_zero_curvature_is_identity():
    pts = np.random.default_rng(0).normal(size=(20, 3))
    assert np.array_equal(animate._bend_points(pts, 0.0), pts)


def test_bend_points_bends_and_preserves_count():
    # a flat strip along x; positive curvature should arch it (z varies)
    x = np.linspace(-25, 25, 40)
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    bent = animate._bend_points(pts, 0.02, axis=0)
    assert bent.shape == pts.shape
    assert np.ptp(bent[:, 2]) > 1.0         # it actually curved out of plane
    assert abs(bent[:, 0]).max() < abs(x).max() + 1e-9  # arc pulls ends inward


@pytest.mark.skipif(not _HAVE_IMAGEIO, reason="imageio ([anim] extra) not installed")
def test_render_explainer_writes_files(tmp_path):
    out = tmp_path / "a.mp4"
    paths = animate.render_explainer(
        "examples/with_grooves.json", out,
        seconds=2, fps=6, size=(240, 240), stations=2,
    )
    assert out.is_file() and out.stat().st_size > 5_000
    gif = out.with_suffix(".gif")
    assert gif in paths and gif.is_file() and gif.stat().st_size > 5_000
