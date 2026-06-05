import pytest

from b3_core import deformed
from b3_core.io import mfem_backend

CASE = "examples/mfem_patterns/two_sided.json"


@pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)
def test_render_deformed_modes(tmp_path):
    out = tmp_path / "modes.png"
    result = deformed.render_deformed_modes(CASE, out, warp=0.3)
    assert result == out
    assert out.is_file() and out.stat().st_size > 10_000
