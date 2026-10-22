import json

import pytest

from b3_core import deformed
from b3_core.io import mfem_backend

CASE = "examples/mfem_patterns/two_sided.json"


@pytest.mark.skipif(not mfem_backend.is_mfem_available(), reason="MFEM not installed")
def test_render_deformed_modes(tmp_path):
    # Rendering check, not accuracy: run the example on a coarse mesh.
    case = tmp_path / "case.json"
    case.write_text(json.dumps({**json.load(open(CASE)), "mesh_h_max": 4.0}))
    out = tmp_path / "modes.png"
    result = deformed.render_deformed_modes(case, out, warp=0.3)
    assert result == out
    assert out.is_file() and out.stat().st_size > 10_000
