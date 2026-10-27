import shutil

import numpy as np
import pytest

from b3_core import datasheet
from b3_core.io import mfem_backend

CASE = "examples/mfem_patterns/two_sided.json"
_HAVE_TYPST = shutil.which("typst") is not None

pytestmark = pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Generate the datasheet once (MFEM solve is slow) and share it."""
    d = tmp_path_factory.mktemp("datasheet")
    out_pdf = d / "ds.pdf" if _HAVE_TYPST else None
    out_png = d / "ds.png" if _HAVE_TYPST else None
    spec = datasheet.generate(
        CASE, out_pdf, out_png=out_png, workdir=d, skip_compile=not _HAVE_TYPST
    )
    return spec, d


def test_spec_tables_and_panels(built):
    spec, _ = built
    assert spec.rve_rows and spec.material_rows and spec.analysis_rows
    assert spec.mesh_n_cells and spec.mesh_n_cells > 0
    assert spec.figure_cuts.is_file() and spec.figure_cuts.stat().st_size > 1000
    assert spec.figure_iso.is_file() and spec.figure_iso.stat().st_size > 1000


def test_stiffness_is_physical(built):
    spec, _ = built
    c = spec.c_eff_gpa
    assert c.shape == (6, 6)
    assert np.allclose(c, c.T, atol=1e-3)  # symmetric
    assert np.all(np.linalg.eigvalsh(c) > 0.0)  # positive definite
    ec = spec.engineering_constants
    assert all(ec[k] > 0 for k in ("E_x", "E_y", "E_z", "G_xy", "G_xz", "G_yz"))


def test_build_typst_contains_tables_and_matrix(built):
    spec, _ = built
    src = datasheet.build_typst(spec)
    for token in (
        "RVE / Geometry",
        "Materials",
        "Analysis",
        "Engineering constants",
        'C_"eff"',
    ):
        assert token in src


@pytest.mark.skipif(not _HAVE_TYPST, reason="typst binary not on PATH")
def test_compiles_pdf_and_png(built):
    _, d = built
    assert (d / "ds.pdf").stat().st_size > 10_000
    assert (d / "ds.png").stat().st_size > 10_000
