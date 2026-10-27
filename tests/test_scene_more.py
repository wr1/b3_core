"""Exercise CoreScene paths not covered by the basic viz suite."""

from __future__ import annotations

import pytest

from b3_core.io import mfem_backend
from b3_core.viz import CoreModel, CoreScene

MESH_CASE = "examples/with_grooves.json"
needs_mfem = pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)


def test_scene_geometry_pipeline(tmp_path):
    model = CoreModel.from_json(MESH_CASE)
    scene = CoreScene(model, off_screen=True, window_size=(400, 300))
    scene.add_phases(edges=True).add_mesh_edges().add_cutaway(
        "x", 0.4
    ).add_slice_planes(x=model.mesh.center[0], z=model.mesh.center[2])
    scene.isometric().add_axes()
    out = scene.screenshot(tmp_path / "scene.png")
    assert out.is_file()
    scene.close()
    scene.close()  # second close is a no-op path


def test_scene_show_requires_interactive():
    model = CoreModel.from_json(MESH_CASE)
    scene = CoreScene(model, off_screen=True)
    with pytest.raises(RuntimeError, match="off_screen"):
        scene.show()
    scene.close()


@needs_mfem
def test_scene_modulus_surface(tmp_path):
    model = CoreModel.from_json(MESH_CASE)
    scene = CoreScene(model, off_screen=True, window_size=(320, 240))
    scene.add_modulus_surface()
    scene.screenshot(tmp_path / "mod.png")
    scene.close()
