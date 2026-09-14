import json

import numpy as np
import pytest
import yaml

from b3_core.core.analysis import geom_analysis
from b3_core.core.cprop import CpropInput, load_case
from b3_core.core.mesh import create_grooved_mesh, create_grooves


def test_create_grooves():
    cuts = [[10, 5, 2, 1]]
    bnd = 20
    bot, top, height, slope = create_grooves(cuts, bnd, meshadd=[0])
    assert len(bot) == len(top) == len(height) == len(slope)
    assert np.all(bot >= 0)
    assert np.all(top <= bnd)
    assert np.any(height != 0)
    assert np.all(slope == 0)  # no curvature -> no taper


def test_create_grooved_mesh():
    mesh = create_grooved_mesh(
        thickness=30.0,
        dx=50.0,
        dy=50.0,
        xcuts=[[10, 5, 2, 1]],
        ycuts=[[10, 5, 2, 1]],
        madd=[0],
        tface=2.0,
    )
    assert "resin" in mesh.cell_data
    assert "face" in mesh.cell_data
    assert mesh.n_cells > 0
    assert np.any(mesh.cell_data["resin"])
    assert np.any(mesh.cell_data["face"])


def test_geom_analysis():
    mesh = create_grooved_mesh(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[], ycuts=[], madd=[0], tface=2.0
    )
    result = geom_analysis(mesh)
    assert "area_increase" in result
    assert "resin_vf" in result
    assert 0 <= result["resin_vf"] <= 1
    assert result["area_increase"] >= 1


def test_load_case_json_and_yaml(tmp_path):
    data = {
        "dx": 50.0,
        "dy": 50.0,
        "thickness": 30.0,
        "xgr": [],
        "ygr": [],
        "core": {"E": 4e9, "nu": 0.3, "rho": 100.0},
        "resin": {"E": 4e9, "nu": 0.3, "rho": 1100.0},
    }
    json_path = tmp_path / "case.json"
    yaml_path = tmp_path / "case.yaml"
    json_path.write_text(json.dumps(data))
    yaml_path.write_text(yaml.dump(data))

    assert load_case(str(json_path))[0] == data
    assert load_case(str(yaml_path))[0] == data


def test_load_case_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "case.toml"
    path.write_text("dx = 50")
    with pytest.raises(ValueError, match="unsupported case file type"):
        load_case(str(path))


def test_cprop_input_accepts_fenicsx_backend():
    data = {
        "dx": 50.0,
        "dy": 50.0,
        "thickness": 30.0,
        "xgr": [],
        "ygr": [],
        "core": {"E": 4e9, "nu": 0.3, "rho": 100.0},
        "resin": {"E": 4e9, "nu": 0.3, "rho": 1100.0},
        "backend": "fenicsx",
        "validate_with_ccx": True,
    }

    cfg = CpropInput(**data)

    assert cfg.backend == "fenicsx"
    assert cfg.validate_with_ccx is True


def test_cprop_input_accepts_mfem_backend():
    data = {
        "dx": 50.0,
        "dy": 50.0,
        "thickness": 30.0,
        "xgr": [],
        "ygr": [],
        "core": {"E": 4e9, "nu": 0.3, "rho": 100.0},
        "resin": {"E": 4e9, "nu": 0.3, "rho": 1100.0},
        "backend": "mfem",
        "validate_with_ccx": True,
    }

    cfg = CpropInput(**data)

    assert cfg.backend == "mfem"
    assert cfg.validate_with_ccx is True


def test_cprop_input_rejects_unknown_backend():
    data = {
        "dx": 50.0,
        "dy": 50.0,
        "thickness": 30.0,
        "xgr": [],
        "ygr": [],
        "core": {"E": 4e9, "nu": 0.3, "rho": 100.0},
        "resin": {"E": 4e9, "nu": 0.3, "rho": 1100.0},
        "backend": "tensormesh",
    }

    with pytest.raises(ValueError):
        CpropInput(**data)
