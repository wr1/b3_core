"""CoreModel and homogenize share one pipeline."""

import numpy as np
import pytest
from tests.fakes import CANONICAL

from b3_core.api import homogenize
from b3_core.cache import MemoryCache
from b3_core.cases import plain
from b3_core.loaders import normalize_case
from b3_core.physics_surrogate import build_training_frame
from b3_core.result import ResultBlock, RunRecord
from b3_core.solvers import available_backends
from b3_core.viz.model import CoreModel


def _ready() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


@pytest.mark.parametrize("name", _ready())
def test_shared_cache_is_the_same_solve(name):
    cache = MemoryCache()
    case = plain()
    model = CoreModel(case, backend=name, cache=cache)
    result = homogenize(case, backend=name, cache=cache)
    np.testing.assert_allclose(model.stiffness, result.stiffness, rtol=0, atol=0)


@pytest.mark.parametrize("name", _ready())
def test_separate_solves_agree(name):
    case = plain()
    model = CoreModel(case, backend=name)
    result = homogenize(case, backend=name)
    np.testing.assert_allclose(model.stiffness, result.stiffness, rtol=1e-10, atol=0)


def test_coremodel_from_path_reads_yaml():
    model = CoreModel.from_path("examples/simple.yaml")
    assert model.case.dx > 0
    assert model.case.thickness > 0


def test_training_frame_keeps_ky(monkeypatch):
    def fake_run(case, **kwargs):
        case_in, _workdir = normalize_case(case)
        return RunRecord(
            case_hash="abc123abc123",
            b3_core_version="0.3.0",
            input=case_in.model_dump(mode="json"),
            geometry={
                "resin_vf": 0.1,
                "area_increase": 1.2,
                "halo_vf": 0.01,
                "effective_resin_vf": 0.11,
                "rho_infused": 180.0,
            },
            result=ResultBlock(
                backend="numpy",
                stiffness=(np.eye(6) * 1.0e8).tolist(),
                properties={
                    key: CANONICAL[key]
                    for key in (
                        "Ex",
                        "Ey",
                        "Ez",
                        "Gxy",
                        "Gxz",
                        "Gyz",
                        "nuxy",
                        "nuxz",
                        "nuyz",
                    )
                },
            ),
        )

    monkeypatch.setattr("b3_core.sweep.curvature_grid.run_case", fake_run)
    frame = build_training_frame(
        kx_values=[0.0, 0.01],
        ky_values=[0.005],
        cell_sizes=[0.6],
        cache=MemoryCache(),
    )
    assert len(frame) == 2
    assert {row["ky"] for row in frame.to_dict("records")} == {0.005}
    assert {row["curvature"]["ky"] for row in frame.to_dict("records")} == {0.005}
