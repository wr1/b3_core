import numpy as np
import pytest

from b3_core.core.scoring import ScoreField, survival

GS30 = {
    "dx": 30,
    "dy": 30,
    "thickness": 20,
    "xgr": [[0, 30, 18.0, 1.0]],
    "ygr": [[0, 30, 18.0, 1.0]],
    "core": {"cell_size": 0.6},
}


# -- survival function ------------------------------------------------------
def test_survival_scalar_is_linear():
    S, reach = survival(0.6)
    assert reach == 0.6
    assert S(0.0) == pytest.approx(1.0)
    assert S(0.3) == pytest.approx(0.5)
    assert S(0.6) == pytest.approx(0.0)
    assert S(1.0) == pytest.approx(0.0)  # clipped


@pytest.mark.parametrize("dist", ["lognormal", "normal"])
def test_survival_distribution_monotone_unit_at_zero(dist):
    S, reach = survival({"mean": 0.4, "std": 0.15, "dist": dist})
    d = np.linspace(0, reach, 50)
    s = S(d)
    assert s[0] == pytest.approx(1.0, abs=1e-6)  # P=1 at the cut
    assert s[-1] < 0.02  # ~0 at reach
    assert np.all(np.diff(s) <= 1e-12)  # monotone decreasing
    # wider spread -> larger reach
    _, reach_wide = survival({"mean": 0.4, "std": 0.30, "dist": dist})
    assert reach_wide > reach


def test_survival_none():
    S, reach = survival(None)
    assert reach == 0.0
    assert np.allclose(S(np.array([0.0, 1.0])), 0.0)


# -- ScoreField geometry ----------------------------------------------------
def test_scorefield_boundary_conditions():
    f = ScoreField(GS30)
    assert f.active and f.reach == 0.6
    pts = np.array(
        [
            [0.2, 15, 5],  # inside the kerf -> 1
            [0.8, 15, 5],  # 0.3 mm outside the x=0.5 wall -> 0.5
            [15, 15, 10],  # deep foam -> 0
            [0.2, 15, 18.3],  # 0.3 mm below the groove root (z=18) -> 0.5
            [15, 15, 19.5],  # under the un-sawn bottom face, far from grooves -> 0
        ]
    )
    p = f.resin_probability(pts)
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.5, abs=0.05)
    assert p[2] == pytest.approx(0.0)
    assert p[3] > 0.3  # root halo present
    assert p[4] == pytest.approx(0.0)  # bottom face is not a cut surface


def test_scorefield_inactive_without_cell_size():
    f = ScoreField({**GS30, "core": {}})
    assert not f.active
    assert np.allclose(f.resin_probability(np.zeros((3, 3))), 0.0)


def test_face_halo_thinner_than_saw_cut():
    f = ScoreField(GS30)
    near_face = np.array([[15.0, 15.0, 19.9]])  # 0.1 mm below top face (z=20)
    near_saw = np.array([[0.6, 15.0, 5.0]])  # 0.1 mm outside x-groove wall
    p_face = f.resin_probability(near_face)[0]
    p_saw = f.resin_probability(near_saw)[0]
    assert p_face > 0.1
    assert p_saw > p_face
    assert f.surfaces["face"]["reach"] == pytest.approx(0.15, abs=0.01)


def test_face_halo_disabled():
    inp = {
        **GS30,
        "scoring": {"surfaces": {"face": {"enabled": False}}},
    }
    f = ScoreField(inp)
    near_face = np.array([[15.0, 15.0, 19.9]])
    assert f.resin_probability(near_face)[0] == pytest.approx(0.0)


def test_halo_distance_wraps_and_is_an_offset_phase():
    """A pitch image across x=0 matches, and an offset is only a phase shift."""
    base = {
        "dx": 30.0,
        "dy": 30.0,
        "thickness": 20.0,
        "xgr": [[0.0, 10.0, 18.0, 1.0]],
        "ygr": [],
        "core": {"cell_size": 0.6},
    }
    field = ScoreField(base)
    # Centre at 0, half-width 0.5. x=0.8 and x=9.2 are the same 0.3 mm stand-off
    # (right wall, and the left wall's image at pitch - 0.5).
    near = np.array([[0.8, 15.0, 5.0], [9.2, 15.0, 5.0]])
    d = field.distance_to_saw_cut(near)
    assert d[0] == pytest.approx(0.3, abs=1e-9)
    assert d[1] == pytest.approx(d[0])

    shift = 3.0
    moved = {**base, "xgr": [[shift, 10.0, 18.0, 1.0]]}
    xs = np.linspace(0.0, 30.0, 60, endpoint=False)
    pts = np.column_stack([xs, np.full_like(xs, 15.0), np.full_like(xs, 5.0)])
    rolled = pts.copy()
    rolled[:, 0] = np.mod(xs - shift, 30.0)
    d_moved = ScoreField(moved).distance_to_saw_cut(pts)
    d_phase = field.distance_to_saw_cut(rolled)
    np.testing.assert_allclose(d_moved, d_phase, atol=1e-9)
    assert field.resin_probability(pts).mean() == pytest.approx(
        ScoreField(moved).resin_probability(pts).mean(), abs=1e-9
    )


def test_saw_cut_explicit_override():
    inp = {
        **GS30,
        "scoring": {
            "surfaces": {"saw_cut": {"cell_size": 0.3}, "face": {"enabled": False}}
        },
    }
    f = ScoreField(inp)
    assert f.surfaces["saw_cut"]["reach"] == pytest.approx(0.3)
    near_saw = np.array([[0.8, 15.0, 5.0]])  # 0.3 mm outside wall -> P ~ 0
    assert f.resin_probability(near_saw)[0] == pytest.approx(0.0, abs=0.05)


# -- shared solver sampling -------------------------------------------------
def test_backends_use_one_shared_sampler():
    """The numpy and MFEM backends must not fork the constitutive map."""
    from b3_core.solvers import mfem, sampling
    from b3_core.solvers.numpy_fe import backend as numpy_backend

    for name in ("per_gp_stiffness", "phase_attributes"):
        shared = getattr(sampling, name)
        assert getattr(numpy_backend, name) is shared
        assert getattr(mfem, name) is shared


def test_numpy_halo_adapter_delegates_to_shared_sampler():
    from b3_core.solvers.numpy_fe.assembly import SHAPE_N
    from b3_core.solvers.numpy_fe.halo_sampling import gauss_point_resin_P
    from b3_core.solvers.sampling import cell_aabb_mm, halo_probability

    field = ScoreField(GS30)
    verts_mm = np.array(
        [
            [0, 0, 0],
            [5, 0, 0],
            [5, 5, 0],
            [0, 5, 0],
            [0, 0, 5],
            [5, 0, 5],
            [5, 5, 5],
            [0, 5, 5],
        ],
        dtype=float,
    )
    points_m = verts_mm / 1000.0
    cells = np.arange(8, dtype=np.int64)[None, :]

    for strategy in ("exact", "local_cloud"):
        got = gauss_point_resin_P(
            points_m, cells, field, strategy=strategy, resolution=2, idw_power=2.0
        )
        gp_mm = np.einsum("gn,enj->egj", SHAPE_N, points_m[cells]) * 1000.0
        lo_mm, hi_mm = cell_aabb_mm(points_m, cells)
        want = halo_probability(
            gp_mm,
            score_field=field,
            strategy=strategy,
            resolution=2,
            idw_power=2.0,
            lo_mm=lo_mm,
            hi_mm=hi_mm,
        )
        np.testing.assert_allclose(got, want)
