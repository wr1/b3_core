"""Fast paths (vectorized kernel, PARDISO, block solve) match the references."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
import scipy.sparse as sp

from b3_core.solvers import linsolve
from b3_core.solvers.numpy_fe import assembly


def _mesh(nx=3, ny=2, nz=2, taper=0.0):
    xs, ys, zs = (
        np.linspace(0, 1, nx + 1),
        np.linspace(0, 0.8, ny + 1),
        np.linspace(0, 0.5, nz + 1),
    )
    idx = {}
    pts = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                idx[i, j, k] = len(pts)
                pts.append([x + taper * x * (1 - x) * z, y, z])
    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k)]
                c += [(a, b, k + 1) for a, b, _ in c]
                cells.append([idx[t] for t in c])
    return np.array(pts), np.array(cells)


def _gp_C(n_elem, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_elem, 8, 6, 6))
    return np.einsum("egij,egkj->egik", A, A) + 6 * np.eye(6)


@pytest.mark.parametrize("taper", [0.0, 0.2])
def test_vectorized_kernel_matches_reference(monkeypatch, taper):
    pts, cells = _mesh(taper=taper)
    C = _gp_C(len(cells))
    ref = assembly._element_terms_reference(pts, cells, C)
    fast = assembly._element_terms_numpy(pts, cells, C)
    for a, b in zip(ref, fast, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12 * np.abs(a).max())


@pytest.mark.parametrize("force_interp", [False, True])
def test_homogenize_kernel_env_identical(monkeypatch, force_interp):
    pts, cells = _mesh()
    C = _gp_C(len(cells), seed=1)
    monkeypatch.setenv("B3_CORE_KERNEL", "reference")
    monkeypatch.setenv("B3_CORE_SOLVER", "splu")
    ref, _ = assembly.homogenize_aniso(pts, cells, C, _force_z_interp=force_interp)
    monkeypatch.setenv("B3_CORE_KERNEL", "numpy")
    fast, _ = assembly.homogenize_aniso(pts, cells, C, _force_z_interp=force_interp)
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())


def test_bad_kernel_env(monkeypatch):
    monkeypatch.setenv("B3_CORE_KERNEL", "fortran")
    with pytest.raises(ValueError):
        assembly._kernel_choice()


def _spd(n=60, seed=2):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.1, random_state=rng)
    return (A @ A.T + n * sp.eye(n)).tocsc()


def test_block_solve_equals_columnwise():
    A = _spd()
    B = np.random.default_rng(3).normal(size=(A.shape[0], 6))
    solve = linsolve.factorize(A, solver="splu")
    block = solve(B)
    cols = np.column_stack([solve(B[:, k]) for k in range(6)])
    np.testing.assert_allclose(block, cols, rtol=1e-12, atol=0)


def test_pardiso_matches_splu():
    pytest.importorskip("pypardiso")
    A = _spd(n=200)
    B = np.random.default_rng(4).normal(size=(A.shape[0], 6))
    x_ref = linsolve.factorize(A, solver="splu")(B)
    x = linsolve.factorize(A, solver="pardiso")(B)
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        linsolve.factorize(A, solver="pardiso")(B[:, 0]), x_ref[:, 0], rtol=1e-8
    )


def test_fallback_without_pypardiso(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "pypardiso":
            raise ImportError("hidden")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.delenv("B3_CORE_SOLVER", raising=False)
    assert linsolve.resolve_solver() == "splu"
    with pytest.raises(ImportError):
        linsolve.resolve_solver("pardiso")
    A = _spd()
    b = np.ones(A.shape[0])
    np.testing.assert_allclose(A @ linsolve.factorize(A)(b), b, rtol=1e-10)


def test_solver_env_validation(monkeypatch):
    monkeypatch.setenv("B3_CORE_SOLVER", "umfpack")
    with pytest.raises(ValueError):
        linsolve.resolve_solver()
    monkeypatch.setenv("B3_CORE_SOLVER", "splu")
    assert linsolve.resolve_solver() == "splu"
