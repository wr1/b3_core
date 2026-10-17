"""Exact calibration to a flat test: as many free knobs as identifiable targets.

The flat test of a symmetric grooved cell measures five independent numbers:
in-plane E (Exx = Eyy, averaged), Ezz, transverse shear G (Gxz = Gyz,
averaged), in-plane shear Gxy and infused density. With the same number of
bounded material knobs the fit is a square root-finding problem, so the
residual can be driven to zero instead of to a least-squares compromise.

Deterministic steps, each a separate primitive:

1. :func:`jacobian` / :func:`identifiability` – log-target Jacobian in unit
   space (bounds → [-1, 1]); rank, singular values, condition number;
   :func:`rank_subsets` ranks candidate knob sets by conditioning.
2. :func:`ccd` – face-centred central composite design (half fraction from
   five knobs up), one true solve per row.
3. :class:`QuadSurface` – full quadratic in log(property), plain least squares.
4. :func:`solve_surface` – bounded Newton (trust-region least squares) on the
   surface.
5. :func:`exact_fit` – chains 2–4, then polishes on true solves with output
   space mapping: shift the target by the surface error at the current point
   and re-solve the surface. One or two true solves usually reach < 0.01 %.

Core stiffness is written as cubic-symmetric (Ex = Ey = Ez = E, all G equal,
ν fixed) so core E and core G are independent knobs; for G = E / 2(1+ν) the
result is identical to the isotropic material.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Solver = Callable[[dict, float, float], dict]


@dataclass(frozen=True)
class Knob:
    """One free material parameter. ``scale`` converts SI to the display unit."""

    name: str
    label: str
    unit: str
    lower: float
    upper: float
    scale: float = 1.0


KNOBS: dict[str, Knob] = {
    k.name: k
    for k in (
        Knob("halo_reach", "halo reach (resin seepage)", "mm", 0.2, 0.9),
        Knob("core_E", "core E", "MPa", 60e6, 110e6, 1e6),
        Knob("core_G", "core G", "MPa", 22e6, 40e6, 1e6),
        Knob("resin_E", "resin E", "GPa", 2.8e9, 3.6e9, 1e9),
        Knob("resin_nu", "resin ν", "–", 0.30, 0.40),
        Knob("resin_rho", "resin ρ", "kg/m³", 1100.0, 1250.0),
        Knob("core_rho", "core ρ", "kg/m³", 72.0, 88.0),
    )
}

# Symmetric cell: the averaged pairs are what a test can tell apart.
TARGETS: dict[str, tuple[str, ...]] = {
    "E_in": ("Exx", "Eyy"),
    "Ezz": ("Ezz",),
    "G_tr": ("Gxz", "Gyz"),
    "Gxy": ("Gxy",),
    "rho": ("rho_infused",),
}
TARGET_LABELS = {
    "E_in": "E_in = (Exx+Eyy)/2",
    "Ezz": "Ezz",
    "G_tr": "G_tr = (Gxz+Gyz)/2",
    "Gxy": "Gxy",
    "rho": "infused ρ",
}


def knob(name: str, bounds: Mapping[str, Sequence[float]] | None = None) -> Knob:
    base = KNOBS[name]
    if bounds and name in bounds:
        lo, hi = (float(v) for v in bounds[name])
        return Knob(base.name, base.label, base.unit, lo, hi, base.scale)
    return base


# --------------------------------------------------------------------------- #
# case <-> knobs
# --------------------------------------------------------------------------- #
def cubic_core(core: Mapping[str, Any]) -> dict:
    """Isotropic ``E, nu`` → cubic orthotropic with independent ``G``."""
    c = dict(core)
    if "E" in c:
        e, nu = float(c.pop("E")), float(c.pop("nu", 0.3))
        g = e / (2 * (1 + nu))
        c.update(Ex=e, Ey=e, Ez=e, Gxy=g, Gxz=g, Gyz=g, nuxy=nu, nuxz=nu, nuyz=nu)
    return c


def apply_knobs(case: Mapping[str, Any], values: Mapping[str, float]) -> dict:
    """Copy ``case`` with knob values written in. Unknown names raise."""
    d = copy.deepcopy(dict(case))
    if {"core_E", "core_G"} & set(values):
        d["core"] = cubic_core(d["core"])
    for name, v in values.items():
        v = float(v)
        if name == "halo_reach":
            d["core"]["cell_size"] = v
        elif name == "core_E":
            d["core"].update(Ex=v, Ey=v, Ez=v)
        elif name == "core_G":
            d["core"].update(Gxy=v, Gxz=v, Gyz=v)
        elif name == "resin_E":
            d["resin"]["E"] = v
        elif name == "resin_nu":
            d["resin"]["nu"] = v
        elif name == "resin_rho":
            d["resin"]["rho"] = v
        elif name == "core_rho":
            d["core"]["rho"] = v
        else:
            raise KeyError(f"unknown knob {name!r}; known: {sorted(KNOBS)}")
    return d


def read_knobs(case: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    """Current knob values of a case (the nominal start)."""
    core, resin = cubic_core(case["core"]), case["resin"]
    get = {
        "halo_reach": lambda: float(core.get("cell_size") or 0.0),
        "core_E": lambda: float(core["Ex"]),
        "core_G": lambda: float(core["Gxz"]),
        "resin_E": lambda: float(resin["E"]),
        "resin_nu": lambda: float(resin["nu"]),
        "resin_rho": lambda: float(resin["rho"]),
        "core_rho": lambda: float(core["rho"]),
    }
    return {n: get[n]() for n in names}


def targets_of(props: Mapping[str, float], names: Sequence[str] = tuple(TARGETS)) -> dict:
    """Averaged symmetric targets from raw solver/test properties."""
    return {n: float(np.mean([props[p] for p in TARGETS[n]])) for n in names}


# --------------------------------------------------------------------------- #
# evaluator with an optional resumable store
# --------------------------------------------------------------------------- #
class Evaluator:
    """Flat true solves at knob vectors, memoised (and stored as JSON lines).

    The store lets a killed job resume without re-solving.
    """

    def __init__(
        self,
        solve: Solver,
        case: Mapping[str, Any],
        knobs: Sequence[Knob],
        targets: Sequence[str] = tuple(TARGETS),
        store: str | Path | None = None,
    ):
        self.solve, self.case = solve, copy.deepcopy(dict(case))
        self.knobs, self.targets = list(knobs), list(targets)
        self.store = Path(store) if store else None
        self.memo: dict[str, dict] = {}
        self.n_solves = 0
        self._salt = hashlib.sha256(
            json.dumps(self.case, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        if self.store and self.store.is_file():
            for line in self.store.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.memo[rec["key"]] = rec["props"]

    def to_unit(self, x: Sequence[float]) -> np.ndarray:
        return np.array(
            [2 * (v - k.lower) / (k.upper - k.lower) - 1 for v, k in zip(x, self.knobs, strict=True)]
        )

    def from_unit(self, u: Sequence[float]) -> np.ndarray:
        return np.array(
            [k.lower + 0.5 * (v + 1) * (k.upper - k.lower) for v, k in zip(u, self.knobs, strict=True)]
        )

    def _key(self, x: np.ndarray) -> str:
        return self._salt + ":" + ",".join(f"{v:.10g}" for v in x)

    def props(self, x: Sequence[float]) -> dict:
        x = np.asarray(x, float)
        key = self._key(x)
        if key not in self.memo:
            vals = dict(zip((k.name for k in self.knobs), x, strict=True))
            out = self.solve(apply_knobs(self.case, vals), 0.0, 0.0)
            props = {k: float(v) for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            self.memo[key] = props
            self.n_solves += 1
            if self.store:
                self.store.parent.mkdir(parents=True, exist_ok=True)
                with self.store.open("a") as fh:
                    fh.write(json.dumps({"key": key, "x": x.tolist(), "props": props}) + "\n")
        return self.memo[key]

    def log_targets(self, x: Sequence[float]) -> np.ndarray:
        t = targets_of(self.props(x), self.targets)
        return np.log([t[n] for n in self.targets])

    def log_targets_u(self, u: Sequence[float]) -> np.ndarray:
        return self.log_targets(self.from_unit(u))


# --------------------------------------------------------------------------- #
# identifiability
# --------------------------------------------------------------------------- #
def jacobian(ev: Evaluator, u0: Sequence[float] | None = None, step: float = 0.1) -> np.ndarray:
    """Central-difference d ln(target) / d u, shape (targets, knobs)."""
    u0 = np.zeros(len(ev.knobs)) if u0 is None else np.asarray(u0, float)
    cols = []
    for i in range(len(u0)):
        up, dn = u0.copy(), u0.copy()
        up[i], dn[i] = min(1.0, u0[i] + step), max(-1.0, u0[i] - step)
        cols.append((ev.log_targets_u(up) - ev.log_targets_u(dn)) / (up[i] - dn[i]))
    return np.column_stack(cols)


def identifiability(j: np.ndarray, rtol: float = 1e-6) -> dict:
    """Rank, singular values and 2-norm condition number of a Jacobian."""
    s = np.linalg.svd(np.asarray(j, float), compute_uv=False)
    rank = int(np.sum(s > rtol * s[0])) if s.size else 0
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")
    return {"rank": rank, "singular_values": s.tolist(), "condition": cond}


def rank_subsets(
    j: np.ndarray, names: Sequence[str], k: int, rows: Sequence[int] | None = None
) -> list[tuple[float, tuple[str, ...]]]:
    """Condition number of every ``k``-knob column subset, best first."""
    jj = np.asarray(j, float)
    if rows is not None:
        jj = jj[list(rows)]
    out = []
    for combo in itertools.combinations(range(len(names)), k):
        out.append((identifiability(jj[:, combo])["condition"], tuple(names[c] for c in combo)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# design and response surface
# --------------------------------------------------------------------------- #
def ccd(n: int) -> np.ndarray:
    """Face-centred CCD in [-1, 1]^n. Half fraction (last = product) from n = 5."""
    full = np.array(list(itertools.product((-1.0, 1.0), repeat=n if n < 5 else n - 1)))
    if n >= 5:
        full = np.column_stack([full, np.prod(full, axis=1)])
    axial = np.vstack([s * np.eye(n)[i] for i in range(n) for s in (-1.0, 1.0)])
    return np.vstack([np.zeros((1, n)), full, axial])


def _phi(u: np.ndarray) -> np.ndarray:
    u = np.atleast_2d(u)
    n = u.shape[1]
    cols = [np.ones(len(u)), *u.T, *(u.T**2)]
    cols += [u[:, i] * u[:, j] for i in range(n) for j in range(i + 1, n)]
    return np.column_stack(cols)


def _dphi(u: np.ndarray) -> np.ndarray:
    """d phi / d u, shape (terms, n)."""
    u = np.asarray(u, float)
    n = u.size
    rows = [np.zeros(n)]
    rows += list(np.eye(n))
    rows += [2 * u[i] * np.eye(n)[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = np.zeros(n)
            r[i], r[j] = u[j], u[i]
            rows.append(r)
    return np.array(rows)


@dataclass
class QuadSurface:
    """Full quadratic in unit space for every log target."""

    coef: np.ndarray  # (terms, targets)
    fit_rms_pct: float = 0.0
    loo_max_pct: float = 0.0

    @classmethod
    def fit(cls, u: np.ndarray, logy: np.ndarray) -> QuadSurface:
        phi = _phi(u)
        if phi.shape[0] < phi.shape[1]:
            raise ValueError(f"{phi.shape[0]} design rows < {phi.shape[1]} quadratic terms")
        coef, *_ = np.linalg.lstsq(phi, logy, rcond=None)
        res = phi @ coef - logy
        # Leave-one-out via the hat matrix: e_i / (1 - h_ii).
        hat = np.einsum("ij,ji->i", phi, np.linalg.pinv(phi))
        loo = res / np.clip(1 - hat[:, None], 1e-9, None)
        return cls(
            coef=coef,
            fit_rms_pct=float(100 * np.sqrt(np.mean(res**2))),
            loo_max_pct=float(100 * np.max(np.abs(loo))) if np.all(hat < 1 - 1e-9) else float("nan"),
        )

    def predict(self, u: Sequence[float]) -> np.ndarray:
        return (_phi(np.asarray(u, float)) @ self.coef)[0]

    def jac(self, u: Sequence[float]) -> np.ndarray:
        return self.coef.T @ _dphi(np.asarray(u, float))


def solve_surface(
    surf: QuadSurface, logy: np.ndarray, starts: Sequence[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Bounded Newton on the surface from several starts. Returns (u, max|r|)."""
    from scipy.optimize import least_squares

    best = None
    for u0 in starts:
        r = least_squares(
            lambda u: surf.predict(u) - logy,
            np.clip(np.asarray(u0, float), -1, 1),
            jac=surf.jac,
            bounds=(-1.0, 1.0),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            method="trf",
        )
        m = float(np.max(np.abs(r.fun)))
        if best is None or m < best[1] - 1e-15:
            best = (r.x, m)
    return best


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
@dataclass
class ExactFit:
    knobs: list[str]
    lower: list[float]
    upper: list[float]
    start: list[float]
    fitted: list[float]
    measured: dict[str, float]
    model: dict[str, float]
    residual_pct: dict[str, float]
    condition: float
    singular_values: list[float]
    jacobian: list[list[float]]
    history: list[dict]
    n_design: int
    n_solves: int
    surface_rms_pct: float
    surface_loo_pct: float
    start_model: dict[str, float] = field(default_factory=dict)
    true: list[float] | None = None
    at_bound: list[str] = field(default_factory=list)
    label: str = "test"
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def max_abs_residual_pct(self) -> float:
        return max(abs(v) for v in self.residual_pct.values())

    def values(self) -> dict[str, float]:
        return dict(zip(self.knobs, self.fitted, strict=True))

    def apply(self, case: Mapping[str, Any]) -> dict:
        return apply_knobs(case, self.values())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExactFit:
        return cls(**dict(d))


def exact_fit(
    solve: Solver,
    case: Mapping[str, Any],
    measured: Mapping[str, float],
    knobs: Sequence[str],
    *,
    targets: Sequence[str] | None = None,
    bounds: Mapping[str, Sequence[float]] | None = None,
    tol_pct: float = 0.01,
    max_polish: int = 3,
    store: str | Path | None = None,
    true: Sequence[float] | None = None,
    label: str = "test",
    note: str = "",
) -> ExactFit:
    """Square, bounded, exact fit of ``knobs`` to the flat ``measured`` targets.

    ``measured`` may hold raw test properties (Exx, Eyy, ...) or averaged
    targets (``E_in``...). ``tol_pct`` is the max |residual| that stops the
    true-solve polish.
    """
    targets = list(targets or TARGETS)
    if len(targets) != len(knobs):
        raise ValueError(f"exact fit needs as many knobs ({len(knobs)}) as targets ({len(targets)})")
    meas = {n: float(measured[n]) if n in measured else targets_of(measured, [n])[n] for n in targets}
    ks = [knob(n, bounds) for n in knobs]
    ev = Evaluator(solve, case, ks, targets, store)
    logy = np.log([meas[n] for n in targets])
    x_start = np.array([read_knobs(case, knobs)[n] for n in knobs])
    x_start = np.clip(x_start, [k.lower for k in ks], [k.upper for k in ks])

    design_u = ccd(len(ks))
    ly = np.array([ev.log_targets_u(u) for u in design_u])
    surf = QuadSurface.fit(design_u, ly)
    dist = np.max(np.abs(ly - logy), axis=1)
    starts = [np.zeros(len(ks)), ev.to_unit(x_start), design_u[int(np.argmin(dist))]]
    u, m_s = solve_surface(surf, logy, starts)
    history = []
    for it in range(max_polish + 1):
        lfe = ev.log_targets_u(u)
        r = lfe - logy
        history.append(
            {
                "stage": "surface" if it == 0 else f"polish {it}",
                "x": ev.from_unit(u).tolist(),
                "max_abs_pct": float(100 * np.max(np.abs(np.expm1(r)))),
            }
        )
        if history[-1]["max_abs_pct"] < tol_pct or it == max_polish:
            break
        # Output space mapping: aim the surface at target − (surface error here).
        shifted = logy - (lfe - surf.predict(u))
        u, _ = solve_surface(surf, shifted, [u])
    x = ev.from_unit(u)
    model = targets_of(ev.props(x), targets)
    j = surf.jac(u)
    ident = identifiability(j)
    at_bound = [k.name for k, v in zip(ks, u, strict=True) if abs(v) > 1 - 1e-6]
    start_model = targets_of(ev.props(x_start), targets)
    return ExactFit(
        knobs=list(knobs),
        lower=[k.lower for k in ks],
        upper=[k.upper for k in ks],
        start=x_start.tolist(),
        fitted=x.tolist(),
        measured=meas,
        model=model,
        residual_pct={n: 100 * (model[n] / meas[n] - 1) for n in targets},
        condition=ident["condition"],
        singular_values=ident["singular_values"],
        jacobian=j.tolist(),
        history=history,
        n_design=len(design_u),
        n_solves=ev.n_solves,
        surface_rms_pct=surf.fit_rms_pct,
        surface_loo_pct=surf.loo_max_pct,
        start_model=start_model,
        true=None if true is None else [float(v) for v in true],
        at_bound=at_bound,
        label=label,
        note=note,
    )


def mock_test(
    solve: Solver,
    case: Mapping[str, Any],
    truth: Mapping[str, float],
    *,
    noise: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Raw flat 'test' properties from a hidden truth, optionally with noise.

    ``noise`` maps raw property → relative sd. Noise is drawn per raw
    property (Exx and Eyy separately, as a test would), in a fixed order.
    """
    raw = sorted({p for ps in TARGETS.values() for p in ps})
    props = solve(apply_knobs(case, truth), 0.0, 0.0)
    out = {p: float(props[p]) for p in raw}
    if noise:
        rng = np.random.default_rng(seed)
        for p in raw:
            out[p] *= 1 + float(noise.get(p, 0.0)) * rng.standard_normal()
    return out
