"""Physics-based surrogate for grooved-core stiffness / mass vs curvature.

Pattern matches ``b3_invsec.physics_surrogate``: a closed-form structural base
carries the trend, and a small log-space polynomial correction is fit to FEM
(or numpy-homogenization) rows:

    T(x)  =  physics_base(x) · exp( φ(x) @ c )

Features are dimensionless design knobs (curvature ``kx``, optional halo
``cell_size``). Targets are engineering moduli, resin volumes, and infused
density / areal mass.

Hot path for panel station lookup:

    surr = fit_physics_surrogate(df)          # or load JSON
    out  = surr.lookup(kx_vector, cell_size=0.6)
    # out: DataFrame with Exx…Gyz, rho_infused, mass_per_m2, resin_vf, …

The base uses the same kerf taper law as the mesh morph
(``slope = −sign(d)·κ·pitch/2``) to estimate neat resin volume, a simple
opened-cell band for the halo, Voigt/Reuss mixture rules for channel vs
cross-channel moduli, and the two-phase density for mass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Targets with a defensible closed-form base + multiplicative log correction.
TARGETS = (
    "Exx",
    "Eyy",
    "Ezz",
    "Gxy",
    "Gxz",
    "Gyz",
    "resin_vf",
    "halo_vf",
    "effective_resin_vf",
    "rho_infused",
    "mass_per_m2",
)

FEATURE_NAMES = ("kx", "cell_size")


def _col(feat, name: str, default: float = 0.0) -> np.ndarray:
    if isinstance(feat, pd.DataFrame):
        if name in feat.columns:
            v = feat[name].to_numpy(dtype=float)
        else:
            v = np.full(len(feat), default, dtype=float)
    elif isinstance(feat, dict):
        v = np.asarray(feat.get(name, default), dtype=float)
    else:
        raise TypeError(f"feat must be DataFrame or dict, got {type(feat)}")
    return np.atleast_1d(v).astype(float)


def _as_feat(X, features: list[str]) -> dict[str, np.ndarray]:
    """Accept DataFrame, dict of arrays, or ndarray in ``features`` order."""
    if isinstance(X, pd.DataFrame):
        return {c: X[c].to_numpy(dtype=float) if c in X.columns else np.zeros(len(X)) for c in features}
    if isinstance(X, dict):
        n = len(np.atleast_1d(next(iter(X.values()))))
        return {c: np.atleast_1d(np.asarray(X.get(c, 0.0), dtype=float)) for c in features}
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    if arr.shape[1] != len(features):
        raise ValueError(
            f"X has {arr.shape[1]} columns, expected {len(features)} ({features})"
        )
    return {c: arr[:, i] for i, c in enumerate(features)}


@dataclass
class GeometrySpec:
    """Fixed RVE geometry / materials used by the physics base.

    Lengths in **mm**, moduli/densities in **SI** (Pa, kg/m³) — same as cprop.
    """

    dx: float = 30.0
    dy: float = 12.0
    thickness: float = 20.0
    # Single x-family: [offset, pitch, depth, width]
    offset: float = 5.0
    pitch: float = 10.0
    depth: float = -17.0
    width: float = 2.0
    E_core: float = 32e6
    E_core_z: float = 70e6
    G_core: float = 19e6
    E_resin: float = 3e9
    G_resin: float = 3e9 / (2.0 * (1.0 + 0.3))
    rho_core: float = 60.0
    rho_resin: float = 1100.0
    min_hw: float = 0.05  # matches mesh _MIN_HW order

    @classmethod
    def from_case(cls, case: dict) -> GeometrySpec:
        """Pull geometry + isotropic/orthotropic materials from a cprop-like case."""
        xgr = (case.get("xgr") or [[5.0, 10.0, -17.0, 2.0]])[0]
        core = case.get("core") or {}
        resin = case.get("resin") or {}
        if "E1" in core:
            Ec, Ecz = float(core["E1"]), float(core.get("E3", core["E1"]))
            Gc = float(core.get("G12", core.get("G13", Ec / 2.6)))
        else:
            Ec = float(core.get("E", 32e6))
            Ecz = Ec
            nu = float(core.get("nu", 0.3))
            Gc = Ec / (2.0 * (1.0 + nu))
        if "E1" in resin:
            Er = float(resin["E1"])
            Gr = float(resin.get("G12", Er / 2.6))
        else:
            Er = float(resin.get("E", 3e9))
            nu_r = float(resin.get("nu", 0.3))
            Gr = Er / (2.0 * (1.0 + nu_r))
        return cls(
            dx=float(case.get("dx", 30.0)),
            dy=float(case.get("dy", 12.0)),
            thickness=float(case.get("thickness", 20.0)),
            offset=float(xgr[0]),
            pitch=float(xgr[1]),
            depth=float(xgr[2]),
            width=float(xgr[3]),
            E_core=Ec,
            E_core_z=Ecz,
            G_core=Gc,
            E_resin=Er,
            G_resin=Gr,
            rho_core=float(core.get("rho", 60.0)),
            rho_resin=float(resin.get("rho", 1100.0)),
        )


def _n_kerfs(geom: GeometrySpec) -> float:
    """Approximate number of kerf instances intersecting [0, dx] (meshadd=[0] lattice)."""
    hw = 0.5 * geom.width
    lefts = np.arange(geom.offset - geom.pitch - hw, geom.dx + geom.pitch, geom.pitch)
    n = 0
    for left in lefts:
        right = left + geom.width
        if right > 1e-9 and left < geom.dx - 1e-9:
            n += 1
    return float(max(n, 1))


def physics_base(feat, geom: GeometrySpec | None = None) -> dict[str, np.ndarray]:
    """Closed-form bases for moduli, volumes, density, and areal mass.

    Kerf taper: ``slope = −sign(depth)·kx·pitch/2``,
    ``hw_mouth = max(ε, hw0 + slope·|depth|)``, trapezoid mean half-width.
    """
    g = geom or GeometrySpec()
    kx = _col(feat, "kx", 0.0)
    cs = np.maximum(_col(feat, "cell_size", 0.0), 0.0)
    n = max(len(kx), len(cs))
    if len(kx) == 1 and n > 1:
        kx = np.full(n, float(kx[0]))
    if len(cs) == 1 and n > 1:
        cs = np.full(n, float(cs[0]))

    hw0 = 0.5 * g.width
    depth_abs = abs(g.depth)
    slope = -np.sign(g.depth) * kx * g.pitch / 2.0
    hw_mouth = np.maximum(g.min_hw, hw0 + slope * depth_abs)
    hw_root = np.full_like(hw_mouth, hw0)
    # Clip walls that would leave the cell (extreme close already at ε).
    hw_mouth = np.minimum(hw_mouth, 0.45 * g.pitch)
    hw_avg = 0.5 * (hw_mouth + hw_root)

    n_k = _n_kerfs(g)
    # 2-D resin area in the x–z plane per unit y; Vf = area / (dx·thickness).
    resin_area = n_k * 2.0 * hw_avg * depth_abs
    domain = g.dx * g.thickness
    resin_vf = np.clip(resin_area / domain, 1e-6, 0.85)

    # Opened-cell halo: two walls × depth × reach, mean P ≈ 1/2 for linear S(d).
    reach = cs
    wall_area = n_k * 2.0 * depth_abs  # mm² of wall surface in x–z (unit y)
    halo_vf = np.clip(0.5 * wall_area * reach / domain, 0.0, 0.5)
    # Halo only outside the neat kerf; cap so total < 1.
    halo_vf = np.minimum(halo_vf, 0.9 * (1.0 - resin_vf))
    eff_vf = np.clip(resin_vf + halo_vf, 1e-6, 0.95)

    # Mixture rules (channel layout: x-grooves → resin channels || y).
    Ec, Er = g.E_core, g.E_resin
    Ecz, Gc, Gr = g.E_core_z, g.G_core, g.G_resin
    # Voigt along channels (y); Reuss across (x); thickness blend.
    Eyy = (1.0 - eff_vf) * Ec + eff_vf * Er
    Exx = 1.0 / np.maximum((1.0 - eff_vf) / Ec + eff_vf / Er, 1e-30)
    Ezz = 1.0 / np.maximum((1.0 - eff_vf) / Ecz + eff_vf / Er, 1e-30)
    # In-plane shear: soft matrix-dominated; use Reuss-like G.
    Gxy = 1.0 / np.maximum((1.0 - eff_vf) / Gc + eff_vf / Gr, 1e-30)
    Gxz = Gxy
    Gyz = (1.0 - eff_vf) * Gc + eff_vf * Gr

    rho = (1.0 - eff_vf) * g.rho_core + eff_vf * g.rho_resin
    # Areal mass [kg/m²]: geometry mm → m.
    mass = rho * (g.thickness * 1e-3)

    return {
        "Exx": np.maximum(Exx, 1e3),
        "Eyy": np.maximum(Eyy, 1e3),
        "Ezz": np.maximum(Ezz, 1e3),
        "Gxy": np.maximum(Gxy, 1e2),
        "Gxz": np.maximum(Gxz, 1e2),
        "Gyz": np.maximum(Gyz, 1e2),
        "resin_vf": resin_vf,
        "halo_vf": halo_vf,
        "effective_resin_vf": eff_vf,
        "rho_infused": np.maximum(rho, 1.0),
        "mass_per_m2": np.maximum(mass, 1e-6),
    }


def _phi(feat, n: int | None = None) -> np.ndarray:
    """Log-correction design matrix: 1, kx, kx², cs, cs², kx·cs."""
    kx = _col(feat, "kx", 0.0)
    cs = _col(feat, "cell_size", 0.0)
    m = n or max(len(kx), len(cs))
    if len(kx) == 1 and m > 1:
        kx = np.full(m, float(kx[0]))
    if len(cs) == 1 and m > 1:
        cs = np.full(m, float(cs[0]))
    # Scale for conditioning (kx ~ 1e-2, cs ~ 1 mm).
    k = kx * 100.0  # now O(1)
    c = cs  # mm
    return np.column_stack(
        [
            np.ones(m),
            k,
            k**2,
            c,
            c**2,
            k * c,
        ]
    )


def _ridge_lstsq(Phi: np.ndarray, r: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    AtA = Phi.T @ Phi
    k = AtA.shape[0]
    scale = float(np.trace(AtA) / max(k, 1))
    return np.linalg.solve(AtA + (lam * scale) * np.eye(k), Phi.T @ r)


@dataclass
class CorePhysicsSurrogate:
    """Fitted physics base + log correction for core properties vs curvature.

    Drop-in style API (``features``, ``targets``, ``predict``) analogous to
    ``b3_invsec.physics_surrogate.PhysicsSurrogate``.
    """

    coefs: dict[str, list[float]]  # target -> φ coefficients
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
    geometry: dict[str, float] = field(default_factory=dict)

    def _geom(self) -> GeometrySpec:
        if not self.geometry:
            return GeometrySpec()
        return GeometrySpec(**{k: self.geometry[k] for k in GeometrySpec.__dataclass_fields__ if k in self.geometry})

    def predict(
        self,
        X,
        targets: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Vectorised prediction.

        ``X`` — DataFrame with ``kx`` / ``cell_size``, dict of arrays, or
        ``(n, n_features)`` ndarray in ``self.features`` order.
        """
        feat = _as_feat(X, self.features)
        want = self.targets if targets is None else [t for t in targets if t in self.targets]
        base = physics_base(feat, self._geom())
        phi = _phi(feat)
        out: dict[str, np.ndarray] = {}
        for t in want:
            b = base[t]
            c = np.asarray(self.coefs.get(t, np.zeros(phi.shape[1])), dtype=float)
            if c.shape[0] != phi.shape[1]:
                c = np.zeros(phi.shape[1])
            out[t] = b * np.exp(phi @ c)
        return out

    def lookup(
        self,
        kx,
        *,
        cell_size: float | np.ndarray = 0.0,
        targets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Mass lookup: properties at each station curvature.

        Parameters
        ----------
        kx
            1-D array of curvatures [1/mm] (e.g. panel stations).
        cell_size
            Scalar or per-station halo width [mm]; 0 = sharp kerf.
        """
        kx_a = np.atleast_1d(np.asarray(kx, dtype=float))
        cs_a = np.asarray(cell_size, dtype=float)
        if cs_a.ndim == 0:
            cs_a = np.full(len(kx_a), float(cs_a))
        if len(cs_a) != len(kx_a):
            raise ValueError("cell_size must be scalar or match len(kx)")
        X = np.column_stack([kx_a, cs_a])
        pred = self.predict(X, targets=targets)
        df = pd.DataFrame(pred)
        df.insert(0, "kx", kx_a)
        df.insert(1, "cell_size", cs_a)
        return df

    def to_json(self, path: str | Path | None = None) -> dict[str, Any]:
        payload = {
            "coefs": {k: list(map(float, v)) for k, v in self.coefs.items()},
            "features": list(self.features),
            "targets": list(self.targets),
            "geometry": dict(self.geometry),
        }
        if path is not None:
            Path(path).write_text(json.dumps(payload, indent=2))
        return payload

    @classmethod
    def from_json(cls, path: str | Path | dict) -> CorePhysicsSurrogate:
        data = path if isinstance(path, dict) else json.loads(Path(path).read_text())
        return cls(
            coefs={k: list(v) for k, v in data["coefs"].items()},
            features=list(data.get("features", FEATURE_NAMES)),
            targets=list(data.get("targets", TARGETS)),
            geometry=dict(data.get("geometry") or {}),
        )


def fit_physics_surrogate(
    df: pd.DataFrame,
    *,
    geometry: GeometrySpec | dict | None = None,
    targets: list[str] | None = None,
) -> CorePhysicsSurrogate:
    """Fit log-space corrections so ``base · exp(φ c) ≈ FEM`` on the training rows.

    ``df`` must contain feature columns (``kx``, optional ``cell_size``) and
    target columns among :data:`TARGETS`. Build such a table with
    :func:`build_training_frame` or any homogenization sweep.
    """
    if isinstance(geometry, dict):
        geom = GeometrySpec(**geometry)
    elif geometry is None:
        geom = GeometrySpec()
    else:
        geom = geometry

    feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
    if "kx" not in feat_cols:
        raise ValueError("training frame must include a 'kx' column")
    if "cell_size" not in df.columns:
        df = df.copy()
        df["cell_size"] = 0.0
        feat_cols = list(FEATURE_NAMES)

    want = [t for t in (targets or TARGETS) if t in df.columns]
    if not want:
        raise ValueError(f"no target columns found; expected one of {TARGETS}")

    base = physics_base(df, geom)
    phi = _phi(df)
    coefs: dict[str, list[float]] = {}
    for t in want:
        y = df[t].to_numpy(dtype=float)
        b = base[t]
        ok = np.isfinite(y) & np.isfinite(b) & (y > 0) & (b > 0)
        if ok.sum() < phi.shape[1] + 2:
            coefs[t] = [0.0] * phi.shape[1]
            continue
        r = np.log(y[ok]) - np.log(b[ok])
        c = _ridge_lstsq(phi[ok], r)
        # Median level shift (robust intercept).
        c[0] += float(np.median(r - phi[ok] @ c))
        coefs[t] = list(map(float, c))

    return CorePhysicsSurrogate(
        coefs=coefs,
        features=list(FEATURE_NAMES),
        targets=want,
        geometry={k: float(getattr(geom, k)) for k in GeometrySpec.__dataclass_fields__},
    )


def build_training_frame(
    *,
    kx_values: list[float] | np.ndarray | None = None,
    cell_sizes: list[float] | np.ndarray | None = None,
    base_case: dict | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run (or load) a κ × cell_size homogenization grid as a training table.

    Wraps :func:`b3_core.viz.halo.sweep_halo_curvature_grid` and adds
    ``mass_per_m2`` from density × thickness.
    """
    from b3_core.viz.halo import (
        _parametric_base_case,
        sweep_halo_curvature_grid,
    )

    case = base_case or _parametric_base_case()
    rows = sweep_halo_curvature_grid(
        kx_values=kx_values,
        cell_sizes=cell_sizes,
        base=case,
        cache_path=cache_path,
    )
    df = pd.DataFrame(rows)
    th_m = float(case.get("thickness", 20.0)) * 1e-3
    if "rho_infused" not in df.columns:
        # reconstruct from vf if needed
        core = case.get("core") or {}
        resin = case.get("resin") or {}
        rc = float(core.get("rho", 60.0))
        rr = float(resin.get("rho", 1100.0))
        vf = df.get("effective_resin_vf", df["resin_vf"]).to_numpy(float)
        df["rho_infused"] = (1.0 - vf) * rc + vf * rr
    df["mass_per_m2"] = df["rho_infused"].to_numpy(float) * th_m
    return df


def fit_from_homogenization(
    *,
    kx_values: list[float] | np.ndarray | None = None,
    cell_sizes: list[float] | np.ndarray | None = None,
    base_case: dict | None = None,
    cache_path: str | Path | None = None,
) -> CorePhysicsSurrogate:
    """Convenience: homogenize a grid, then fit the physics surrogate."""
    from b3_core.viz.halo import _parametric_base_case

    case = base_case or _parametric_base_case()
    df = build_training_frame(
        kx_values=kx_values,
        cell_sizes=cell_sizes,
        base_case=case,
        cache_path=cache_path,
    )
    return fit_physics_surrogate(df, geometry=GeometrySpec.from_case(case))
