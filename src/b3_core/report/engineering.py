"""Engineering datasheet: flat card, plain-foam baseline, curvature sweep, calibration.

Every number on the page comes from FE solves made here (FEniCSx by default):

* page 1: key numbers vs plain foam (absolute and per unit density), inputs and
  assumptions, provenance (backend, commit, mesh, convergence status), RVE iso
  view with the resin-halo band and a true-scale wall inset, phase cuts placed
  between the grooves, and the value-of-grooving bars;
* page 2: curvature map (% vs the flat card, shared symmetric coolwarm scale,
  mould radius ticks R = 1/(1000 k) m, closed-groove cells hatched) and the
  single-curvature line cut with the closure band;
* page 3 (optional): calibration to measured flat data: measured vs model,
  the fitted parameter with bounds and 95 % CI, residuals and the SSR scan.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import subprocess
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROPS = ("Exx", "Eyy", "Ezz", "Gxz", "Gyz", "Gxy")
CAL_PROPS = (*PROPS, "rho_infused")
MAP_PROPS = (
    ("Exx", "$E_{xx}$"),
    ("Eyy", "$E_{yy}$"),
    ("Ezz", "$E_{zz}$"),
    ("Gxz", "$G_{xz}$"),
    ("Gyz", "$G_{yz}$"),
    ("rho_infused", "infused ρ"),
)
DEFAULT_RADII_M = (0.2, 0.5, 1.0, 2.0)
# Finer than the package default: Ezz within ~1.7 % of h→0 (docs convergence).
ENGINEERING_H_MAX = 1.0
# Above this flat-vs-k→0 gap the curved and flat solves are not one model.
CONTINUITY_TOL_PCT = 1.0
_TEX = {
    "Exx": "$E_(x x)$",
    "Eyy": "$E_(y y)$",
    "Ezz": "$E_(z z)$",
    "Gxz": "$G_(x z)$",
    "Gyz": "$G_(y z)$",
    "Gxy": "$G_(x y)$",
    "rho_infused": "ρ [kg/m³]",
}

Solver = Callable[[dict, float, float], dict]


# --------------------------------------------------------------------------- #
# number formatting
# --------------------------------------------------------------------------- #
def fmt_modulus(value_pa: float) -> str:
    """Three significant figures, MPa below 1 GPa, never ``0.4 GPa``."""
    v = float(value_pa)
    if abs(v) >= 1e9:
        return f"{v / 1e9:.3g} GPa"
    return f"{v / 1e6:.3g} MPa"


def fmt_mpa(value_pa: float) -> str:
    v = float(value_pa) / 1e6
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def closure_curvature(case: dict) -> dict[str, float | None]:
    """Signed curvature (1/mm) at which each axis' narrowest groove closes.

    A groove of half-width ``w/2``, depth ``d`` and pitch ``p`` pinches its
    mouth at ``|k| = w / (p·|d|)``; a top-mouth groove (negative depth) closes
    for ``k < 0``, a bottom-mouth one for ``k > 0``. ``kx`` acts on x-grooves.
    """
    out: dict[str, float | None] = {}
    for axis, key in (("kx", "xgr"), ("ky", "ygr")):
        best = None
        for row in case.get(key) or []:
            _off, pitch, depth, width = (float(v) for v in row[:4])
            if pitch <= 0 or depth == 0:
                continue
            k = width / (pitch * abs(depth))
            k = -k if depth < 0 else k
            if best is None or abs(k) < abs(best):
                best = k
        out[axis] = best
    return out


def is_clamped(case: dict, kx: float, ky: float) -> bool:
    """True when a groove is past closure and the morph clamps it shut."""
    close = closure_curvature(case)
    for axis, k in (("kx", kx), ("ky", ky)):
        kc = close.get(axis)
        if (
            kc is not None
            and k != 0
            and np.sign(k) == np.sign(kc)
            and abs(k) >= abs(kc)
        ):
            return True
    return False


def radius_label(k: float) -> str:
    return "flat" if k == 0 else f"{1 / (1000 * k):+g}"


# --------------------------------------------------------------------------- #
# solves
# --------------------------------------------------------------------------- #
def fenicsx_solver(backend: str = "fenicsx") -> Solver:
    from b3_core.api import run_case

    def solve(case: dict, kx: float = 0.0, ky: float = 0.0) -> dict:
        rec = run_case(
            copy.deepcopy(case), kx=kx or None, ky=ky or None, backend=backend
        )
        out = {
            k: float(v) for k, v in rec.flat().items() if isinstance(v, (int, float))
        }
        stiffness = getattr(rec.result, "stiffness", None)
        if stiffness is not None:
            out["_C"] = np.asarray(stiffness, dtype=float).tolist()
        out["_backend"] = (
            rec.result.backend if hasattr(rec.result, "backend") else backend
        )
        return out

    return solve


def _with_param(case: dict, path: str, value: float) -> dict:
    d = copy.deepcopy(case)
    node = d
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = float(value)
    return d


@dataclass
class Calibration:
    """Measured flat data and the one free model parameter to fit."""

    measured: dict[str, float]
    sigma_rel: dict[str, float]
    param: str = "core.cell_size"
    bounds: tuple[float, float] = (0.2, 0.9)
    start: float = 0.5
    label: str = "measured"
    note: str = ""

    @classmethod
    def from_any(cls, obj: Any) -> Calibration:
        if isinstance(obj, Calibration):
            return obj
        if isinstance(obj, (str, Path)):
            obj = json.loads(Path(obj).read_text(encoding="utf-8"))
        obj = dict(obj)
        obj["bounds"] = tuple(obj.get("bounds", (0.2, 0.9)))
        return cls(**obj)


@dataclass
class CalibrationResult:
    value: float
    se: float
    ci95: tuple[float, float]
    start_model: dict[str, float]
    fit_model: dict[str, float]
    residual_pct: dict[str, float]
    rms_pct: float
    chi2_red: float
    scan: list[tuple[float, float]] = field(default_factory=list)
    at_bound: bool = False


def calibrate(case: dict, cal: Calibration, solve: Solver) -> CalibrationResult:
    """Bounded weighted least squares on relative residuals, one parameter."""
    from scipy.optimize import least_squares

    props = [p for p in CAL_PROPS if p in cal.measured]
    meas = np.array([cal.measured[p] for p in props])
    w = 1.0 / np.array([cal.sigma_rel.get(p, 0.03) for p in props])
    memo: dict[float, np.ndarray] = {}

    def model(x: float) -> np.ndarray:
        key = round(float(x), 6)
        if key not in memo:
            f = solve(_with_param(case, cal.param, key), 0.0, 0.0)
            memo[key] = np.array([f[p] for p in props])
        return memo[key]

    lo, hi = cal.bounds
    for x in np.linspace(lo, hi, 5):
        model(x)
    scan_best = min(memo, key=lambda x: float(np.sum((w * (memo[x] / meas - 1)) ** 2)))
    ls = least_squares(
        lambda x: w * (model(x[0]) / meas - 1),
        [scan_best],
        bounds=([lo], [hi]),
        diff_step=1e-2,
        xtol=1e-4,
    )
    x = float(ls.x[0])
    dof = max(1, len(props) - 1)
    chi2 = float(2 * ls.cost / dof)
    jtj = float((ls.jac.T @ ls.jac)[0, 0])
    se = float(np.sqrt(chi2 / jtj)) if jtj > 0 else float("nan")
    fit = model(x)
    res = 100 * (fit / meas - 1)
    scan = sorted(
        (k, float(np.sum((w * (v / meas - 1)) ** 2))) for k, v in memo.items()
    )
    at_bound = min(x - lo, hi - x) < 1e-3 * (hi - lo)
    if at_bound:
        warnings.warn(
            f"calibrated {cal.param} = {x:.4g} sits on a bound {cal.bounds}",
            RuntimeWarning,
            stacklevel=2,
        )
    return CalibrationResult(
        value=x,
        se=se,
        ci95=(x - 1.96 * se, x + 1.96 * se),
        start_model=dict(zip(props, model(cal.start).tolist(), strict=True)),
        fit_model=dict(zip(props, fit.tolist(), strict=True)),
        residual_pct=dict(zip(props, res.tolist(), strict=True)),
        rms_pct=float(np.sqrt(np.mean(res**2))),
        chi2_red=chi2,
        scan=scan,
        at_bound=at_bound,
    )


@dataclass
class SheetData:
    case: dict
    flat: dict
    near_zero: dict
    plain: dict
    sweep: list[dict]
    ks: list[float]
    closure: dict
    clamped_points: int
    continuity_pct: dict[str, float]
    mesh_cells: int
    provenance: dict[str, str]
    calibration: Calibration | None = None
    cal_result: CalibrationResult | None = None


def collect(
    case: dict,
    *,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    biaxial: bool = True,
    calibration: Any = None,
    solve: Solver | None = None,
    backend: str = "fenicsx",
) -> SheetData:
    """Run every solve the sheet needs and return the numbers."""
    from b3_core.loaders import normalize_case
    from b3_core.pipeline import prepare

    solve = solve or fenicsx_solver(backend)
    case = copy.deepcopy(case)
    case.setdefault("mesh_h_max", ENGINEERING_H_MAX)
    case.setdefault("kerf_z_elems", 4)
    cal = cal_res = None
    if calibration is not None:
        cal = Calibration.from_any(calibration)
        cal_res = calibrate(case, cal, solve)
        case = _with_param(case, cal.param, cal_res.value)

    flat = solve(case, 0.0, 0.0)
    near = solve(case, 1e-9, 1e-9)
    plain_case = copy.deepcopy(case)
    plain_case["xgr"], plain_case["ygr"] = [], []
    plain = solve(plain_case, 0.0, 0.0)
    continuity = {p: 100 * (near[p] / flat[p] - 1) for p in PROPS}
    if max(abs(v) for v in continuity.values()) > CONTINUITY_TOL_PCT:
        warnings.warn(
            "curved solve at k→0 differs from the flat card by "
            f"{max(abs(v) for v in continuity.values()):.2f} %",
            RuntimeWarning,
            stacklevel=2,
        )

    ks = sorted(
        {0.0, *(1 / (1000 * r) for r in radii_m), *(-1 / (1000 * r) for r in radii_m)}
    )
    sweep = []
    pairs = (
        [(kx, ky) for ky in ks for kx in ks]
        if biaxial
        else [(k, 0.0) for k in ks] + [(0.0, k) for k in ks if k]
    )
    clamped = 0
    for kx, ky in pairs:
        f = flat if (kx == 0 and ky == 0) else solve(case, kx, ky)
        c = is_clamped(case, kx, ky)
        clamped += c
        sweep.append(
            {
                "kx": kx,
                "ky": ky,
                "clamped": c,
                **{p: f[p] for p in (*PROPS, "rho_infused")},
            }
        )
    if clamped:
        warnings.warn(
            f"{clamped} sweep point(s) are past groove closure; the morph clamps "
            "the kerf at its minimum width there and the values are a bound, "
            "not a prediction",
            RuntimeWarning,
            stacklevel=2,
        )

    ci, _ = normalize_case(copy.deepcopy(case))
    mesh_cells = int(prepare(ci).mesh.n_cells)
    prov = provenance(case, ci, flat.get("_backend", backend), mesh_cells, continuity)
    return SheetData(
        case=case,
        flat=flat,
        near_zero=near,
        plain=plain,
        sweep=sweep,
        ks=ks,
        closure=closure_curvature(case),
        clamped_points=clamped,
        continuity_pct=continuity,
        mesh_cells=mesh_cells,
        provenance=prov,
        calibration=cal,
        cal_result=cal_res,
    )


def provenance(
    case: dict, ci: Any, backend: str, cells: int, continuity: dict
) -> dict[str, str]:
    from b3_core.provenance import identity

    ident = identity(solver_stamp="")
    h, kz = ci.mesh_h_max, ci.kerf_z_elems
    known = {
        1.0: "engineering grid: in-plane and shears within 0.7 %, Ezz ≈ +1.7 % of h→0",
        1.5: "package default grid: within ~1 %, Ezz ≈ +2.4 % of h→0",
    }
    try:
        import dolfinx

        stack = f"dolfinx {dolfinx.__version__}"
    except Exception:  # pragma: no cover - only without FEniCSx
        stack = "dolfinx not importable"
    if h is None:
        status = "legacy coarse grid (deprecated, not converged)"
    elif kz == 4 and h in known:
        status = known[h]
    else:
        status = "custom grid; see docs/reference/convergence"
    return {
        "backend": f"{backend} ({stack}), periodic, 6 unit strains"
        if backend == "fenicsx"
        else str(backend),
        "code": f"b3_core {ident.version} · {ident.commit or 'unknown'}{' (dirty)' if ident.dirty else ''}",
        "mesh": f"{cells} hex cells · mesh_h_max {h} mm · kerf_z_elems {kz}",
        "convergence": status,
        "k→0 continuity": f"max {max(abs(v) for v in continuity.values()):.2f} % vs flat",
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    return mpl, plt


def _grid_pct(data: SheetData, prop: str) -> np.ndarray:
    n = len(data.ks)
    g = np.full((n, n), np.nan)
    for r in data.sweep:
        g[data.ks.index(r["ky"]), data.ks.index(r["kx"])] = 100 * (
            r[prop] / data.flat[prop] - 1
        )
    return g


def fig_heatmaps(data: SheetData, out: Path) -> Path:
    _mpl_mod, plt = _mpl()
    from matplotlib.colors import TwoSlopeNorm

    ks, n = data.ks, len(data.ks)
    grids = [_grid_pct(data, p) for p, _ in MAP_PROPS]
    vmax = max(
        5.0, float(np.ceil(np.nanmax([np.nanmax(np.abs(g)) for g in grids]) / 5) * 5)
    )
    clamp = np.zeros((n, n), bool)
    for r in data.sweep:
        clamp[ks.index(r["ky"]), ks.index(r["kx"])] = r["clamped"]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 5.4), constrained_layout=True)
    im = None
    for ax, (p, title), g in zip(axs.flat, MAP_PROPS, grids, strict=True):
        im = ax.imshow(
            g, origin="lower", cmap="coolwarm", norm=TwoSlopeNorm(0, -vmax, vmax)
        )
        ax.set_xticks(range(n), [radius_label(k) for k in ks], rotation=90)
        ax.set_yticks(range(n), [radius_label(k) for k in ks])
        ax.set_title(title + ("  (kx drives)" if p in ("Eyy", "Gyz") else ""))
        ax.spines[:].set_visible(False)
        for i in range(n):
            for j in range(n):
                if np.isnan(g[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{g[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    color="k" if abs(g[i, j]) < 0.6 * vmax else "w",
                )
                if clamp[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="////",
                            lw=0,
                            ec="0.25",
                        )
                    )
    fig.supxlabel(
        "mould radius $R_x$ = 1/(1000·$k_x$) [m]   (+ opens top-mouth grooves, − closes)",
        fontsize=8,
    )
    fig.supylabel("mould radius $R_y$ [m]", fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.8, label="% change vs flat card")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_line_cut(data: SheetData, out: Path) -> Path:
    _mpl_mod, plt = _mpl()
    cut = sorted((r for r in data.sweep if r["ky"] == 0), key=lambda r: r["kx"])
    x = np.array([r["kx"] for r in cut]) * 1000
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    colours = ["#3b4cc0", "#7b9ff9", "#b40426", "#f49a7b", "#555555", "#999999"]
    for (p, title), col in zip(MAP_PROPS, colours, strict=True):
        ax.plot(
            x,
            [100 * (r[p] / data.flat[p] - 1) for r in cut],
            "o-",
            ms=2.5,
            lw=1.2,
            label=title,
            color=col,
        )
    kc = data.closure.get("kx")
    pad = 0.04 * float(np.ptp(x) or 1.0)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    if kc is not None:
        edge = 1000 * kc
        r_close = abs(1 / kc) / 1000
        if x.min() <= edge <= x.max():
            lim = ax.get_xlim()[0] if edge < 0 else ax.get_xlim()[1]
            ax.axvspan(min(edge, lim), max(edge, lim), color="0.85", lw=0, zorder=0)
            ax.axvline(edge, ls="--", lw=0.8, color="k")
            ax.text(
                edge,
                0.97,
                f" grooves closed\n R≈{r_close:.2f} m",
                fontsize=6,
                va="top",
                ha="left" if edge < 0 else "right",
                transform=ax.get_xaxis_transform(),
            )
        else:
            ax.text(
                0.02,
                0.03,
                f"grooves close at R≈{r_close:.2f} m (outside this sweep)",
                fontsize=6,
                transform=ax.transAxes,
            )
    ax.axhline(0, lw=0.5, color="k")
    ax.set_xlabel("curvature $k_x$ [1/m]  (closing ←  → opening)")
    ax.set_ylabel("% change vs flat card")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(
        [1000 * k for k in data.ks if k],
        [radius_label(k) for k in data.ks if k],
        fontsize=5.5,
    )
    sec.set_xlabel("R [m]", fontsize=7)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_grooving(data: SheetData, out: Path) -> Path:
    _mpl_mod, plt = _mpl()
    bars = [
        ("Exx", "$E_{xx}$"),
        ("Ezz", "$E_{zz}$"),
        ("Gxz", "$G_{xz}$"),
        ("Gxy", "$G_{xy}$"),
        ("rho_infused", "ρ"),
    ]
    f, p0 = data.flat, data.plain
    v = [f[p] / p0[p] for p, _ in bars]
    vs = [(f[p] / f["rho_infused"]) / (p0[p] / p0["rho_infused"]) for p, _ in bars[:-1]]
    fig, ax = plt.subplots(figsize=(3.6, 2.3), constrained_layout=True)
    xx = np.arange(len(bars))
    ax.bar(xx - 0.18, v, 0.36, color="#3b4cc0", label="absolute")
    ax.bar(xx[:-1] + 0.18, vs, 0.36, color="#f49a7b", label="per unit density")
    for i, a in enumerate(v):
        ax.text(i - 0.18, a, f"{a:.3g}×", ha="center", va="bottom", fontsize=6.5)
    for i, a in enumerate(vs):
        ax.text(i + 0.18, a, f"{a:.3g}×", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(1, lw=0.6, color="k")
    ax.set_xticks(xx, [t for _, t in bars])
    ax.set_ylabel("grooved ÷ plain foam")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_calibration(data: SheetData, out: Path) -> Path:
    _mpl_mod, plt = _mpl()
    cal, res = data.calibration, data.cal_result
    props = list(res.fit_model)
    m = np.array([cal.measured[p] for p in props])
    f = np.array([res.fit_model[p] for p in props])
    s0 = np.array([res.start_model[p] for p in props])
    sd = np.array([cal.sigma_rel.get(p, 0.03) for p in props])
    lab = {
        p: _TEX[p].replace("(", "{").replace(")", "}").replace("$E_{x x}", "$E_{xx}")
        for p in props
    }
    names = [
        {
            "Exx": "$E_{xx}$",
            "Eyy": "$E_{yy}$",
            "Ezz": "$E_{zz}$",
            "Gxz": "$G_{xz}$",
            "Gyz": "$G_{yz}$",
            "Gxy": "$G_{xy}$",
            "rho_infused": "ρ",
        }.get(p, lab[p])
        for p in props
    ]
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    xx = np.arange(len(props))
    a1.bar(
        xx - 0.27, s0 / m, 0.27, color="#bbbbbb", label=f"model, start {cal.start:g}"
    )
    a1.bar(xx, np.ones(len(props)), 0.27, color="#3b4cc0", label=f"{cal.label} (= 1)")
    a1.bar(
        xx + 0.27, f / m, 0.27, color="#f49a7b", label=f"model, fitted {res.value:.3g}"
    )
    a1.errorbar(
        xx, np.ones(len(props)), yerr=sd, fmt="none", ecolor="k", lw=0.7, capsize=2
    )
    for i, val in enumerate(f / m):
        a1.text(
            i + 0.27,
            val + 0.01,
            f"{100 * (val - 1):+.1f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    lo = min(0.9, float(np.min(np.r_[s0 / m, f / m])) - 0.05)
    a1.set_ylim(lo, max(1.15, float(np.max(np.r_[s0 / m, f / m])) + 0.08))
    a1.set_xticks(xx, names)
    a1.set_ylabel(f"model ÷ {cal.label}")
    a1.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=3)
    g = np.array([s[0] for s in res.scan])
    ssr = np.array([s[1] for s in res.scan])
    a2.plot(g, ssr, "o-", ms=3, lw=0.8, color="#555555", label="FE solves")
    a2.axvspan(*res.ci95, color="#f49a7b", alpha=0.3, lw=0)
    a2.axvline(res.value, color="#b40426", lw=1, label=f"fit {res.value:.3g}")
    for b in cal.bounds:
        a2.axvline(b, color="0.5", lw=0.6, ls=":")
    a2.set_yscale("log")
    a2.set_xlabel(f"{cal.param} (bounds dotted)")
    a2.set_ylabel("weighted SSR")
    a2.legend(frameon=False, fontsize=5.5)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_iso_halo(case_path: Path, cell_size: float | None, out: Path) -> Path:
    """RVE isometric with the halo band, legend and a true-scale wall inset."""
    _mpl_mod, plt = _mpl()
    from matplotlib.patches import Patch, Rectangle

    from b3_core.viz.model import CoreModel
    from b3_core.viz.scene import CoreScene

    model = CoreModel.from_json(str(case_path))
    halo_col, resin_col = "#f6c26b", "#2a9d8f"
    scene = CoreScene(model).add_phases()
    halo = np.asarray(
        model.mesh.cell_data.get("halo", np.zeros(model.mesh.n_cells)), bool
    )
    if halo.any():
        scene.plotter.add_mesh(
            model.mesh.extract_cells(np.where(halo)[0]),
            color=halo_col,
            opacity=0.6,
            ambient=0.35,
            smooth_shading=False,
        )
    raw = out.with_name(out.stem + "_3d.png")
    scene.add_axes().isometric()
    scene.screenshot(raw)
    scene.close()
    img = plt.imread(raw)
    fig = plt.figure(figsize=(5.2, 4.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    handles = [
        Patch(fc="#d9d9d9", ec="#888", label="foam"),
        Patch(fc=resin_col, label="resin in grooves"),
    ]
    if cell_size and halo.any():
        cs = float(cell_size)
        hw = 0.5 * float(min(r[3] for r in model.inp["xgr"] + model.inp["ygr"]))
        ia = fig.add_axes([0.68, 0.60, 0.30, 0.30])
        ia.add_patch(Rectangle((-hw, 0), 2 * hw, 3, color=resin_col))
        n = 40
        for s in (-1, 1):
            for i in range(n):
                a = i / n
                xa = hw + a * cs
                ia.add_patch(
                    Rectangle(
                        (xa if s > 0 else -xa - cs / n, 0),
                        cs / n,
                        3,
                        color=halo_col,
                        alpha=0.95 * (1 - a) + 0.05,
                        lw=0,
                    )
                )
        span = hw + 2.0
        ia.set_xlim(-span, span)
        ia.set_ylim(0, 3)
        ia.set_facecolor("#e4e4e4")
        ia.annotate(
            "",
            xy=(hw, 2.2),
            xytext=(hw + cs, 2.2),
            arrowprops={"arrowstyle": "<->", "lw": 0.8},
        )
        ia.text(hw + cs / 2 + 0.3, 2.45, f"{cs:.2f} mm", ha="center", fontsize=6)
        ia.set_yticks([])
        ia.tick_params(labelsize=5.5)
        ia.set_xlabel("across groove wall [mm]", fontsize=5.5, labelpad=1)
        ia.set_title("zoom: groove wall, true scale", fontsize=6.5, pad=2)
        handles.append(
            Patch(fc=halo_col, label=f"halo: resin-seeped foam ({cs:.2f} mm)")
        )
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.5)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig_cuts(case_path: Path, case: dict, out: Path) -> Path:
    _mpl_mod, plt = _mpl()
    from b3_core.viz import slices
    from b3_core.viz.geometry import between_grooves
    from b3_core.viz.model import CoreModel

    at = {
        0: between_grooves(case.get("xgr"), float(case["dx"])),
        1: between_grooves(case.get("ygr"), float(case["dy"])),
    }
    fig, _ = slices.plot_orthogonal_cuts(CoreModel.from_json(str(case_path)), at=at)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Typst
# --------------------------------------------------------------------------- #
def _esc(text: str) -> str:
    for a, b in (
        ("\\", "\\\\"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("[", "\\["),
        ("]", "\\]"),
        ("_", "\\_"),
        ("@", "\\@"),
        ("<", "\\<"),
        ("*", "\\*"),
    ):
        text = text.replace(a, b)
    return text


def _cells(rows: list[Sequence[str]], raw_first: bool = False) -> str:
    out = []
    for row in rows:
        for i, c in enumerate(row):
            out.append(
                f"[{c}]"
                if (raw_first and i == 0) or c.startswith("*")
                else f"[{_esc(c)}]"
            )
    return ", ".join(out)


def _mat_text(m: dict) -> str:
    e = m.get("E", m.get("Ex"))
    nu = m.get("nu", m.get("nuxy"))
    return f"E {fmt_modulus(e)}, ν {nu:.2f}, ρ {m['rho']:.0f} kg/m³"


def build_typst(
    data: SheetData, *, title: str, subtitle: str, figs: dict[str, str]
) -> str:
    c, f, p0 = data.case, data.flat, data.plain
    ratio = lambda p: f[p] / p0[p]  # noqa: E731
    dens = lambda p: ratio(p) / ratio("rho_infused")  # noqa: E731
    key = [
        (
            "$E_(x x)$ / $E_(y y)$ [MPa]",
            f"{fmt_mpa(f['Exx'])} / {fmt_mpa(f['Eyy'])}",
            fmt_mpa(p0["Exx"]),
            f"{ratio('Exx'):.2f}×",
            f"{dens('Exx'):.2f}×",
        ),
        (
            "$E_(z z)$ [MPa]",
            fmt_mpa(f["Ezz"]),
            fmt_mpa(p0["Ezz"]),
            f"{ratio('Ezz'):.2f}×",
            f"{dens('Ezz'):.2f}×",
        ),
        (
            "$G_(x z)$ / $G_(y z)$ [MPa]",
            f"{fmt_mpa(f['Gxz'])} / {fmt_mpa(f['Gyz'])}",
            fmt_mpa(p0["Gxz"]),
            f"{ratio('Gxz'):.2f}×",
            f"{dens('Gxz'):.2f}×",
        ),
        (
            "$G_(x y)$ [MPa]",
            fmt_mpa(f["Gxy"]),
            fmt_mpa(p0["Gxy"]),
            f"{ratio('Gxy'):.2f}×",
            f"{dens('Gxy'):.2f}×",
        ),
        (
            "$nu_(x y)$ / $nu_(x z)$",
            f"{f.get('nuxy', float('nan')):.3f} / {f.get('nuxz', float('nan')):.3f}",
            f"{p0.get('nuxy', float('nan')):.3f}",
            "—",
            "—",
        ),
        (
            "infused ρ [kg/m³]",
            f"{f['rho_infused']:.1f}",
            f"{p0['rho_infused']:.1f}",
            f"{ratio('rho_infused'):.2f}×",
            "—",
        ),
        (
            "resin Vf (groove + halo)",
            f"{100 * f.get('resin_vf', 0):.1f} + {100 * f.get('halo_vf', 0):.1f} %",
            "0",
            "—",
            "—",
        ),
    ]
    key_rows = ", ".join(
        f"[{a}], [{_esc(b)}], [{_esc(cc)}], [{_esc(d)}], [{_esc(e)}]"
        for a, b, cc, d, e in key
    )
    close = [
        f"{ax} {abs(1 / k) / 1000:.2f} m ({'k<0' if k < 0 else 'k>0'})"
        for ax, k in data.closure.items()
        if k is not None
    ]
    inputs = [
        ("Core", _mat_text(c["core"])),
        ("Resin", _mat_text(c["resin"])),
        ("RVE", f"{c['dx']:g} × {c['dy']:g} × {c['thickness']:g} mm, periodic"),
    ]
    for key_, ax in (("xgr", "x"), ("ygr", "y")):
        for row in c.get(key_) or []:
            off, pitch, depth, width = row[:4]
            inputs.append(
                (
                    f"{ax}-grooves",
                    f"{width:g} mm wide, {abs(depth):g} mm deep, pitch {pitch:g}, "
                    f"offset {off:g}, mouth {'z = t' if depth < 0 else 'z = 0'}",
                )
            )
    halo = c["core"].get("cell_size")
    if halo:
        src = "fitted, p. 3" if data.cal_result else "input"
        inputs.append(("Resin halo", f"saw-cut reach {halo:.3g} mm ({src})"))
    inputs.append(
        ("Curvature", f"R = 1/(1000 k) m; grooves close at {', '.join(close) or '—'}")
    )
    inputs.append(
        (
            "Assumptions",
            "linear elastic, perfect bonding, no perforation, resin fills grooves",
        )
    )
    prov_rows = list(data.provenance.items())

    cont = max(abs(v) for v in data.continuity_pct.values())
    warn_lines = []
    if data.clamped_points:
        warn_lines.append(
            f"{data.clamped_points} sweep point(s) are past groove closure (hatched): "
            "the kerf is clamped at minimum width, so those values are bounds."
        )
    if cont > CONTINUITY_TOL_PCT:
        warn_lines.append(f"k→0 curved solve differs from flat by {cont:.2f} %.")
    if data.cal_result and data.cal_result.at_bound:
        warn_lines.append("Calibrated parameter sits on a bound.")
    warn = "".join(
        f'#text(7pt, fill: rgb("#b40426"))[⚠ {_esc(w)}] \\\n' for w in warn_lines
    )

    def tbl(cols: str, header: Sequence[str], rows: str) -> str:
        head = ", ".join(f"[*{h}*]" for h in header)
        return f"#table(columns: {cols}, {head}, {rows})"

    page1 = f"""
#text(15pt, weight: "bold")[{_esc(title)}] \\
#text(8pt, fill: rgb("#b40426"))[{_esc(subtitle)}]
#line(length: 100%, stroke: 0.6pt)
#grid(columns: (1fr, 1fr), gutter: 10pt,
[== Key numbers (flat, infused)
{tbl("(1.5fr, 1.1fr, 0.9fr, 0.7fr, 0.8fr)", ["Property", "Grooved", "Plain foam", "Ratio", "Ratio / ρ"], key_rows)}
== Inputs & assumptions
#table(columns: (auto, 1fr), {_cells(inputs)})
== Provenance
#table(columns: (auto, 1fr), {_cells(prov_rows)})
{warn}#image("{figs["grooving"]}", width: 92%)
#text(7pt)[*Value of grooving* vs plain foam, absolute and per unit density.]],
[#image("{figs["iso"]}", width: 100%)
#text(7pt)[*RVE.* Teal = resin in grooves, amber = halo; inset at true scale. z = 0 is the mould face.]
#v(4pt)
#align(center)[#image("{figs["cuts"]}", width: 88%)]
#text(7pt)[*Phase cuts* between the grooves (x = {_fmt_cut(c, "xgr", "dx")}, y = {_fmt_cut(c, "ygr", "dy")} mm) with the mesh.]])
"""
    page2 = f"""
#pagebreak()
== Curvature: what the mould does to the core
#image("{figs["heatmaps"]}", width: 100%)
#text(7pt)[*Curvature map:* % change vs the flat card over $k_x$ × $k_y$, shared symmetric coolwarm scale. Ticks are mould radius R = 1/(1000 k) m; + opens top-mouth grooves, − closes them. *$k_x$ drives $E_(y y)$ and $G_(y z)$* (it opens/closes the x-grooves, whose walls carry y-direction stress); $k_y$ drives $E_(x x)$ and $G_(x z)$. Hatched = grooves closed (clamped).]
#grid(columns: (1.1fr, 1fr), gutter: 10pt,
image("{figs["line"]}", width: 100%),
[#v(6pt)*Single-curvature cut ($k_y$ = 0).* Grey band: grooves closed, values are bounds. The k → 0 curved solve matches the flat card within {cont:.2f} %.
#v(4pt)
#table(columns: (auto, auto, auto, auto), [*R [m]*], [*$E_(y y)$*], [*$E_(z z)$*], [*$G_(y z)$*], {_cut_rows(data)})])
"""
    page3 = "\n#pagebreak()\n" + _stiffness_block(data)
    if data.cal_result:
        cal, res = data.calibration, data.cal_result
        rows = ", ".join(
            f"[{_TEX[p]}], [{_esc(_fmt_prop(p, cal.measured[p]))}], [{_esc(_fmt_prop(p, res.fit_model[p]))}], [{res.residual_pct[p]:+.1f} %]"
            for p in res.fit_model
        )
        par = [
            (
                "Free",
                f"{cal.param}, bounds {cal.bounds[0]:g}–{cal.bounds[1]:g}, start {cal.start:g}",
            ),
            (
                "Fitted",
                f"{res.value:.3g} (95 % CI {res.ci95[0]:.3g}–{res.ci95[1]:.3g})",
            ),
            ("Fit quality", f"RMS {res.rms_pct:.1f} %, χ²red {res.chi2_red:.2f}"),
            (
                "Method",
                "bounded weighted least squares on relative residuals, direct FE solves",
            ),
        ]
        note = f"#text(6.5pt, fill: gray)[{_esc(cal.note)}]" if cal.note else ""
        page3 += f"""
== Calibration to flat test ({_esc(cal.label)})
#image("{figs["calibration"]}", width: 100%)
#grid(columns: (1.25fr, 1fr), gutter: 10pt,
table(columns: (auto, 1fr, 1fr, auto), [*Property*], [*{_esc(cal.label)}*], [*Model (fit)*], [*Residual*], {rows}),
[#table(columns: (auto, 1fr), {_cells(par)})
{note}])
"""
    head = (
        '#set page(paper: "a4", margin: (x: 1.5cm, y: 1.4cm), footer: [#set text(6.5pt, fill: gray); '
        f"{_esc(data.provenance['code'])} · {_esc(data.provenance['backend'])} "
        '#h(1fr) #context counter(page).display("1/1", both: true)])\n'
        '#set text(font: "DejaVu Sans", size: 8.5pt)\n'
        "#set table(stroke: 0.4pt + luma(180), inset: 3.5pt)\n"
    )
    return head + page1 + page2 + page3


def _stiffness_block(data: SheetData) -> str:
    """Full constants of the flat card, plus the 6×6 when the solver gave it."""
    f = data.flat
    names = [
        ("Exx", "$E_(x x)$"),
        ("Eyy", "$E_(y y)$"),
        ("Ezz", "$E_(z z)$"),
        ("Gyz", "$G_(y z)$"),
        ("Gxz", "$G_(x z)$"),
        ("Gxy", "$G_(x y)$"),
    ]
    nus = [("nuxy", "$nu_(x y)$"), ("nuxz", "$nu_(x z)$"), ("nuyz", "$nu_(y z)$")]
    cells = [f"[{t}], [{fmt_mpa(f[k])} MPa]" for k, t in names if k in f]
    cells += [f"[{t}], [{f[k]:.3f}]" for k, t in nus if k in f]
    out = "== Flat card: engineering constants\n#grid(columns: (auto, 1fr), gutter: 10pt,\n"
    out += f"table(columns: (auto, auto), {', '.join(cells)}),\n"
    c6 = f.get("_C")
    if c6 is not None:
        labels = ["11", "22", "33", "23", "13", "12"]
        cmax = max(abs(v) for row in c6 for v in row)
        head = "[], " + ", ".join(f"[*{lab}*]" for lab in labels)
        body = ", ".join(
            f"[*{labels[i]}*], "
            + ", ".join(f"[{_c_entry(c6[i][j], cmax)}]" for j in range(6))
            for i in range(6)
        )
        out += (
            "[#text(7.5pt)[*Effective stiffness C [MPa]*, Voigt order 11 22 33 23 13 12]\n"
            f"#table(columns: 7, align: right, {head}, {body})])\n"
        )
    else:
        out += "[])\n"
    return out


def _c_entry(value: float, cmax: float) -> str:
    """MPa to 3 s.f.; solver round-off below 1e-6 of the largest entry shows as 0."""
    return "0" if abs(value) < 1e-6 * cmax else f"{value / 1e6:.3g}"


def _fmt_cut(c: dict, key: str, dim: str) -> str:
    from b3_core.viz.geometry import between_grooves

    return f"{between_grooves(c.get(key), float(c[dim])):.3g}"


def _fmt_prop(p: str, v: float) -> str:
    return f"{v:.1f}" if p == "rho_infused" else f"{fmt_mpa(v)} MPa"


def _cut_rows(data: SheetData) -> str:
    rows = []
    for r in sorted((r for r in data.sweep if r["ky"] == 0), key=lambda r: r["kx"]):
        pct = lambda p: 100 * (r[p] / data.flat[p] - 1)  # noqa: E731,B023
        mark = " †" if r["clamped"] else ""
        rows.append(
            f"[{radius_label(r['kx'])}{mark}], [{pct('Eyy'):+.1f} %], [{pct('Ezz'):+.1f} %], [{pct('Gyz'):+.1f} %]"
        )
    return ", ".join(rows)


def compile_typst(src: str, root: Path, pdf: Path) -> None:
    typ = root / "sheet.typ"
    typ.write_text(src, encoding="utf-8")
    if shutil.which("typst") is None:
        raise RuntimeError("typst is not installed")
    proc = subprocess.run(
        ["typst", "compile", typ.name, str(pdf.resolve())],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst compile failed ({proc.returncode}):\n{proc.stderr}")


def generate_engineering(
    case: dict | str | Path,
    out_dir: str | Path,
    *,
    title: str | None = None,
    subtitle: str = "",
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    biaxial: bool = True,
    calibration: Any = None,
    backend: str = "fenicsx",
    solve: Solver | None = None,
    compile: bool = True,
) -> Path:
    """Run the solves, draw the figures and write ``datasheet.pdf`` + ``numbers.json``."""
    if not isinstance(case, dict):
        case = json.loads(Path(case).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(
        case,
        radii_m=radii_m,
        biaxial=biaxial,
        calibration=calibration,
        solve=solve,
        backend=backend,
    )
    case_path = out / "case_used.json"
    case_path.write_text(json.dumps(data.case, indent=1), encoding="utf-8")
    figs = {
        "iso": fig_iso_halo(
            case_path, data.case["core"].get("cell_size"), out / "fig_rve_iso_halo.png"
        ).name,
        "cuts": fig_cuts(case_path, data.case, out / "fig_rve_cuts.png").name,
        "grooving": fig_grooving(data, out / "fig_grooving.png").name,
        "heatmaps": fig_heatmaps(data, out / "fig_curvature_map.png").name,
        "line": fig_line_cut(data, out / "fig_kx_cut.png").name,
    }
    if data.cal_result:
        figs["calibration"] = fig_calibration(data, out / "fig_calibration.png").name
    numbers = {
        "provenance": data.provenance,
        "flat": {k: v for k, v in data.flat.items() if not k.startswith("_")},
        "stiffness_pa": data.flat.get("_C"),
        "plain_foam": {k: v for k, v in data.plain.items() if not k.startswith("_")},
        "continuity_pct": data.continuity_pct,
        "closure_k_per_mm": data.closure,
        "sweep": data.sweep,
        "calibration": None
        if not data.cal_result
        else {
            "param": data.calibration.param,
            "bounds": data.calibration.bounds,
            "value": data.cal_result.value,
            "ci95": data.cal_result.ci95,
            "residual_pct": data.cal_result.residual_pct,
            "rms_pct": data.cal_result.rms_pct,
            "chi2_red": data.cal_result.chi2_red,
            "scan": data.cal_result.scan,
        },
    }
    (out / "numbers.json").write_text(
        json.dumps(numbers, indent=1, default=str), encoding="utf-8"
    )
    pdf = out / "datasheet.pdf"
    src = build_typst(data, title=title or "Grooved core", subtitle=subtitle, figs=figs)
    if compile:
        compile_typst(src, out, pdf)
    else:
        (out / "sheet.typ").write_text(src, encoding="utf-8")
    return pdf
