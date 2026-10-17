"""Engineering datasheet: flat card, plain-foam baseline, curvature sweep, calibration.

Every number on the page comes from FE solves made here (FEniCSx by default):

* page 1: key numbers vs plain foam (absolute and per unit density), inputs and
  assumptions, provenance (backend, commit, mesh, convergence status), RVE iso
  view with the resin-halo band and a true-scale wall inset, phase cuts placed
  between the grooves, and the value-of-grooving bars;
* page 2: curvature map (% vs the flat card, shared symmetric coolwarm scale,
  mould radius ticks R = 1/(1000 k) m, closed-groove cells hatched) and the
  single-curvature line cut with the closure band;
* page 3 (optional): exact calibration to a flat test (:mod:`b3_core.fit.exact`):
  parity plot, free parameters with bounds / nominal / fitted (/ truth),
  residuals, condition number and the FE-solve history.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
import subprocess
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROPS = ("Exx", "Eyy", "Ezz", "Gxz", "Gyz", "Gxy")
CAL_PROPS = (*PROPS, "rho_infused")
# Exact-fit knobs, one per identifiable flat target (see b3_core.fit.exact).
CAL_KNOBS = ("halo_reach", "core_E", "core_G", "resin_E")
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


_XY_SWAP = {"Exx": "Eyy", "Eyy": "Exx", "Gxz": "Gyz", "Gyz": "Gxz"}


def xy_symmetric(case: dict) -> bool:
    """True when x and y are interchangeable (same pitch, same groove rows).

    Then the (ky, kx) solve is the (kx, ky) solve with x and y swapped, so a
    biaxial sweep needs only half the solves.
    """
    rows = lambda key: [[float(v) for v in r[:4]] for r in case.get(key) or []]  # noqa: E731
    return float(case["dx"]) == float(case["dy"]) and rows("xgr") == rows("ygr")


def _swap_xy(f: dict) -> dict:
    return {_XY_SWAP.get(k, k): v for k, v in f.items() if k in (*PROPS, "rho_infused")}


def radius_label(k: float) -> str:
    return "flat" if k == 0 else f"{1 / (1000 * k):+g}"


# --------------------------------------------------------------------------- #
# solves
# --------------------------------------------------------------------------- #
def fenicsx_solver(backend: str = "fenicsx", cache: Any = None) -> Solver:
    """Flat/curved solve → flat property dict. ``cache`` (e.g. a ``DiskCache``)
    makes a long sheet build resumable."""
    from b3_core.api import run_case

    def solve(case: dict, kx: float = 0.0, ky: float = 0.0) -> dict:
        rec = run_case(
            copy.deepcopy(case),
            kx=kx or None,
            ky=ky or None,
            backend=backend,
            cache=cache,
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


def load_calibration(calibration: Any, case: dict, solve: Solver):
    """``ExactFit`` from a fit object, a saved fit, or a spec that is solved here.

    A spec holds ``measured`` (raw flat properties or averaged targets),
    ``knobs`` (default :data:`CAL_KNOBS`), optional ``bounds``, ``targets``,
    ``true``, ``label`` and ``note``. A saved fit (``ExactFit.to_dict()``) has
    ``fitted`` and is used as is.
    """
    from b3_core.fit.exact import ExactFit, exact_fit

    if isinstance(calibration, ExactFit):
        return calibration
    if isinstance(calibration, (str, Path)):
        calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
    spec = dict(calibration)
    if "fitted" in spec:
        return ExactFit.from_dict(spec)
    if "param" in spec:
        raise ValueError(
            "single-parameter calibration specs are replaced by the exact fit; "
            "give 'measured' and 'knobs' (see b3_core.fit.exact)"
        )
    return exact_fit(
        solve,
        case,
        spec["measured"],
        spec.get("knobs", CAL_KNOBS),
        targets=spec.get("targets"),
        checks=spec.get("checks"),
        bounds=spec.get("bounds"),
        true=spec.get("true"),
        label=spec.get("label", "test"),
        note=spec.get("note", ""),
        store=spec.get("store"),
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
    calibration: Any = None  # b3_core.fit.exact.ExactFit


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
    cal = None
    if calibration is not None:
        cal = load_calibration(calibration, case, solve)
        case = cal.apply(case)

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
    mirror = xy_symmetric(case)
    done: dict[tuple[float, float], dict] = {}
    for kx, ky in pairs:
        if kx == 0 and ky == 0:
            f = flat
        elif mirror and (ky, kx) in done:
            f = _swap_xy(done[(ky, kx)])
        else:
            f = solve(case, kx, ky)
        done[(kx, ky)] = f
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


_TARGET_TEX = {
    "E_in": "$E_{in}$",
    "Ezz": "$E_{zz}$",
    "G_tr": "$G_{tr}$",
    "Gxy": "$G_{xy}$",
    "rho": "ρ",
}
_TARGET_TYP = {
    "E_in": "$E_(i n)$",
    "Ezz": "$E_(z z)$",
    "G_tr": "$G_(t r)$",
    "Gxy": "$G_(x y)$",
    "rho": "infused ρ",
}


def _target_display(name: str, v: float) -> float:
    """MPa for moduli, kg/m³ for density."""
    return float(v) if name == "rho" else float(v) / 1e6


def fig_calibration(data: SheetData, out: Path) -> Path:
    """Parity (test vs model), knobs inside their bounds, residual history."""
    _mpl_mod, plt = _mpl()
    from b3_core.fit.exact import knob

    fit = data.calibration
    names = list(fit.measured)
    cols = plt.get_cmap("tab10")(np.arange(10))
    fig, (a1, a2, a3) = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.75),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.35, 0.8]},
    )
    # (a) parity
    m = np.array([_target_display(n, fit.measured[n]) for n in names])
    f = np.array([_target_display(n, fit.model[n]) for n in names])
    s0 = np.array([_target_display(n, fit.start_model.get(n, np.nan)) for n in names])
    lo, hi = 0.7 * np.nanmin(np.r_[m, f, s0]), 1.4 * np.nanmax(np.r_[m, f, s0])
    xx = np.geomspace(lo, hi, 50)
    a1.fill_between(xx, 0.97 * xx, 1.03 * xx, color="0.9", lw=0, label="±3 %")
    a1.plot(xx, xx, color="0.4", lw=0.7)
    for i, n in enumerate(names):
        a1.plot(m[i], s0[i], "o", mfc="none", mec="0.55", ms=4.5, mew=0.8)
        a1.plot([m[i], m[i]], [s0[i], f[i]], color="0.75", lw=0.6, zorder=1)
        a1.plot(m[i], f[i], "o", color=cols[i], ms=5, label=_TARGET_TEX.get(n, n))
    for j, n in enumerate(fit.check_measured):
        cm = _target_display(n, fit.check_measured[n])
        cf = _target_display(n, fit.check_model[n])
        a1.plot(
            cm,
            cf,
            "*",
            mfc="white",
            mec=cols[(len(names) + j) % 10],
            ms=8,
            mew=1.0,
            label=f"{_TARGET_TEX.get(n, n)} (held out)",
        )
    a1.plot([], [], "o", mfc="none", mec="0.55", ms=4.5, label="nominal")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(lo, hi)
    a1.set_ylim(lo, hi)
    a1.set_xlabel(f"{fit.label} [MPa; ρ kg/m³]")
    a1.set_ylabel("model [MPa; ρ kg/m³]")
    a1.set_title("parity", fontsize=7.5)
    a1.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1)
    # (b) knobs within bounds
    noisy = (fit.extra or {}).get("noisy")
    for i, name in enumerate(fit.knobs):
        k = knob(
            name,
            dict(zip(fit.knobs, zip(fit.lower, fit.upper, strict=True), strict=True)),
        )
        y = len(fit.knobs) - 1 - i
        pos = lambda v: (float(v) - k.lower) / (k.upper - k.lower)  # noqa: E731,B023
        a2.plot([0, 1], [y, y], color="0.85", lw=5, solid_capstyle="butt", zorder=0)
        a2.plot(pos(fit.start[i]), y, "o", mfc="none", mec="0.5", ms=5, mew=0.8)
        if noisy:
            a2.plot(pos(noisy["fitted"][i]), y, "D", color="#f4a261", ms=4.2)
        if fit.true:
            a2.plot(pos(fit.true[i]), y, "x", color="k", ms=7, mew=1.2)
        a2.plot(pos(fit.fitted[i]), y, "o", color="#b40426", ms=3.6)
        a2.text(
            -0.03, y, f"{k.lower / k.scale:.3g}", ha="right", va="center", fontsize=5.5
        )
        a2.text(
            1.03,
            y,
            f"{k.upper / k.scale:.3g} {k.unit}",
            ha="left",
            va="center",
            fontsize=5.5,
        )
        label = k.label + (" (resin seepage)" if name == "halo_reach" else "")
        a2.text(0.5, y + 0.32, label, ha="center", va="bottom", fontsize=6)
    handles = [
        plt.Line2D([], [], ls="", marker="o", mfc="none", mec="0.5", label="nominal"),
        plt.Line2D([], [], ls="", marker="o", color="#b40426", ms=3.6, label="fitted"),
    ]
    if fit.true:
        handles.append(
            plt.Line2D([], [], ls="", marker="x", color="k", label="hidden truth")
        )
    if noisy:
        handles.append(
            plt.Line2D(
                [],
                [],
                ls="",
                marker="D",
                color="#f4a261",
                ms=4,
                label=noisy.get("label", "noisy"),
            )
        )
    a2.legend(
        handles=handles,
        frameon=False,
        fontsize=5.5,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.2,
        columnspacing=0.8,
    )
    a2.set_xlim(-0.4, 1.45)
    a2.set_ylim(-0.6, len(fit.knobs) - 0.2)
    a2.axis("off")
    a2.set_title("free parameters inside their bounds", fontsize=7.5)
    # (c) residual history
    stages = ["nominal"] + [h["stage"] for h in fit.history]
    start_res = (
        max(abs(100 * (fit.start_model[n] / fit.measured[n] - 1)) for n in names)
        if fit.start_model
        else np.nan
    )
    vals = [start_res] + [h["max_abs_pct"] for h in fit.history]
    a3.semilogy(
        range(len(vals)), np.maximum(vals, 1e-6), "o-", color="#3b4cc0", ms=3.5, lw=0.9
    )
    a3.axhline(0.1, color="#b40426", lw=0.7, ls="--")
    a3.text(len(vals) - 1, 0.13, "0.1 %", color="#b40426", fontsize=5.5, ha="right")
    for i, v in enumerate(vals):
        a3.text(i, max(v, 1e-6) * 1.6, f"{v:.2g}", ha="center", fontsize=5.5)
    a3.set_xticks(range(len(vals)), stages, rotation=30, ha="right", fontsize=5.8)
    a3.set_ylabel("max |residual| [%]")
    a3.set_ylim(
        min(1e-3, 0.3 * min(np.maximum(vals, 1e-6))), max(30, 4 * np.nanmax(vals))
    )
    a3.set_title("FE solves to exact fit", fontsize=7.5)
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
    g = "" if "E" in m or m.get("Gxz") is None else f", G {fmt_modulus(m['Gxz'])}"
    return f"E {fmt_modulus(e)}{g}, ν {nu:.2f}, ρ {m['rho']:.0f} kg/m³"


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
        src = "fitted, p. 3" if data.calibration is not None else "input"
        inputs.append(
            (
                "Resin halo",
                f"seepage reach {halo:.3g} mm from groove walls, "
                f"{100 * data.flat.get('halo_vf', 0):.1f} % of volume ({src})",
            )
        )
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
    if data.calibration is not None and data.calibration.at_bound:
        warn_lines.append(
            f"Calibrated {', '.join(data.calibration.at_bound)} on a bound."
        )
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
    if data.calibration is not None:
        page3 += _calibration_block(data, figs)
    head = (
        '#set page(paper: "a4", margin: (x: 1.5cm, y: 1.4cm), footer: [#set text(6.5pt, fill: gray); '
        f"{_esc(data.provenance['code'])} · {_esc(data.provenance['backend'])} "
        '#h(1fr) #context counter(page).display("1/1", both: true)])\n'
        '#set text(font: "DejaVu Sans", size: 8.5pt)\n'
        "#set table(stroke: 0.4pt + luma(180), inset: 3.5pt)\n"
    )
    return head + page1 + page2 + page3


def _calibration_block(data: SheetData, figs: dict[str, str]) -> str:
    """Exact multi-parameter fit: figure, parameter table, residuals, method."""
    from b3_core.fit.exact import knob

    fit = data.calibration
    noisy = (fit.extra or {}).get("noisy")
    bounds = dict(zip(fit.knobs, zip(fit.lower, fit.upper, strict=True), strict=True))
    head = ["Parameter", "Bounds", "Nominal", "Fitted"]
    if fit.true:
        head.append("Truth")
    if noisy:
        head.append(noisy.get("label", "noisy"))
    prow = []
    for i, name in enumerate(fit.knobs):
        k = knob(name, bounds)
        v = lambda x: f"{x / k.scale:.4g}"  # noqa: E731,B023
        row = [
            f"{k.label} [{k.unit}]",
            f"{v(k.lower)}–{v(k.upper)}",
            v(fit.start[i]),
            f"*{v(fit.fitted[i])}*",
        ]
        if fit.true:
            row.append(v(fit.true[i]))
        if noisy:
            row.append(v(noisy["fitted"][i]))
        prow.append(row)
    ptab = (
        f"#table(columns: {len(head)}, align: (left, {', '.join(['right'] * (len(head) - 1))}), "
        + ", ".join(f"[*{_esc(h)}*]" for h in head)
        + ", "
        + _cells(prow)
        + ")"
    )
    rrow = []
    for n in fit.measured:
        fmt = (lambda x: f"{x:.1f}") if n == "rho" else (lambda x: f"{x / 1e6:.4g}")
        unit = "kg/m³" if n == "rho" else "MPa"
        start = fit.start_model.get(n)
        rrow.append(
            f"[{_TARGET_TYP.get(n, _esc(n))} [{unit}]], [{fmt(fit.measured[n])}], "
            f"[{'—' if start is None else fmt(start)}], [{fmt(fit.model[n])}], "
            f"[*{fit.residual_pct[n]:+.3f} %*]"
        )
    for n in fit.check_measured:
        fmt = lambda x: f"{x / 1e6:.4g}"  # noqa: E731
        res = 100 * (fit.check_model[n] / fit.check_measured[n] - 1)
        rrow.append(
            f"[{_TARGET_TYP.get(n, _esc(n))} [MPa] (held out)], "
            f"[{fmt(fit.check_measured[n])}], [—], [{fmt(fit.check_model[n])}], "
            f"[{res:+.2f} %]"
        )
    rtab = (
        "#table(columns: 5, align: (left, right, right, right, right), "
        f"[*Target*], [*{_esc(fit.label)}*], [*Nominal*], [*Fitted*], [*Residual*], "
        + ", ".join(rrow)
        + ")"
    )
    stages = " → ".join(f"{h['stage']} {h['max_abs_pct']:.2g} %" for h in fit.history)
    info = [
        (
            "Fit",
            f"{len(fit.knobs)} parameters ↔ {len(fit.measured)} targets: square, so exact. "
            "E_in = mean(Exx, Eyy), G_tr = mean(Gxz, Gyz): equal by symmetry, so "
            "averaged. Halo reach = resin seepage depth into the foam.",
        ),
        ("Max |residual|", f"{fit.max_abs_residual_pct:.3f} % ({stages})"),
        (
            "Conditioning",
            f"cond(J) = {fit.condition:.3g} at the fit (∂ ln target / ∂ parameter, bounds → ±1)",
        ),
        (
            "Method",
            f"{fit.n_design}-point face-centred CCD of FE solves → quadratic in ln(property) "
            f"(LOO max {fit.surface_loo_pct:.2g} %) → bounded Newton on the surface → "
            "polish on FE solves (target shifted by the surface error)",
        ),
        ("FE solves", f"{fit.n_solves} for the fit"),
    ]
    if (fit.extra or {}).get("identifiability_note"):
        info.insert(3, ("Why these", fit.extra["identifiability_note"]))
    if noisy:
        info.append(
            (
                "Sensitivity",
                f"{noisy.get('label', 'noisy')}: same truth + test noise, fit still exact "
                f"(max |residual| {noisy['max_abs_pct']:.3f} %); parameters move up to "
                f"{noisy['max_shift_pct']:.0f} % of their value",
            )
        )
    note = f"#text(6.5pt, fill: gray)[{_esc(fit.note)}]" if fit.note else ""
    return f"""
== Calibration to the flat test ({_esc(fit.label)}): exact fit
#image("{figs["calibration"]}", width: 100%)
#grid(columns: (1.15fr, 1fr), gutter: 8pt,
[{ptab}
#v(2pt)
{rtab}],
[#table(columns: (auto, 1fr), {_cells(info)})
{note}])
"""


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
    if data.calibration is not None:
        figs["calibration"] = fig_calibration(data, out / "fig_calibration.png").name
    numbers = {
        "provenance": data.provenance,
        "flat": {k: v for k, v in data.flat.items() if not k.startswith("_")},
        "stiffness_pa": data.flat.get("_C"),
        "plain_foam": {k: v for k, v in data.plain.items() if not k.startswith("_")},
        "continuity_pct": data.continuity_pct,
        "closure_k_per_mm": data.closure,
        "sweep": data.sweep,
        "calibration": None if data.calibration is None else data.calibration.to_dict(),
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
