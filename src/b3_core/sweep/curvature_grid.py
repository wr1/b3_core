"""κ × cell_size (× ky) homogenisation grid for surrogate training.

The parametric base case is the compact top-mouth RVE previously built inside
``viz.halo``. Flat rows use ``auto`` (FEniCSx when it imports, otherwise
MFEM). Curved rows use FEniCSx, or MFEM with ``allow_pair_periodicity`` when
FEniCSx is absent, so surrogate training still runs. Numpy stays the explicit
curved opt-in for a published case.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from b3_core.api import run_case
from b3_core.cache import Cache, DiskCache

_PARAM_FOAM = {
    "E1": 32e6,
    "E2": 32e6,
    "E3": 70e6,
    "G12": 19e6,
    "G13": 19e6,
    "G23": 19e6,
    "nu12": 0.3,
    "nu13": 0.3,
    "nu23": 0.3,
    "rho": 60,
}
_PARAM_RESIN = {"E": 3e9, "nu": 0.3, "rho": 1100}


def parametric_base_case() -> dict[str, Any]:
    """Compact top-mouth RVE for κ × cell_size sweeps.

    Same sign convention as curved_panel: ``depth < 0``, ``kx > 0`` opens.
    """
    return {
        "dx": 30.0,
        "dy": 12.0,
        "thickness": 20.0,
        "xgr": [[5.0, 10.0, -17.0, 2.0]],
        "ygr": [],
        "madd": [0.0],
        "core": dict(_PARAM_FOAM),
        "resin": dict(_PARAM_RESIN),
        "scoring": {"damage_cells": 1.0, "surfaces": {"face": {"enabled": False}}},
        "curvature": {"kx": 0.0, "ky": 0.0},
    }


def homogenize_halo_curvature(
    kx: float,
    cell_size: float | None,
    *,
    base: dict | None = None,
    ky: float = 0.0,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """One homogenization at ``kx``, ``ky`` and halo width ``cell_size``.

    The backend is ``auto``. A curved row (``kx`` or ``ky`` nonzero) follows
    the curved rule: FEniCSx when it imports, otherwise MFEM with
    ``allow_pair_periodicity`` so this training grid still runs where
    FEniCSx is absent. ``cell_size is None`` or ``<= 0`` is a sharp kerf.
    Face thickness comes from ``base`` (the parametric case has none).
    """
    inp = dict(base or parametric_base_case())
    core = dict(inp.get("core") or {})
    resin = dict(inp.get("resin") or _PARAM_RESIN)
    if cell_size is None or float(cell_size) <= 0.0:
        core.pop("cell_size", None)
    else:
        core["cell_size"] = float(cell_size)
    inp["core"] = core
    inp["resin"] = resin
    curvature = {"kx": float(kx), "ky": float(ky)}
    inp["curvature"] = curvature
    kwargs: dict[str, Any] = {"cache": cache}
    if float(kx) != 0.0 or float(ky) != 0.0:
        from b3_core.pipeline import fenicsx_installed

        if not fenicsx_installed():
            # Surrogate training, not a published curved datasheet. Pair
            # periodicity is the MFEM cross-check. The curved reference is
            # the tolerance table in the backends reference.
            kwargs["backend"] = "mfem"
            kwargs["allow_pair_periodicity"] = True
    record = run_case(inp, **kwargs)
    flat = record.flat()
    exx, eyy, ezz = float(flat["Exx"]), float(flat["Eyy"]), float(flat["Ezz"])
    return {
        "kx": float(kx),
        "ky": float(ky),
        "cell_size": float(cell_size or 0.0),
        "curvature": curvature,
        "Exx": exx,
        "Eyy": eyy,
        "Ezz": ezz,
        "Ex": exx,
        "Ey": eyy,
        "Ez": ezz,
        "Gxy": float(flat["Gxy"]),
        "Gxz": float(flat["Gxz"]),
        "Gyz": float(flat["Gyz"]),
        "resin_vf": float(flat["resin_vf"]),
        "halo_vf": float(flat["halo_vf"]),
        "effective_resin_vf": float(flat["effective_resin_vf"]),
        "rho_infused": float(flat["rho_infused"]),
    }


def sweep_halo_curvature_grid(
    *,
    kx_values: Sequence[float] | np.ndarray | None = None,
    ky_values: Sequence[float] | np.ndarray | None = None,
    cell_sizes: Sequence[float] | np.ndarray | None = None,
    base: dict | None = None,
    cache: Cache | None = None,
    cache_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Cartesian product of ``kx`` × ``ky`` × ``cell_size`` (0 = sharp kerf).

    ``cache_path`` is a 0.3 alias: it wraps ``DiskCache(parent/stem)`` and
    does not read the old single-file JSON cache.
    """
    if cache_path is not None:
        warnings.warn(
            "cache_path is deprecated; pass cache=DiskCache(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        if cache is None:
            path = Path(cache_path)
            cache = DiskCache(path.parent / path.stem)
    kx_list = list(
        kx_values if kx_values is not None else np.linspace(-0.010, 0.010, 9)
    )
    ky_list = list(ky_values if ky_values is not None else [0.0])
    size_list = list(
        cell_sizes if cell_sizes is not None else [0.0, 0.3, 0.6, 1.0, 1.5]
    )
    rows: list[dict[str, Any]] = []
    case = base or parametric_base_case()
    for ky in ky_list:
        for cell_size in size_list:
            for kx in kx_list:
                rows.append(
                    homogenize_halo_curvature(
                        float(kx),
                        None if float(cell_size) <= 0.0 else float(cell_size),
                        base=case,
                        ky=float(ky),
                        cache=cache,
                    )
                )
    return rows


__all__ = [
    "homogenize_halo_curvature",
    "parametric_base_case",
    "sweep_halo_curvature_grid",
]
