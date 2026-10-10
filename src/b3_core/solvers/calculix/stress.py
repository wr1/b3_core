"""Full 6×6 stiffness from CalculiX volume-averaged stresses.

Each load case applies one macro displacement. The six cases are not pure
unit strains (the reference-node value is ``1e-3``, and the other components
follow from the periodic constraints), so the stiffness is

    C = Σ  σ_k  ⊗  e_k     solved as  C = S E^{-1}

with σ_k the volume-averaged Cauchy stress and e_k the macro strain measured
from the face displacements. Component order on input from CalculiX ``*EL PRINT`` is
``SXX, SYY, SZZ, SXY, SXZ, SYZ``; the package Voigt order is
``xx, yy, zz, yz, xz, xy``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# CCX ``*EL PRINT`` index (SXX, SYY, SZZ, SXY, SXZ, SYZ) → package Voigt.
_CCX_TO_VOIGT = (0, 1, 2, 5, 4, 3)
_CASE_OF = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}
_FLOAT = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def ccx_stress_to_voigt(components: np.ndarray) -> np.ndarray:
    """Map one CCX stress row (SXX..SZX) into package Voigt order."""
    row = np.asarray(components, dtype=float).reshape(-1)
    if row.size < 6:
        raise ValueError(f"expected 6 stress components, got {row.size}")
    out = np.zeros(6, dtype=float)
    ccx = row[:6]
    for src, dst in enumerate(_CCX_TO_VOIGT):
        out[dst] = ccx[src]
    return out


def parse_dat_stresses(text: str) -> dict[int, list[np.ndarray]]:
    """Element id → integration-point stresses in CCX component order."""
    found: dict[int, list[np.ndarray]] = {}
    for line in text.splitlines():
        nums = _FLOAT.findall(line.replace("D", "E").replace("d", "e"))
        if len(nums) < 8:
            continue
        try:
            values = [float(n) for n in nums[:8]]
        except ValueError:
            continue
        elem = int(values[0])
        # Second column is the integration point (small integer).
        if not values[1].is_integer() or values[1] < 1 or values[1] > 27:
            continue
        if elem < 1:
            continue
        found.setdefault(elem, []).append(np.asarray(values[2:8], dtype=float))
    return found


def volume_average_from_dat(dat_text: str, volumes: np.ndarray) -> np.ndarray:
    """Volume-weighted mean stress (package Voigt) from a CalculiX ``.dat``."""
    per_elem = parse_dat_stresses(dat_text)
    if not per_elem:
        raise ValueError("CalculiX .dat file has no element stress table")
    volumes = np.asarray(volumes, dtype=float).reshape(-1)
    total = np.zeros(6, dtype=float)
    weight = 0.0
    for elem, rows in per_elem.items():
        if elem - 1 >= len(volumes):
            continue
        # VTK hexes in this mesh are wound with a negative Jacobian, so
        # ``compute_cell_sizes`` reports a negative volume. The weight is the
        # absolute cell volume.
        vol = abs(float(volumes[elem - 1]))
        if vol <= 0.0:
            continue
        mean = np.mean(np.stack(rows, axis=0), axis=0)
        total += ccx_stress_to_voigt(mean) * vol
        weight += vol
    if weight <= 0.0:
        raise ValueError("element stress table did not overlap the mesh volumes")
    return total / weight


def macro_strain_from_displacement(mesh, disp: np.ndarray) -> np.ndarray:
    """Engineering macro strain from a nodal displacement field (metres)."""
    pts = np.asarray(mesh.points, dtype=float)
    disp = np.asarray(disp, dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    tol = 1e-7

    def _face(mask: np.ndarray) -> np.ndarray:
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            return np.zeros(3)
        return disp[ids].mean(axis=0)

    ddx = _face(pts[:, 0] > xmax - tol) - _face(pts[:, 0] < xmin + tol)
    ddy = _face(pts[:, 1] > ymax - tol) - _face(pts[:, 1] < ymin + tol)
    ddz = _face(pts[:, 2] > zmax - tol) - _face(pts[:, 2] < zmin + tol)
    eps = np.zeros(6, dtype=float)
    eps[0] = ddx[0] / dx
    eps[1] = ddy[1] / dy
    eps[2] = ddz[2] / dz
    eps[3] = ddz[1] / dz + ddy[2] / dy  # gamma_yz
    eps[4] = ddz[0] / dz + ddx[2] / dx  # gamma_xz
    eps[5] = ddy[0] / dy + ddx[1] / dx  # gamma_xy
    return eps


def split_stiffness_from_responses(
    strains: list[np.ndarray] | np.ndarray,
    stresses: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Published ``0.5(C+Cᵀ)`` and the matrix from ``C = σ ε^{-1}`` before that."""
    strain = np.column_stack([np.asarray(s, dtype=float).reshape(6) for s in strains])
    stress = np.column_stack([np.asarray(s, dtype=float).reshape(6) for s in stresses])
    raw = stress @ np.linalg.inv(strain)
    return 0.5 * (raw + raw.T), raw


def stiffness_from_responses(
    strains: list[np.ndarray] | np.ndarray,
    stresses: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """``C = S E^{-1}`` from six (strain, stress) pairs, then symmetrised."""
    published, _raw = split_stiffness_from_responses(strains, stresses)
    return published


def case_tag(path: str | Path) -> str:
    stem = Path(path).stem
    for tag in ("xx", "yy", "zz", "xy", "xz", "yz"):
        if stem.endswith("_" + tag):
            return tag
    raise ValueError(f"cannot read a load-case tag from {path}")


def case_column(tag: str) -> int:
    try:
        return _CASE_OF[tag]
    except KeyError as exc:
        raise ValueError(f"unknown load case {tag!r}") from exc
