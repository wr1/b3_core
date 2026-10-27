"""Curvature units.

Stored curvature is always 1/mm. ``k > 0`` puts the centre of curvature on the
mould side of ``z = 0``. The signed radius in metres is ``1 / (1000 k)``.
``k = 0`` is flat.
"""

from __future__ import annotations

import numpy as np

_ACCEPTED = ("1/mm", "1/m", "R-mm", "R-m")


def to_per_mm(value: float, unit: str = "1/mm") -> float:
    """Convert one curvature or radius into 1/mm. The sign is kept."""
    number = float(value)
    name = unit.strip()
    if name == "1/mm":
        return number
    if name == "1/m":
        return number / 1000.0
    if number == 0.0:
        return 0.0
    if name == "R-mm":
        return 1.0 / number
    if name == "R-m":
        return 1.0 / (1000.0 * number)
    accepted = ", ".join(_ACCEPTED)
    raise ValueError(f"unknown curvature unit {unit!r}; use one of {accepted}")


def radius_m(k_per_mm: float) -> float | None:
    """Signed radius in metres. ``None`` when the panel is flat."""
    k = float(k_per_mm)
    if k == 0.0:
        return None
    return 1.0 / (1000.0 * k)


def curvature_tick(k_per_mm: float) -> str:
    """Axis tick: ``k (R = … m)``, or ``0 (flat)``."""
    radius = radius_m(k_per_mm)
    if radius is None:
        return "0 (flat)"
    return f"{float(k_per_mm):g} (R = {radius:g} m)"


def axis_to_per_mm(text: str, unit: str = "1/mm") -> list[float]:
    """One value, or ``start:stop:count``, each sample stored as 1/mm.

    The span is sampled in the input unit, then converted. A radius range is
    therefore uniform in radius, not in curvature.
    """
    raw = text.strip()
    if not raw:
        samples = [0.0]
    elif ":" not in raw:
        samples = [float(raw)]
    else:
        start_text, stop_text, count_text = raw.split(":")
        count = int(count_text)
        if count < 1:
            raise ValueError(f"axis count must be >= 1, got {count}")
        samples = [
            float(value)
            for value in np.linspace(float(start_text), float(stop_text), count)
        ]
    return [to_per_mm(value, unit) for value in samples]


def points_to_per_mm(text: str, unit: str = "1/mm") -> list[tuple[float, float]]:
    """``"kx,ky kx,ky"`` in ``unit`` → pairs stored as 1/mm."""
    rows: list[tuple[float, float]] = []
    for token in text.split():
        if "," not in token:
            raise ValueError(f"point {token!r} must be kx,ky (for example 5e-5,0)")
        kx_text, ky_text = token.split(",", 1)
        rows.append((to_per_mm(float(kx_text), unit), to_per_mm(float(ky_text), unit)))
    if not rows:
        raise ValueError("points is empty")
    return rows
