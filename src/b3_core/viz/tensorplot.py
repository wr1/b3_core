"""Publication matplotlib panels for the homogenised stiffness tensor."""

from __future__ import annotations

import numpy as np

from b3_core.viz import tensor
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme

_LABELS = ("11", "22", "33", "23", "13", "12")


def _engineering_scale(matrix: np.ndarray) -> float:
    peak = float(np.max(np.abs(matrix)))
    if peak == 0.0:
        return 1.0
    return 10.0 ** np.floor(np.log10(peak))


def plot_stiffness_heatmap(C: np.ndarray, *, ax=None, theme: CoreTheme = DEFAULT_THEME):
    """Signed heatmap of the 6x6 stiffness (Pa), annotated, in GPa.

    Returns the matplotlib Axes. Creates its own figure when ``ax`` is None.
    """
    import matplotlib.pyplot as plt

    C = np.asarray(C, dtype=float)
    shown = C / 1e9  # Pa -> GPa
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    vmax = float(np.max(np.abs(shown))) or 1.0
    im = ax.imshow(shown, cmap=theme.cmap_stiffness, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6), labels=_LABELS)
    ax.set_yticks(range(6), labels=_LABELS)
    ax.set_title(r"$C_\mathrm{eff}$ [GPa]")
    for r in range(6):
        for c in range(6):
            val = shown[r, c]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(
                c, r, f"{val:.2g}", ha="center", va="center", color=color, fontsize=7
            )
    ax.figure.colorbar(im, ax=ax, shrink=0.82)
    return ax


def plot_modulus_polar(
    C: np.ndarray,
    *,
    planes=("xy", "xz", "yz"),
    axes=None,
    theme: CoreTheme = DEFAULT_THEME,
):
    """Polar plots of the directional Young's modulus E(theta) [GPa] per plane.

    Returns the list of polar Axes. Creates a 1xN figure when ``axes`` is None.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(
            1,
            len(planes),
            subplot_kw={"projection": "polar"},
            figsize=(3.0 * len(planes), 3.0),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
    for ax, plane in zip(axes, planes, strict=False):
        theta, E = tensor.polar_modulus(C, plane=plane)
        ax.plot(theta, E / 1e9, color=theme.resin_color, lw=1.6)
        ax.fill(theta, E / 1e9, color=theme.resin_color, alpha=0.18)
        ax.set_title(f"$E(\\theta)$  {plane}-plane [GPa]", fontsize=8, pad=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
    return list(axes)
