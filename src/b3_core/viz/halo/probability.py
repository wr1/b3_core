"""Resin-halo probability and modulus-ratio curves."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from b3_core.core.scoring import survival
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def resin_probability_vs_distance(
    cell_size: float | dict[str, Any] | None,
    *,
    n: int = 300,
    reach_pad: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample ``P(resin) = S(d)`` from distance ``d`` [mm] to the nearest cut.

    Returns ``(d, P, reach)`` where ``reach`` is the nominal max cell size.
    """
    s_fn, reach = survival(cell_size)
    if reach <= 0.0:
        d = np.array([0.0])
        return d, np.zeros_like(d), 0.0
    d = np.linspace(0.0, reach * (1.0 + reach_pad), n)
    return d, s_fn(d), float(reach)


def effective_modulus_ratio(
    p: np.ndarray,
    *,
    e_foam: float,
    e_resin: float,
) -> np.ndarray:
    """Isotropic rule-of-mixtures: ``E_eff / E_foam`` from ``P(resin)``."""
    ratio = e_resin / e_foam
    return p * ratio + (1.0 - p)


def plot_halo_degradation(
    cell_sizes: list[float | dict[str, Any] | None],
    labels: list[str] | None = None,
    *,
    e_foam: float | None = None,
    e_resin: float | None = None,
    highlight_index: int = 0,
    modulus_label: str = "E",
    theme: CoreTheme = DEFAULT_THEME,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot ``P(resin)`` and optional effective-modulus grading vs cut distance.

    Parameters
    ----------
    cell_sizes
        Foam ``cell_size`` specs (scalar, distribution dict, or ``None``).
    labels
        Curve labels; defaults to stringified ``cell_size`` values.
    e_foam, e_resin
        When both are set, the lower panel shows ``E_eff(d) / E_foam`` for the
        first (highlighted) ``cell_size`` entry.
    """
    if labels is None:
        labels = [str(cs) for cs in cell_sizes]
    if len(labels) != len(cell_sizes):
        msg = "labels must match cell_sizes length"
        raise ValueError(msg)

    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(
            1,
            2 if e_foam and e_resin else 1,
            figsize=figsize if e_foam and e_resin else (figsize[0] * 0.55, figsize[1]),
            squeeze=False,
            layout="constrained",
        )
        ax_p = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cell_sizes)))

        max_reach = 0.0
        for cs, lab, col in zip(cell_sizes, labels, colors, strict=True):
            d, p, reach = resin_probability_vs_distance(cs)
            max_reach = max(max_reach, reach)
            ax_p.plot(d, p, lw=2.2, color=col, label=lab)
            if reach > 0:
                ax_p.axvline(reach, color=col, ls=":", lw=1.0, alpha=0.55)

        ax_p.set_xlim(0.0, max_reach * 1.08 if max_reach else 1.0)
        ax_p.set_ylim(0.0, 1.02)
        ax_p.set_xlabel("Distance from cut surface [mm]")
        ax_p.set_ylabel("P(resin)")
        ax_p.set_title("Resin presence (survival function)")
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc="upper right", fontsize=8)

        if e_foam and e_resin:
            ax_e = axes[0, 1]
            highlight = cell_sizes[highlight_index]
            d, p, reach = resin_probability_vs_distance(highlight)
            e_ratio = effective_modulus_ratio(p, e_foam=e_foam, e_resin=e_resin)
            foam_frac = 1.0 - p

            ax_e.plot(
                d,
                e_ratio,
                color=theme.resin_color,
                lw=2.4,
                label=f"{modulus_label}_eff / {modulus_label}_foam",
            )
            ax_e.plot(
                d,
                foam_frac,
                color=theme.core_color,
                lw=1.8,
                ls="--",
                label="1 − P(resin)  (intact foam fraction)",
            )
            if reach > 0:
                ax_e.axvline(reach, color="#888888", ls=":", lw=1.0, alpha=0.7)
                ax_e.axhline(1.0, color="#cccccc", lw=0.8)
            ax_e.set_xlim(0.0, reach * 1.08 if reach else 1.0)
            ax_e.set_ylim(0.0, max(e_ratio.max() * 1.05, 1.02))
            ax_e.set_xlabel("Distance from cut surface [mm]")
            ax_e.set_ylabel("Normalized property")
            ax_e.set_title(
                f"Stiffness grading ({modulus_label}: "
                f"{e_foam / 1e6:.0f} → {e_resin / 1e9:.1f} GPa mix)"
            )
            ax_e.grid(True, alpha=0.25)
            ax_e.legend(loc="upper right", fontsize=8)

        if title:
            fig.suptitle(title, fontsize=11)

    return fig, axes
