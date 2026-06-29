#!/usr/bin/env python3
"""Intuitive visualizations of the grid-scored resin halo.

Shows the three physical zones (neat kerf, opened-cell halo, intact foam),
the ``P(resin)`` field on a side cut and in 3D, plus degradation curves.

    uv run python examples/halo_degradation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b3_core.viz.halo import (
    plot_halo_cross_section_strip,
    plot_halo_degradation,
    plot_halo_intuitive_board,
    plot_halo_sharp_vs_scored,
    plot_halo_side_cut,
    render_halo_3d_png,
)

HERE = Path(__file__).parent
OUT = HERE / "img"
CASE = json.loads((HERE / "diab_gs30_scored.json").read_text())
SHARP = json.loads((HERE / "diab_gs30.json").read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    core = CASE["core"]
    resin = CASE["resin"]
    e3_foam = float(core["E3"])
    e_resin = float(resin["E"])

    fig, _ = plot_halo_degradation(
        [0.3, 0.6, {"mean": 0.25, "std": 0.08, "dist": "lognormal"}],
        labels=[
            "uniform cell_size = 0.3 mm",
            "uniform cell_size = 0.6 mm (DIAB H60 scale)",
            "lognormal mean=0.25 mm, σ=0.08 mm",
        ],
        e_foam=e3_foam,
        e_resin=e_resin,
        highlight_index=1,
        modulus_label="E₃",
        title="Grid-scored foam: resin halo degradation from cut surface",
    )
    curve_path = OUT / "halo_degradation.png"
    fig.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig2, _ = plot_halo_cross_section_strip(CASE)
    strip_path = OUT / "halo_strip_diab_gs30.png"
    fig2.savefig(strip_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, _ = plot_halo_side_cut(CASE)
    side_path = OUT / "halo_side_cut.png"
    fig3.savefig(side_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    fig4 = plot_halo_intuitive_board(CASE)
    board_path = OUT / "halo_intuitive_board.png"
    fig4.savefig(board_path, dpi=200, bbox_inches="tight")
    plt.close(fig4)

    render_halo_3d_png(CASE, OUT / "halo_3d.png")

    fig5, _ = plot_halo_sharp_vs_scored(SHARP, CASE)
    compare_path = OUT / "halo_sharp_vs_scored.png"
    fig5.savefig(compare_path, dpi=200, bbox_inches="tight")
    plt.close(fig5)

    for p in (
        curve_path, strip_path, side_path, board_path,
        OUT / "halo_3d.png", compare_path,
    ):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()