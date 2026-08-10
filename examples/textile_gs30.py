#!/usr/bin/env python3
"""Textile-as-code: DIAB-style grid-scored core without writing YAML first.

uv run python examples/textile_gs30.py
uv run python examples/textile_gs30.py --dump /tmp/gs30.json
"""

from __future__ import annotations

import argparse

from b3_core import grid_scored, homogenize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dump",
        metavar="PATH",
        help="Optional: write JSON snapshot for CLI (b3_core run PATH).",
    )
    p.add_argument("--kx", type=float, default=0.0, help="Mould curvature [1/mm].")
    p.add_argument("--cell-size", type=float, default=0.6, help="Halo cell size [mm].")
    p.add_argument(
        "--no-solve",
        action="store_true",
        help="Only build/dump the case (skip homogenization).",
    )
    args = p.parse_args()

    textile = grid_scored(cell_size=args.cell_size).with_curvature(kx=args.kx)
    if args.dump:
        path = textile.to_json(args.dump)
        print(f"wrote {path}")

    if args.no_solve:
        print(textile.input.model_dump())
        return

    result = homogenize(textile, name="gs30_textile")
    m = result.material
    print(
        f"Ex={m.Ex / 1e6:.2f} MPa  Ey={m.Ey / 1e6:.2f} MPa  Ez={m.Ez / 1e6:.2f} MPa  "
        f"rho={m.rho:.1f}  resin_vf={result.resin_volume_fraction:.3f}"
    )


if __name__ == "__main__":
    main()
