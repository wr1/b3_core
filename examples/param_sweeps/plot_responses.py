#!/usr/bin/env python3
"""Plot homogenised-property response curves from cached sweep results.

Reads ``out/`` (no re-solve) and writes publication matplotlib figures to ``img/``.

    uv run python examples/param_sweeps/plot_responses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    HERE,
    IMG,
    MODULI,
    PATTERNS,
    collect_sweep,
    parse_kx_tag,
    parse_pattern_tag,
    parse_thickness_tag,
)
from b3_core.viz.theme import DEFAULT_THEME

THEME = DEFAULT_THEME


def _setup_rc():
    plt.rcParams.update(THEME.publication_rcparams())


def _ensure_data(rows: list, name: str) -> None:
    if not rows:
        print(f"no cached {name} results in out/ — run sweep_{name}.py first", file=sys.stderr)
        sys.exit(1)


def plot_thickness_response(path: Path) -> None:
    rows = collect_sweep("thickness_")
    _ensure_data(rows, "thickness")

    pts = []
    for tag, _case, result in rows:
        t = parse_thickness_tag(tag)
        if t is not None:
            pts.append((t, result))
    pts.sort(key=lambda x: x[0])
    xs = [t for t, _ in pts]
    resin = [r["resin_vf"] for _, r in pts]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle("Effective moduli vs core thickness (uniaxial grooves)", fontsize=12)

    for ax, key in zip(axes.ravel(), MODULI, strict=True):
        ys = [r[key] / 1e9 for _, r in pts]
        ax.plot(xs, ys, "-o", color=THEME.resin_color, ms=5, lw=1.8, label=key)
        ax.set_ylabel(f"{key} [GPa]")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(xs, resin, "--", color="#888888", lw=1.2, alpha=0.7, label="resin_vf")
        ax2.set_ylabel("resin_vf", color="#888888", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#888888", labelsize=7)
        ax2.set_ylim(0, max(resin) * 1.15 + 1e-9)

    for ax in axes[1]:
        ax.set_xlabel("thickness [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curvature_response(path: Path) -> None:
    rows = collect_sweep("kx_")
    _ensure_data(rows, "curvature")

    pts = []
    for tag, _case, result in rows:
        kx = parse_kx_tag(tag)
        if kx is not None:
            pts.append((kx, result))
    pts.sort(key=lambda x: x[0])
    xs = [k for k, _ in pts]
    resin = [r["resin_vf"] for _, r in pts]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle("Effective moduli vs mold curvature (deep grooves)", fontsize=12)

    for ax, key in zip(axes.ravel(), MODULI, strict=True):
        ys = [r[key] / 1e9 for _, r in pts]
        ax.plot(xs, ys, "-o", color=THEME.resin_color, ms=5, lw=1.8)
        ax.axvline(0, color="#cccccc", lw=0.8, ls=":")
        ax.set_ylabel(f"{key} [GPa]")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(xs, resin, "--", color="#888888", lw=1.2, alpha=0.7)
        ax2.set_ylabel("resin_vf", color="#888888", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#888888", labelsize=7)

    for ax in axes[1]:
        ax.set_xlabel(r"curvature $k_x$ [1/mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_patterns_comparison(path: Path) -> None:
    rows = collect_sweep("pattern_")
    _ensure_data(rows, "patterns")

    pts = []
    for tag, _case, result in rows:
        name = parse_pattern_tag(tag)
        if name is not None:
            pts.append((name, result))
    order = {n: i for i, n in enumerate(PATTERNS)}
    pts.sort(key=lambda x: order.get(x[0], 99))
    names = [n for n, _ in pts]
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Effective moduli by groove topology (t = 30 mm)", fontsize=12)

    for i, key in enumerate(MODULI):
        ys = [r[key] / 1e9 for _, r in pts]
        ax.bar(x + (i - 1.5) * width, ys, width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("modulus [GPa]")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    inset = fig.add_axes([0.62, 0.55, 0.33, 0.35])
    resin = [r["resin_vf"] for _, r in pts]
    area = [r["area_increase"] for _, r in pts]
    inset.bar(x - 0.15, resin, 0.3, color=THEME.resin_color, label="resin_vf")
    inset.bar(x + 0.15, area, 0.3, color=THEME.face_color, label="area_inc")
    inset.set_xticks(x)
    inset.set_xticklabels(names, fontsize=7)
    inset.set_title("geometry metrics", fontsize=8)
    inset.legend(fontsize=7, loc="upper right")
    inset.tick_params(labelsize=7)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_summary(path: Path) -> None:
    panels = [
        ("thickness", IMG / "thickness_response.png"),
        ("curvature", IMG / "curvature_response.png"),
        ("patterns", IMG / "patterns_comparison.png"),
    ]
    for _name, p in panels:
        if not p.exists():
            print(f"missing {p} — run plot_responses first", file=sys.stderr)
            sys.exit(1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    fig.suptitle("Parametric sweep summary — grooved core homogenisation", fontsize=13)
    for ax, (title, img_path) in zip(axes, panels, strict=True):
        ax.imshow(plt.imread(img_path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    _setup_rc()
    IMG.mkdir(parents=True, exist_ok=True)

    plot_thickness_response(IMG / "thickness_response.png")
    plot_curvature_response(IMG / "curvature_response.png")
    plot_patterns_comparison(IMG / "patterns_comparison.png")
    plot_sweep_summary(IMG / "sweep_summary.png")

    for name in (
        "thickness_response.png",
        "curvature_response.png",
        "patterns_comparison.png",
        "sweep_summary.png",
    ):
        print(f"wrote {IMG / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())