#!/usr/bin/env python3
"""Build looping GIFs from param_sweep geometry and response curves.

    uv run python examples/param_sweeps/make_gifs.py

Needs the ``[anim]`` extra (``uv sync --extra anim``) for imageio + pillow.
Uses cached ``out/`` results when available.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    IMG,
    MODULI,
    collect_sweep,
    parse_kx_tag,
    parse_thickness_tag,
)
from _viz import (
    case_from_cache,
    curvature_cases,
    pattern_cases,
    render_frame,
    thickness_cases,
)
from b3_core.viz.theme import DEFAULT_THEME

THEME = DEFAULT_THEME
GIF_FPS = 2.0
HOLD_FRAMES = 2


def _require_imageio():
    try:
        import imageio.v2 as imageio
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "GIF export needs the [anim] extra; install with `uv sync --extra anim`"
        ) from exc
    return imageio, Image


def _ping_pong(frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(frames) < 2:
        return frames
    return frames + frames[-2:0:-1]


def _hold(frames: list[np.ndarray], n: int = HOLD_FRAMES) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        out.extend([frame] * n)
    return out


def write_gif(frames: list[np.ndarray], path: Path, *, fps: float = GIF_FPS) -> None:
    imageio, Image = _require_imageio()
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / fps
    with imageio.get_writer(str(path), mode="I", duration=duration, loop=0) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def geometry_gif(
    cases: list[tuple[str, dict]],
    path: Path,
    *,
    camera: str = "xz",
    parallel: bool = True,
) -> None:
    frames = [
        render_frame(case, label, camera=camera, parallel=parallel)
        for label, case in cases
    ]
    write_gif(_hold(_ping_pong(frames)), path)


def _sorted_thickness_rows() -> list[tuple[float, dict]]:
    rows = collect_sweep("thickness_")
    pts = []
    for tag, _case, result in rows:
        t = parse_thickness_tag(tag)
        if t is not None:
            pts.append((t, result))
    return sorted(pts, key=lambda x: x[0])


def _sorted_curvature_rows() -> list[tuple[float, dict]]:
    rows = collect_sweep("kx_")
    pts = []
    for tag, _case, result in rows:
        kx = parse_kx_tag(tag)
        if kx is not None:
            pts.append((kx, result))
    return sorted(pts, key=lambda x: x[0])


def _fig_to_rgb(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    _, Image = _require_imageio()
    return np.asarray(Image.open(buf).convert("RGB"))


def response_gif(
    pts: list[tuple[float, dict]],
    *,
    x_label: str,
    title: str,
    path: Path,
) -> None:
    if len(pts) < 2:
        print(f"skip {path.name}: need cached sweep results", file=sys.stderr)
        return

    xs = [x for x, _ in pts]
    resin = [r["resin_vf"] for _, r in pts]
    frames: list[np.ndarray] = []

    with plt.rc_context(THEME.publication_rcparams()):
        for n in range(2, len(pts) + 1):
            fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
            fig.suptitle(title, fontsize=11)
            sub_x = xs[:n]
            sub_resin = resin[:n]
            for ax, key in zip(axes.ravel(), MODULI, strict=True):
                ys = [r[key] / 1e9 for _, r in pts[:n]]
                ax.plot(sub_x, ys, "-o", color=THEME.resin_color, ms=5, lw=1.8)
                ax.set_ylabel(f"{key} [GPa]")
                ax.set_xlim(min(xs) - 0.02 * (max(xs) - min(xs) + 1), max(xs))
                ax.grid(True, alpha=0.25)
                ax2 = ax.twinx()
                ax2.plot(sub_x, sub_resin, "--", color="#888888", lw=1.2, alpha=0.7)
                ax2.set_ylim(0, max(resin) * 1.15 + 1e-9)
            for ax in axes[1]:
                ax.set_xlabel(x_label)
            fig.tight_layout()
            frames.append(_fig_to_rgb(fig))

    write_gif(_hold(_ping_pong(frames), n=3), path, fps=1.5)


def gallery_montage_gif(path: Path) -> None:
    _, Image = _require_imageio()
    gallery_dir = IMG / "galleries"
    stems = [
        "gallery_uniaxial_t20",
        "gallery_uniaxial_t50",
        "gallery_curved_kx_closed",
        "gallery_curved_kx0",
        "gallery_curved_kx_open",
        "gallery_pattern_plain",
        "gallery_pattern_uniaxial",
        "gallery_pattern_crossed",
        "gallery_pattern_two_sided",
    ]
    paths = [gallery_dir / f"{stem}.png" for stem in stems]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"skip {path.name}: run render.py first ({missing[0].name} missing)", file=sys.stderr)
        return

    target_w = 900
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        nh = max(1, round(h * target_w / w))
        frames.append(np.asarray(img.resize((target_w, nh))))
    write_gif(_hold(_ping_pong(frames), n=4), path, fps=0.5)


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)

    thickness = case_from_cache("thickness_", thickness_cases)
    curvature = case_from_cache("kx_", curvature_cases)
    patterns = case_from_cache("pattern_", pattern_cases)

    geometry_gif(thickness, IMG / "thickness.gif", camera="xz", parallel=True)
    geometry_gif(curvature, IMG / "curvature.gif", camera="xz", parallel=True)
    geometry_gif(patterns, IMG / "patterns.gif", camera="iso", parallel=False)

    t_rows = _sorted_thickness_rows()
    k_rows = _sorted_curvature_rows()
    response_gif(
        t_rows,
        x_label="thickness [mm]",
        title="Moduli vs core thickness",
        path=IMG / "thickness_response.gif",
    )
    response_gif(
        k_rows,
        x_label=r"curvature $k_x$ [1/mm]",
        title="Moduli vs mold curvature",
        path=IMG / "curvature_response.gif",
    )

    gallery_montage_gif(IMG / "galleries.gif")

    for name in (
        "thickness.gif",
        "curvature.gif",
        "patterns.gif",
        "thickness_response.gif",
        "curvature_response.gif",
        "galleries.gif",
    ):
        p = IMG / name
        if p.is_file():
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())