"""Short social-media explainer animation for grooved-core homogenisation.

A frame-driven timeline built on the :mod:`b3_core.viz` layer that walks through
the whole modelling approach — grooved geometry, resin infusion, FE mesh,
orthogonal slices, the **curvature sim** (groove taper + drape + effective
properties), and the emergent stiffness tensor — and encodes it to MP4 + GIF.

    render_explainer("case.json", "explainer.mp4", gif=True)

Needs the ``[anim]`` extra (imageio, imageio-ffmpeg, pillow).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

from b3_core.core.mesh import create_grooved_mesh
from b3_core.viz import geometry
from b3_core.viz._deps import ensure_headless, require_pyvista
from b3_core.viz.model import CoreModel
from b3_core.viz.theme import CoreTheme

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Periodic homogenisation load cases (backend order).
_LOAD_CASES = ("xx", "yy", "zz", "yz", "xz", "xy")

# Dark, high-contrast theme tuned for silent social-media viewing.
ANIM_THEME = CoreTheme(
    core_color="#c8d0da",
    resin_color="#27e0c8",
    face_color="#e0a458",
    edge_color="#46506a",
    cut_line="#27e0c8",
    background="#0e1116",
    core_opacity=0.13,
)

_TITLE = "Grooved-core homogenisation · b3_core"
_LOGO_PATH = Path(__file__).parent / "assets" / "b3_logo.png"


# --------------------------------------------------------------------------- #
# optional-dependency access
# --------------------------------------------------------------------------- #
def _require_anim():
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - needs the [anim] extra
        raise RuntimeError(
            "the explainer animation needs the [anim] extra; install it with "
            "`uv sync --extra anim` (imageio, imageio-ffmpeg, pillow)"
        ) from exc
    return imageio, Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False):
    from matplotlib import font_manager
    from PIL import ImageFont

    path = font_manager.findfont(
        font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"
        )
    )
    return ImageFont.truetype(path, size)


def _hex_rgba(hex_color: str, alpha: int = 255):
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _load_logo(path, target_w: int):
    """Load and width-scale a logo to an RGBA image, or None if unavailable."""
    if path is None:
        return None
    _, Image, _, _ = _require_anim()
    p = Path(path)
    if not p.is_file():
        logger.warning("logo not found at %s; skipping", p)
        return None
    img = Image.open(p).convert("RGBA")
    w, h = img.size
    return img.resize((target_w, max(1, round(h * target_w / w))))


# --------------------------------------------------------------------------- #
# maths helpers
# --------------------------------------------------------------------------- #
def ease(t: float) -> float:
    """Smoothstep easing on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _bend_points(
    points: np.ndarray, kappa: float, axis: int = 0, z_ref=None
) -> np.ndarray:
    """Wrap a flat slab onto a cylinder of radius 1/kappa (visual drape).

    Bending hinges about the mid-``axis`` line at height ``z_ref``; thickness is
    preserved and ``kappa -> 0`` is the identity.
    """
    if abs(kappa) < 1e-9:
        return points
    p = points.copy()
    R = 1.0 / kappa
    u, z = p[:, axis], p[:, 2]
    uc = 0.5 * (u.min() + u.max())
    if z_ref is None:
        z_ref = 0.5 * (z.min() + z.max())
    theta = (u - uc) / R
    r = z - z_ref + R
    p[:, axis] = uc + r * np.sin(theta)
    p[:, 2] = (z_ref - R) + r * np.cos(theta)
    return p


def _curved_grid(inp: dict, kappa: float):
    """FEA mesh at curvature ``kappa`` (interval-affine wall morph) then rolled.

    Same kinematics as the curved-panel viz: walls track ``hw(z)`` on the flat
    RVE, then material ``x`` is mapped onto a cylinder for the drape shot.
    """
    mesh = create_grooved_mesh(
        thickness=inp["thickness"],
        dx=inp["dx"],
        dy=inp["dy"],
        xcuts=inp["xgr"],
        ycuts=inp["ygr"],
        madd=tuple(inp["madd"]),
        tface=(inp.get("face") or {}).get("thickness", 0.0),
        kx=kappa,
        ky=0.0,
    )
    grid = mesh.cast_to_unstructured_grid()
    grid.cell_data["__phase"] = geometry.cell_material(mesh)
    if abs(kappa) > 1e-9:
        grid.points = _bend_points(grid.points, kappa, axis=0)
    return grid


def _curvature_stations(inp: dict, kappa_max: float, n: int = 6):
    """Precompute homogenised E_y, E_z vs curvature (one MFEM solve per station)."""
    kappas = np.linspace(0.0, kappa_max, n)
    ey, ez = [], []
    for k in kappas:
        model = CoreModel.from_dict({**inp, "curvature": {"kx": float(k)}})
        ec = model.engineering_constants
        ey.append(ec["E_y"])
        ez.append(ec["E_z"])
        logger.info(
            "curvature station kx=%.4g: E_y=%.3g E_z=%.3g", k, ec["E_y"], ec["E_z"]
        )
    return kappas, np.array(ey), np.array(ez)


# --------------------------------------------------------------------------- #
# overlay compositing (PIL)
# --------------------------------------------------------------------------- #
def _inset_plot(stations, upto: float, theme: CoreTheme, px: int):
    """Live E(kappa) plot up to fraction ``upto`` as an RGBA PIL image."""
    import matplotlib.pyplot as plt
    from PIL import Image

    kap, ey, ez = stations
    n = max(2, round(upto * len(kap)))
    fig = plt.figure(figsize=(px / 100.0, px / 100.0 * 0.72), dpi=100)
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0.0))
    ax.plot(
        kap[:n] * 1e3,
        ey[:n] / 1e9,
        "-o",
        color=theme.resin_color,
        ms=3,
        lw=1.6,
        label=r"$E_y$",
    )
    ax.plot(
        kap[:n] * 1e3,
        ez[:n] / 1e9,
        "-o",
        color=theme.face_color,
        ms=3,
        lw=1.6,
        label=r"$E_z$",
    )
    ax.set_xlabel(
        r"curvature $\kappa\,\cdot 10^{3}$  [1/mm]", color="white", fontsize=8
    )
    ax.set_ylabel(r"$E$  [GPa]", color="white", fontsize=8)
    ax.set_xlim(0, kap[-1] * 1e3 * 1.02)
    lo = min(ey.min(), ez.min()) / 1e9
    hi = max(ey.max(), ez.max()) / 1e9
    pad = 0.05 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    for spine in ax.spines.values():
        spine.set_color("#8893a8")
    ax.tick_params(colors="#cfd6dd", labelsize=7)
    leg = ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor="white")
    leg.get_frame().set_alpha(0.0)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return Image.fromarray(buf)


def _compose(raw, *, title, caption, progress, theme, inset=None, big=None, logo=None):
    _, Image, ImageDraw, _ = _require_anim()
    img = Image.fromarray(raw).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    if logo is not None:
        lw = logo.size[0]
        img.paste(logo, (W - lw - int(W * 0.025), int(H * 0.025)), logo)
    f_title = _font(max(12, W // 42))
    f_cap = _font(max(14, W // 26), bold=True)
    f_big = _font(max(12, W // 34))
    accent = _hex_rgba(theme.resin_color)

    draw.text((W * 0.03, H * 0.035), title, font=f_title, fill=(255, 255, 255, 205))
    cap_w = draw.textlength(caption, font=f_cap)
    draw.text(
        ((W - cap_w) / 2, H * 0.895), caption, font=f_cap, fill=(255, 255, 255, 240)
    )

    draw.rectangle([0, H - max(5, H // 180), W, H], fill=(255, 255, 255, 30))
    draw.rectangle([0, H - max(5, H // 180), int(W * progress), H], fill=accent)

    if big:
        y = H * 0.30
        for i, line in enumerate(big):
            fnt = f_cap if i == 0 else f_big
            draw.text((W * 0.05, y), line, font=fnt, fill=(255, 255, 255, 240))
            y += fnt.size * 1.45

    if inset is not None:
        iw, ih = inset.size
        img.paste(inset, (W - iw - int(W * 0.03), H - ih - int(H * 0.11)), inset)

    return np.asarray(img)


# --------------------------------------------------------------------------- #
# scene context + camera
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    plotter: object
    model: CoreModel
    theme: CoreTheme
    stations: tuple
    size: tuple
    kappa_max: float
    gt: float = 0.0  # global progress 0..1 (set by the timeline)
    inset: object = None
    big: list = field(default_factory=list)
    strain_plotter: object = None  # lazily-built 2x3 montage for the strain finale


def _new_frame(ctx: _Ctx):
    p = ctx.plotter
    p.clear()
    p.set_background(ctx.theme.background)
    ctx.inset = None
    ctx.big = []
    return p


def _camera(ctx: _Ctx, *, zoom=1.0, elev=16.0, az0=25.0, az_span=70.0):
    p = ctx.plotter
    p.view_isometric()
    p.camera.Azimuth(az0 + az_span * ctx.gt)
    p.camera.Elevation(elev)
    p.camera.Zoom(zoom)


def _shot(ctx: _Ctx):
    return ctx.plotter.screenshot(return_img=True)


# --------------------------------------------------------------------------- #
# scenes  (each: build actors for local time t in [0,1], return raw RGB)
# --------------------------------------------------------------------------- #
def _scene_geometry(ctx, t):
    p = _new_frame(ctx)
    ph = geometry.split_phases(ctx.model.mesh)
    core, grooves = ph["core"], ph["resin"]
    zmin, zmax = ctx.model.mesh.bounds[4], ctx.model.mesh.bounds[5]
    level = zmin + ease(t) * (zmax - zmin) * 1.08

    def _rise(m):
        if t >= 0.99 or m.n_cells == 0:
            return m
        return m.clip(normal=(0, 0, 1), origin=(0, 0, level), invert=True)

    # Translucent core envelope so the machined groove network reads through it;
    # grooves are solid slate channels (they fill with resin in the next scene).
    cshow = _rise(core)
    if cshow.n_cells:
        p.add_mesh(cshow, color=ctx.theme.core_color, opacity=0.16, smooth_shading=True)
    gshow = _rise(grooves)
    if gshow.n_cells:
        p.add_mesh(
            gshow,
            color="#6b7488",
            show_edges=True,
            edge_color=ctx.theme.resin_color,
            line_width=0.6,
        )
    _camera(ctx, zoom=1.05)
    return _shot(ctx)


def _scene_resin(ctx, t):
    p = _new_frame(ctx)
    ph = geometry.split_phases(ctx.model.mesh)
    p.add_mesh(ph["core"], color=ctx.theme.core_color, opacity=ctx.theme.core_opacity)
    resin = ph["resin"]
    if resin.n_cells:
        zmin, zmax = resin.bounds[4], resin.bounds[5]
        level = zmin + ease(t) * (zmax - zmin) * 1.06
        shown = (
            resin.clip(normal=(0, 0, 1), origin=(0, 0, level), invert=True)
            if t < 0.99
            else resin
        )
        if shown.n_cells:
            p.add_mesh(
                shown,
                color=ctx.theme.resin_color,
                show_edges=True,
                edge_color=ctx.theme.edge_color,
                line_width=0.4,
            )
    _camera(ctx, zoom=1.05)
    return _shot(ctx)


def _scene_mesh(ctx, t):
    p = _new_frame(ctx)
    ph = geometry.split_phases(ctx.model.mesh)
    p.add_mesh(ph["core"], color=ctx.theme.core_color, opacity=ctx.theme.core_opacity)
    if ph["resin"].n_cells:
        p.add_mesh(ph["resin"], color=ctx.theme.resin_color)
    p.add_mesh(
        ctx.model.mesh,
        style="wireframe",
        color=ctx.theme.resin_color,
        line_width=1.0,
        opacity=0.12 + 0.55 * ease(t),
    )
    _camera(ctx, zoom=1.05)
    return _shot(ctx)


def _scene_slices(ctx, t):
    p = _new_frame(ctx)
    view = ctx.model.mesh.copy()
    view.cell_data["__phase"] = ctx.model.material_codes
    ph = geometry.split_phases(ctx.model.mesh)
    p.add_mesh(ph["core"], color=ctx.theme.core_color, opacity=0.05)
    if ph["resin"].n_cells:
        p.add_mesh(ph["resin"], color=ctx.theme.resin_color, opacity=0.22)
    center = view.center
    for i, axis in enumerate(("z", "x", "y")):
        if t > i / 3.0:
            origin = list(center)
            sl = view.slice(normal=axis, origin=origin)
            if sl.n_cells:
                p.add_mesh(
                    sl,
                    scalars="__phase",
                    cmap=ctx.theme.phase_colors(),
                    clim=[0, 2],
                    show_scalar_bar=False,
                    show_edges=True,
                    edge_color=ctx.theme.edge_color,
                    line_width=0.25,
                )
    _camera(ctx, zoom=1.05)
    return _shot(ctx)


def _scene_curvature(ctx, t):
    p = _new_frame(ctx)
    k = ctx.kappa_max * ease(t)
    grid = _curved_grid(ctx.model.inp, k)
    core = grid.threshold([-0.5, 0.5], scalars="__phase")
    resin = grid.threshold([0.5, 1.5], scalars="__phase")
    if core.n_cells:
        p.add_mesh(core, color=ctx.theme.core_color, opacity=ctx.theme.core_opacity)
    if resin.n_cells:
        p.add_mesh(
            resin,
            color=ctx.theme.resin_color,
            show_edges=True,
            edge_color=ctx.theme.edge_color,
            line_width=0.4,
        )
    _camera(ctx, zoom=1.0, az0=20.0, az_span=110.0)
    ctx.inset = _inset_plot(
        ctx.stations, ease(t), ctx.theme, px=int(ctx.size[0] * 0.30)
    )
    R = 1.0 / k if k > 1e-9 else float("inf")
    r_txt = "flat" if R == float("inf") else f"R = {R:.0f} mm"
    ctx.big = [
        "curvature sim",
        f"kx = {k * 1e3:.1f}e-3 /mm    {r_txt}",
        "wall morph hw(z) · roll onto arc",
    ]
    return _shot(ctx)


def _scene_strains(ctx, t):
    """2x3 montage of the six periodic load cases, deformed (translucent)."""
    pv = require_pyvista()
    if ctx.strain_plotter is None:
        ensure_headless()
        ctx.strain_plotter = pv.Plotter(
            shape=(2, 3), off_screen=True, window_size=list(ctx.size), border=False
        )
    p = ctx.strain_plotter
    p.clear()
    grid = ctx.model.mesh.cast_to_unstructured_grid()
    warp = 0.35 * ease(min(1.0, t / 0.6))  # deform in over the first 60%, then hold
    fsz = max(12, ctx.size[0] // 64)
    for i, lc in enumerate(_LOAD_CASES):
        p.subplot(i // 3, i % 3)
        p.set_background(ctx.theme.background)
        g = grid.copy()
        g["u"] = ctx.model.displacements(lc) * 1000.0  # m -> mm
        g["umag_mm"] = np.linalg.norm(g["u"], axis=1)
        warped = g.warp_by_vector("u", factor=warp)
        core = warped.threshold(0.5, scalars="resin", invert=True)
        resin = warped.threshold(0.5, scalars="resin")
        if core.n_cells:
            p.add_mesh(core, color=ctx.theme.core_color, opacity=0.10)
        if resin.n_cells:
            p.add_mesh(
                resin,
                scalars="umag_mm",
                cmap=ctx.theme.cmap_displacement,
                opacity=0.72,
                show_scalar_bar=False,
            )
        p.add_text(f"strain {lc}", font_size=fsz, color="white")
        p.camera_position = "iso"
    ctx.big = []
    ctx.inset = None
    return p.screenshot(return_img=True)


_SCENES = [
    ("orthogonal grooves · pitch · depth · width", 2.5, _scene_geometry),
    ("vacuum resin infusion", 2.0, _scene_resin),
    ("periodic FE mesh (C3D8)", 1.5, _scene_mesh),
    ("orthogonal slices reveal the nesting", 2.5, _scene_slices),
    ("curvature — wall morph + drape + E(κ)", 4.0, _scene_curvature),
    ("periodic strain response · six unit load cases", 3.0, _scene_strains),
]


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def render_explainer(
    case,
    out: str | Path = "explainer.mp4",
    *,
    seconds: float | None = None,
    fps: int = 30,
    size: tuple[int, int] = (1080, 1080),
    gif: bool = True,
    theme: CoreTheme = ANIM_THEME,
    kappa_max: float = 0.012,
    stations: int = 6,
    logo=_LOGO_PATH,
) -> list[Path]:
    """Render the explainer to ``out`` (MP4) and, if ``gif``, a sibling GIF.

    ``seconds`` overrides the storyboard's total duration (scene lengths scale to
    fit). ``logo`` is a PNG branded top-right (None to disable). Returns the
    written paths.
    """
    imageio, Image, _, _ = _require_anim()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logo_img = _load_logo(logo, max(64, size[0] // 9))

    model = CoreModel.from_json(case)
    logger.info("precomputing %d curvature stations (MFEM)", stations)
    station_data = _curvature_stations(model.inp, kappa_max, n=stations)

    scale = (seconds / sum(s for _, s, _ in _SCENES)) if seconds else 1.0
    plan = [(cap, max(1, round(sec * scale * fps)), fn) for cap, sec, fn in _SCENES]
    total = sum(n for _, n, _ in plan)

    ensure_headless()
    pv = require_pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    ctx = _Ctx(plotter, model, theme, station_data, size, kappa_max)

    mp4_writer = imageio.get_writer(
        str(out), fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    gif_path = out.with_suffix(".gif")
    gif_writer = (
        imageio.get_writer(str(gif_path), mode="I", duration=1.0 / min(fps, 12))
        if gif
        else None
    )
    gif_step = max(1, round(fps / 12))
    gif_w = max(120, size[0] // 2)
    gif_h = max(120, size[1] // 2)

    idx = 0
    try:
        for caption, nframes, fn in plan:
            for i in range(nframes):
                t = i / (nframes - 1) if nframes > 1 else 1.0
                ctx.gt = idx / max(1, total - 1)
                raw = fn(ctx, t)
                frame = _compose(
                    raw,
                    title=_TITLE,
                    caption=caption,
                    progress=(idx + 1) / total,
                    theme=theme,
                    inset=ctx.inset,
                    big=ctx.big,
                    logo=logo_img,
                )
                mp4_writer.append_data(frame)
                if gif_writer is not None and idx % gif_step == 0:
                    small = np.asarray(Image.fromarray(frame).resize((gif_w, gif_h)))
                    gif_writer.append_data(small)
                idx += 1
    finally:
        mp4_writer.close()
        if gif_writer is not None:
            gif_writer.close()
        plotter.close()
        if ctx.strain_plotter is not None:
            ctx.strain_plotter.close()

    written = [out] + ([gif_path] if gif else [])
    logger.info("wrote %s", ", ".join(str(p) for p in written))
    return written
