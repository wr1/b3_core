"""One-page Typst datasheet for a grooved-core homogenisation run.

Mirrors the layout of the b3_tex material datasheet: a title banner, three
header tables (RVE/geometry, materials, analysis), two figure panels (groove
cross-section cuts with the mesh, and a 3D isometric of the internal structure),
and a footer with the engineering constants and the 6x6 effective stiffness.

The composition is Typst (vector tables + embedded matplotlib PNG panels); the
effective stiffness and per-load-case data come from the MFEM backend
(`return_details=True`), the only backend that returns the full tensor.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from b3_core import __version__
from b3_core.core.analysis import geom_analysis
from b3_core.core.cprop import CpropInput
from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import mfem_backend

# Headless: the figure panels never need an interactive display.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_FIG_DPI = 150

# Per-cell material code and its drawing colour.
_CORE, _RESIN, _FACE = 0, 1, 2
_MAT_COLORS = {_CORE: "#d9d9d9", _RESIN: "#2ca7a0", _FACE: "#d8b274"}
_CUT_LINE = "#39d0ff"


@dataclass
class DatasheetSpec:
    """Everything the Typst page needs, assembled by `collect_spec`."""

    title: str
    config_path: str
    version: str
    rve_rows: list[tuple[str, str]]
    material_rows: list[tuple[str, str]]
    analysis_rows: list[tuple[str, str]]
    engineering_constants: dict[str, float] | None = None
    c_eff_gpa: np.ndarray | None = None
    mesh_n_cells: int | None = None
    figure_cuts: Path | None = None
    figure_iso: Path | None = None
    figure_col_fracs: tuple[float, float] = (0.5, 0.5)


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #
def _cell_material(mesh) -> np.ndarray:
    """Per-cell material code: 0 core, 1 resin, 2 face."""
    resin = np.asarray(mesh.cell_data["resin"]).astype(bool)
    face = np.asarray(mesh.cell_data["face"]).astype(bool)
    mat = np.zeros(mesh.n_cells, dtype=int)
    mat[resin] = _RESIN
    mat[face] = _FACE
    return mat


def _axis_vectors(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted unique grid-line coordinates along x, y, z (the rectilinear lines)."""
    return np.unique(mesh.x), np.unique(mesh.y), np.unique(mesh.z)


def _best_slab(mesh, mat: np.ndarray, axis: int, centers: np.ndarray) -> float:
    """Cell-centre coordinate of the slab along `axis` richest in resin.

    Picking the most resin-filled slab guarantees the cut actually intersects
    grooves; falls back to the median plane when there is no resin at all.
    """
    coords = np.round(centers[:, axis], 6)
    uniq = np.unique(coords)
    resin = mat == _RESIN
    counts = np.array([resin[coords == c].sum() for c in uniq])
    if counts.max() == 0:
        return float(uniq[len(uniq) // 2])
    return float(uniq[int(counts.argmax())])


def _sample_plane(mesh, mat, u_axis, v_axis, fixed_axis, coord, u_vals, v_vals):
    """Material code on a plane, as a float grid (NaN outside the RVE)."""
    uu, vv = np.meshgrid(u_vals, v_vals)
    pts = np.zeros((uu.size, 3))
    pts[:, u_axis] = uu.ravel()
    pts[:, v_axis] = vv.ravel()
    pts[:, fixed_axis] = coord
    cids = np.asarray(mesh.find_containing_cell(pts.astype(float)))
    out = np.full(uu.size, np.nan)
    inside = cids >= 0
    out[inside] = mat[cids[inside]]
    return out.reshape(uu.shape)


def _mat_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap([_MAT_COLORS[_CORE], _MAT_COLORS[_RESIN], _MAT_COLORS[_FACE]])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def _lin(lo, hi, length, ref, px):
    """Sample coordinates spanning [lo, hi], density proportional to its length."""
    n = int(np.clip(px * (hi - lo) / ref, 40, px))
    eps = 1e-4 * max(hi - lo, ref)
    return np.linspace(lo + eps, hi - eps, n)


def _draw_panel(ax, grid, extent, cmap, norm, lines_u, lines_v, title):
    ax.imshow(
        grid, origin="lower", extent=extent, cmap=cmap, norm=norm,
        aspect="equal", interpolation="nearest",
    )
    u0, u1, v0, v1 = extent
    for u in lines_u:
        ax.axvline(u, color="0.35", lw=0.15, alpha=0.45)
    for v in lines_v:
        ax.axhline(v, color="0.35", lw=0.15, alpha=0.45)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def render_section_cuts(mesh, out_png: Path, *, px: int = 240) -> float:
    """Plan + two side cross-sections coloured by material, with the mesh overlaid.

    Returns the figure aspect ratio (width / height) for Typst column sizing.
    """
    mat = _cell_material(mesh)
    centers = mesh.cell_centers().points
    xv, yv, zv = _axis_vectors(mesh)
    x0, x1, y0, y1, z0c, z1c = xv[0], xv[-1], yv[0], yv[-1], zv[0], zv[-1]
    Lx, Ly, Lz = x1 - x0, y1 - y0, z1c - z0c
    ref = max(Lx, Ly, Lz)

    zc = _best_slab(mesh, mat, 2, centers)  # plan cut height
    yc = _best_slab(mesh, mat, 1, centers)  # top (xz) cut
    xc = _best_slab(mesh, mat, 0, centers)  # side (yz) cut

    ux, uy, uz = (
        _lin(x0, x1, Lx, ref, px),
        _lin(y0, y1, Ly, ref, px),
        _lin(z0c, z1c, Lz, ref, px),
    )
    # plan z=zc : x horizontal, y vertical
    plan = _sample_plane(mesh, mat, 0, 1, 2, zc, ux, uy)
    # top y=yc : x horizontal, z vertical
    top = _sample_plane(mesh, mat, 0, 2, 1, yc, ux, uz)
    # side x=xc : z horizontal, y vertical
    side = _sample_plane(mesh, mat, 2, 1, 0, xc, uz, uy)

    cmap, norm = _mat_cmap()
    fig = plt.figure(figsize=((Lz + Lx) / ref * 4.2, (Lz + Ly) / ref * 4.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[Lz, Lx], height_ratios=[Lz, Ly], hspace=0.28, wspace=0.22
    )
    fig.add_subplot(gs[0, 0]).axis("off")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_plan = fig.add_subplot(gs[1, 1])

    _draw_panel(ax_top, top, (x0, x1, z0c, z1c), cmap, norm, xv, zv, f"top  y={yc:.3g}")
    _draw_panel(ax_side, side, (z0c, z1c, y0, y1), cmap, norm, zv, yv, f"side  x={xc:.3g}")
    _draw_panel(ax_plan, plan, (x0, x1, y0, y1), cmap, norm, xv, yv, f"plan  z={zc:.3g}")

    # cut-line annotations on the plan view
    ax_plan.axvline(xc, color=_CUT_LINE, lw=0.9, ls="--")
    ax_plan.axhline(yc, color=_CUT_LINE, lw=0.9, ls="--")
    ax_plan.plot([xc], [yc], "+", color=_CUT_LINE, ms=8, mew=1.4)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=_MAT_COLORS[_CORE], edgecolor="0.4", label="core"),
        Patch(facecolor=_MAT_COLORS[_RESIN], edgecolor="0.4", label="resin"),
    ]
    if (mat == _FACE).any():
        handles.append(Patch(facecolor=_MAT_COLORS[_FACE], edgecolor="0.4", label="face"))
    fig.legend(handles=handles, loc="lower left", fontsize=6, frameon=False, ncol=3)

    fig.savefig(out_png, dpi=_FIG_DPI, bbox_inches="tight")
    w, h = fig.get_size_inches()
    plt.close(fig)
    return float(w / h)


def render_isometric(mesh, out_png: Path) -> bool:
    """3D isometric: translucent core + solid resin grooves + face skin.

    Uses pyvista off-screen; falls back to a matplotlib scatter axonometric when
    headless GL is unavailable, so the datasheet never hard-fails.
    """
    try:
        import pyvista as pv

        try:
            pv.start_xvfb()
        except Exception:  # pragma: no cover - display already present / unsupported
            pass
        view = mesh.copy()
        view.cell_data["__mat"] = _cell_material(mesh)
        core = view.threshold([-0.5, 0.5], scalars="__mat")
        resin = view.threshold([0.5, 1.5], scalars="__mat")
        face = view.threshold([1.5, 2.5], scalars="__mat")

        p = pv.Plotter(off_screen=True, window_size=(960, 840))
        p.set_background("white")
        if core.n_cells:
            p.add_mesh(core, color=_MAT_COLORS[_CORE], opacity=0.12, show_edges=False)
        if face.n_cells:
            p.add_mesh(face, color=_MAT_COLORS[_FACE], opacity=0.55, show_edges=False)
        if resin.n_cells:
            p.add_mesh(
                resin, color=_MAT_COLORS[_RESIN], show_edges=True,
                edge_color="#15605b", line_width=0.4,
            )
        p.enable_parallel_projection()
        p.camera_position = "iso"
        p.add_axes(line_width=2, labels_off=False)
        p.screenshot(str(out_png))
        p.close()
        return True
    except Exception as exc:  # pragma: no cover - depends on GL availability
        logger.warning("pyvista isometric failed (%s); matplotlib fallback", exc)
        _render_isometric_fallback(mesh, out_png)
        return False


def _render_isometric_fallback(mesh, out_png: Path) -> None:
    centers = mesh.cell_centers().points
    mat = _cell_material(mesh)
    fig = plt.figure(figsize=(5.0, 4.6))
    ax = fig.add_subplot(projection="3d")
    for code, alpha, size in ((_FACE, 0.2, 6), (_RESIN, 0.9, 8)):
        m = mat == code
        if m.any():
            ax.scatter(
                centers[m, 0], centers[m, 1], centers[m, 2],
                c=_MAT_COLORS[code], marker="s", s=size, alpha=alpha, linewidths=0,
            )
    xv, yv, zv = _axis_vectors(mesh)
    ax.set_box_aspect((xv.ptp(), yv.ptp(), zv.ptp()))
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=6)
    fig.savefig(out_png, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# spec assembly
# --------------------------------------------------------------------------- #
def _groove_rows(prefix: str, grooves: list[list[float]]) -> list[tuple[str, str]]:
    rows = []
    for k, g in enumerate(grooves, 1):
        off, pitch, depth, width = g
        rows.append(
            (f"{prefix}-groove {k}", f"off {off:.3g}, pitch {pitch:.3g}, "
             f"depth {depth:.3g}, w {width:.3g}")
        )
    return rows


def collect_spec(inp: dict, mesh, geom: dict, details, *, name: str, config_name: str):
    """Build the three header-table row lists + the result block from a run."""
    rve_rows: list[tuple[str, str]] = [
        ("RVE size [mm]", f"{inp['dx']:.4g} × {inp['dy']:.4g} × {inp['thickness']:.4g}"),  # noqa: RUF001
        ("x-grooves", str(len(inp["xgr"]))),
        *_groove_rows("x", inp["xgr"]),
        ("y-grooves", str(len(inp["ygr"]))),
        *_groove_rows("y", inp["ygr"]),
    ]
    cur = inp.get("curvature") or {}
    if cur.get("kx") or cur.get("ky"):
        rve_rows.append(
            ("curvature [1/mm]", f"kx {cur.get('kx', 0):.4g}, ky {cur.get('ky', 0):.4g}")
        )
    face = inp.get("face") or {}
    rve_rows.append(("face thickness [mm]", f"{face.get('thickness', 0.0):.3g}"))
    rve_rows.append(("mesh refine (madd)", ", ".join(f"{m:.2g}" for m in inp["madd"])))

    def _mat(label, m):
        return (label, f"E {m['E'] / 1e9:.3g} GPa, ν {m['nu']:.3g}, ρ {m['rho']:.0f}")  # noqa: RUF001

    material_rows = [
        _mat("core", inp["core"]),
        _mat("resin", inp["resin"]),
        ("resin Vf", f"{geom['resin_vf']:.3f}"),
        ("ρ infused [kg/m³]", f"{geom['rho_infused']:.1f}"),  # noqa: RUF001
        ("groove area factor", f"{geom['area_increase']:.3f}"),
    ]
    analysis_rows = [
        ("backend", "mfem (periodic)"),
        ("element type", inp["element_type"]),
        ("mesh cells", f"{mesh.n_cells}"),
        ("BC", "periodic, 6 unit strains"),
        ("units", "geom mm · props SI"),
    ]
    p = details.properties
    eng = {
        "E_x": p["Exx"], "E_y": p["Eyy"], "E_z": p["Ezz"],
        "G_xy": p["Gxy"], "G_xz": p["Gxz"], "G_yz": p["Gyz"],
        "nu_xy": p["nuxy"], "nu_xz": p["nuxz"], "nu_yz": p["nuyz"],
    }
    return DatasheetSpec(
        title=f"Grooved core — {name}",
        config_path=config_name,
        version=f"b3_core {__version__}",
        rve_rows=rve_rows,
        material_rows=material_rows,
        analysis_rows=analysis_rows,
        engineering_constants=eng,
        c_eff_gpa=np.asarray(details.stiffness) * 1e-9,
        mesh_n_cells=mesh.n_cells,
    )


# --------------------------------------------------------------------------- #
# Typst composition (helpers adapted from b3_tex/datasheet.py)
# --------------------------------------------------------------------------- #
def _typst_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\").replace("#", "\\#").replace("$", "\\$")
        .replace("[", "\\[").replace("]", "\\]").replace("_", "\\_")
    )


def _render_table(header: str, rows: list[tuple[str, str]]) -> str:
    cells = ", ".join(
        f"[#text(size: 6.5pt)[{_typst_escape(c)}]]"
        for a, b in rows for c in (a, b)
    )
    return (
        f'#text(weight: "bold", size: 7.5pt)[{header}]\n'
        "#table(\n  columns: (auto, 1fr),\n  inset: 1.5pt,\n  stroke: 0.3pt,\n"
        f"  align: (left + top),\n  {cells}\n)\n"
    )


def _figure_image(path: Path) -> str:
    return f'#image("{path.name}", width: 100%, height: 100%, fit: "contain")\n'


def _png_aspect(path: Path) -> float:
    arr = plt.imread(path)
    h, w = arr.shape[0], arr.shape[1]
    return float(w / h)


def build_typst(spec: DatasheetSpec) -> str:
    rve = _render_table("RVE / Geometry", spec.rve_rows)
    materials = _render_table("Materials", spec.material_rows)
    analysis = _render_table("Analysis", spec.analysis_rows)

    title_block = (
        f'#align(center)[#text(size: 10pt, weight: "bold")[{_typst_escape(spec.title)}]'
        f" #text(size: 6pt)[· {_typst_escape(spec.config_path)} · "
        f"{_typst_escape(spec.version)}]]\n"
        "#v(1pt)\n#grid(\n  columns: (1fr, 1fr, 1fr),\n  column-gutter: 0.14cm,\n"
        f"  [{rve}],\n  [{materials}],\n  [{analysis}],\n)\n"
    )

    if spec.engineering_constants and spec.c_eff_gpa is not None:
        ec = spec.engineering_constants
        eng_block = (
            '#text(weight: "bold", size: 7pt)[Engineering constants]\n'
            "#text(size: 6.5pt)["
            f"$E$ = ({ec['E_x'] / 1e9:.1f}, {ec['E_y'] / 1e9:.1f}, {ec['E_z'] / 1e9:.1f}) GPa; "
            f"$G$ = ({ec['G_xy'] / 1e9:.2f}, {ec['G_xz'] / 1e9:.2f}, {ec['G_yz'] / 1e9:.2f}) GPa; "
            f"$nu$ = ({ec['nu_xy']:.2f}, {ec['nu_xz']:.2f}, {ec['nu_yz']:.2f})]\n"
        )
        labels = ["11", "22", "33", "23", "13", "12"]
        c = spec.c_eff_gpa
        matrix_rows = [("", *labels)]
        for i in range(6):
            matrix_rows.append((labels[i], *(f"{c[i, j]:.2f}" for j in range(6))))
        mcells = ", ".join(
            f"[#text(size: 6pt)[{_typst_escape(str(x))}]]"
            for row in matrix_rows for x in row
        )
        matrix_block = (
            "#table(\n  columns: (auto,) + (auto,) * 6,\n  inset: 1.5pt,\n"
            f"  stroke: 0.3pt,\n  {mcells}\n)\n"
        )
        footer = (
            "#grid(\n  columns: (1fr, 1fr),\n  column-gutter: 0.25cm,\n"
            "  align: (left + top, left + top),\n"
            f"  [{eng_block}],\n"
            f'  [#text(weight: "bold", size: 7pt)[$C_"eff"$ [GPa]] #v(0.5pt) {matrix_block}],\n)\n'
        )
    else:
        footer = '#text(size: 7pt)[(homogenisation skipped — no $C_"eff"$)]\n'

    panels = []
    if spec.figure_cuts and spec.figure_cuts.is_file():
        panels.append((_figure_image(spec.figure_cuts), spec.figure_col_fracs[0]))
    if spec.figure_iso and spec.figure_iso.is_file():
        panels.append((_figure_image(spec.figure_iso), spec.figure_col_fracs[1]))

    head = (
        '#set page(paper: "a4", flipped: true, margin: (x: 0.32cm, y: 0.2cm))\n'
        '#set text(size: 7.5pt, font: "Liberation Sans")\n'
        "#set par(leading: 0.42em)\n"
    )
    if not panels:
        return head + title_block + "#v(0.5pt)\n" + footer + "\n"

    cols = ", ".join(f"{frac}fr" for _, frac in panels)
    cells = "".join(f"  [{img}],\n" for img, _ in panels)
    aligns = ", ".join(["center + horizon"] * len(panels))
    figure_row = (
        "#block(width: 100%, height: 100%)[#grid(\n"
        f"  columns: ({cols}),\n  rows: (1fr,),\n  column-gutter: 0.15cm,\n"
        f"  align: ({aligns}),\n{cells})]\n"
    )
    return (
        head + "#grid(\n  rows: (auto, 1fr, auto),\n  row-gutter: 0.12cm,\n"
        f"  [{title_block}],\n  [{figure_row}],\n  [{footer}],\n)\n"
    )


def compile_datasheet(
    typst_src: str, out_pdf: Path, *, out_png: Path | None = None, root: Path | None = None
) -> None:
    """Compile Typst to PDF (and optionally PNG); `root` holds the figure PNGs."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root or out_pdf.parent)
    src = root / "datasheet.typ"
    src.write_text(typst_src)
    proc = subprocess.run(
        ["typst", "compile", src.name, str(out_pdf.resolve())],
        cwd=root, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst compile failed ({proc.returncode}):\n{proc.stderr}")
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        proc2 = subprocess.run(
            ["typst", "compile", "--format", "png", "--ppi", "150", src.name,
             "datasheet-{p}.png"],
            cwd=root, capture_output=True, text=True,
        )
        if proc2.returncode != 0:
            raise RuntimeError(f"typst png export failed ({proc2.returncode}):\n{proc2.stderr}")
        shutil.copy(next(root.glob("datasheet-*.png")), out_png)


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def generate(
    json_path: str | Path,
    out_pdf: str | Path | None = None,
    *,
    out_png: str | Path | None = None,
    name: str | None = None,
    workdir: str | Path | None = None,
    skip_compile: bool = False,
) -> DatasheetSpec:
    """Build a datasheet for a CpropInput JSON case.

    Runs the MFEM backend for the full stiffness tensor, renders the two figure
    panels and (unless `skip_compile`) compiles the Typst page to `out_pdf`
    (and `out_png` if given). Returns the populated `DatasheetSpec`.
    """
    json_path = Path(json_path)
    inp = CpropInput(**json.loads(json_path.read_text())).model_dump()
    name = name or json_path.stem

    mesh = create_grooved_mesh(
        thickness=inp["thickness"], dx=inp["dx"], dy=inp["dy"],
        xcuts=inp["xgr"], ycuts=inp["ygr"], madd=tuple(inp["madd"]),
        tface=(inp.get("face") or {}).get("thickness", 0.0),
        kx=(inp.get("curvature") or {}).get("kx", 0.0),
        ky=(inp.get("curvature") or {}).get("ky", 0.0),
    )
    geom = geom_analysis(mesh)
    geom["rho_infused"] = (
        inp["core"]["rho"] * (1.0 - geom["resin_vf"])
        + inp["resin"]["rho"] * geom["resin_vf"]
    )
    logger.info("running MFEM backend for the full stiffness tensor")
    details = mfem_backend.runmfem(
        mesh, inp["resin"], inp["core"], inp.get("face"), return_details=True
    )

    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="b3core_ds_"))
    root.mkdir(parents=True, exist_ok=True)
    fig_cuts, fig_iso = root / "cuts.png", root / "iso.png"
    a_cuts = render_section_cuts(mesh, fig_cuts)
    render_isometric(mesh, fig_iso)
    a_iso = _png_aspect(fig_iso)
    total = a_cuts + a_iso

    spec = collect_spec(inp, mesh, geom, details, name=name, config_name=json_path.name)
    spec.figure_cuts = fig_cuts
    spec.figure_iso = fig_iso
    spec.figure_col_fracs = (a_cuts / total, a_iso / total)

    if out_pdf is not None and not skip_compile:
        compile_datasheet(build_typst(spec), out_pdf, out_png=out_png, root=root)
    return spec
