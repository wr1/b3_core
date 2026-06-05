# b3_core

Homogenized elastic properties for sandwich-panel cores with sawcuts and
machined grooves. The cuts fill with resin during infusion, altering the
effective stiffness and density of the core; this package predicts the
resulting orthotropic material and returns it as a `b3_mat.OrthotropicMaterial`
so it drops into the wider b3 section / beam pipeline.

## Approach

Structured 3D RVE mesh (PyVista) with cell-level tagging of core vs.
resin-filled groove. Six periodic-BC load cases (εxx, εyy, εzz, εxy, εxz, εyz)
run in parallel and the averaged reactions give the effective engineering
constants (Ex/Ey/Ez, Gxy/Gxz/Gyz, Poisson ratios) and infused density. Grooves
can be tapered by mould curvature (`kx`, `ky`) for curved panels.

Backends: **CalculiX** (default) and **MFEM** (installed as standard), plus
optional **FEniCSx**; set `"validate_with_ccx": true` to cross-check a backend
against ccx.

## Requirements

- Python ≥ 3.11, `uv`
- CalculiX (`ccx`) on PATH (default backend)
- `frd2vtu`, `treeparse`, `mfem`, `b3_mat` (resolved via `pyproject.toml`)
- `typst` on PATH (only for `b3_core datasheet`)

```bash
uv sync   # core + ccx + mfem backends
```

## CLI

```bash
uv run b3_core json examples/simple.json          # run a JSON case
uv run b3_core run --dx 50 --dy 50 -t 30 \
    --xgr "5,10,15,3" --core-e 4e9 --resin-e 3.5e9 -o out/   # direct flags
```

Each run writes `run<HASH>.json` with the engineering constants, geometry
metrics, and (optionally) a backend-vs-ccx comparison.

## Python API

```python
from b3_core import homogenize

# json_data is a path to a CpropInput JSON file (or an equivalent dict)
result = homogenize("examples/simple.json")
print(result.material)                  # b3_mat.OrthotropicMaterial
print(result.resin_volume_fraction, result.surface_area_factor)
```

`homogenize` wraps the lower-level `cprop` pipeline; call `cprop(...)` directly
for the full raw output dict.

See `examples/curved_panel/` and `examples/mfem_patterns/` for galleries.

## Visualization

`b3_core.viz` is a unified, high-level layer that makes a grooved-core design
understandable at a glance — geometry, phases, the FE mesh, orthogonal slices,
periodic deformation, and the homogenised stiffness tensor — for both
publication figures and interactive exploration. `GroovedCoreView` is the
one-call entry point; `CoreScene` is a fluent pyvista builder; `CoreModel`
caches the mesh + MFEM homogenisation; `CoreTheme` is the shared styling.

```python
from b3_core.viz import GroovedCoreView

view = GroovedCoreView.from_json("examples/mfem_patterns/two_sided.json")
view.gallery("board.png")                 # composite publication board
view.modulus_surface_png("modulus.png")   # directional Young's modulus E(n)
view.show()                               # native interactive window
view.serve("core.html")                   # interactive HTML viewer ([interactive] extra)
```

```bash
uv run b3_core view examples/mfem_patterns/two_sided.json --what gallery -o board.png
uv run b3_core view examples/mfem_patterns/two_sided.json --what all -o out/
```

The composite board — 3D geometry, an internal-architecture cutaway, the
directional modulus **surface** `E(n)` (it bulges along stiff directions), the
orthogonal cuts with the mesh, the signed 6×6 `C_eff` heatmap and the polar
`E(θ)` plots:

![Visualization gallery](docs/viz_gallery.png)

The directional Young's-modulus surface alone — the clearest single view of the
effective anisotropy:

![Directional modulus surface](docs/modulus_surface.png)

## Datasheet

`b3_core datasheet` renders a one-page report of a case — RVE/geometry, materials
and analysis settings, the internal groove structure with the mesh (plan + side
cross-sections), a 3D isometric of the resin-filled grooves, and the homogenised
engineering constants + 6×6 effective stiffness (via the MFEM backend). Needs the
[`typst`](https://typst.app) binary on PATH.

```bash
uv run b3_core datasheet examples/mfem_patterns/two_sided.json -o card.pdf --png card.png
```

```python
from b3_core.datasheet import generate
generate("examples/mfem_patterns/two_sided.json", "card.pdf", out_png="card.png")
```

![Example datasheet](docs/datasheet_example.png)

## Periodic deformation modes

Separate from the datasheet, `b3_core deformed` warps the RVE by the true
periodic displacement `u = E·x + w` for each of the six unit-strain load cases
(xx, yy, zz, yz, xz, xy) and renders a 2×3 montage — resin grooves coloured by
displacement magnitude, core translucent. Because the fluctuation `w` is
periodic, opposite faces deform compatibly (the visual check on the periodic BC).

```bash
uv run b3_core deformed examples/mfem_patterns/two_sided.json -o modes.png --warp 0.3
```

![Periodic deformation modes](docs/deformed_example.png)

## Animation

`b3_core animate` renders a short, silent-friendly "how it works" explainer
(MP4 + looping GIF) for social media — grooved geometry → resin infusion → FE
mesh → orthogonal slices → the **curvature sim** (grooves taper, the core drapes,
and the homogenised `E(κ)` curve grows) → the emergent stiffness tensor. Built on
`b3_core.viz`; needs the `[anim]` extra (`uv sync --extra anim`).

```bash
uv run b3_core animate examples/mfem_patterns/two_sided.json -o explainer.mp4
```

```python
from b3_core.viz.animate import render_explainer
render_explainer("examples/mfem_patterns/two_sided.json", "explainer.mp4", gif=True)
```

![Explainer animation](docs/explainer.gif)

## License

MIT
