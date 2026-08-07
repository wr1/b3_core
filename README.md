<img src="docs/b3.svg" width="120" align="right" alt="b3 logo">

# b3_core

[![CI](https://github.com/wr1/b3_core/actions/workflows/ci.yml/badge.svg)](https://github.com/wr1/b3_core/actions/workflows/ci.yml)
[![Release](https://github.com/wr1/b3_core/actions/workflows/release.yml/badge.svg)](https://github.com/wr1/b3_core/actions/workflows/release.yml)
[![Pages](https://github.com/wr1/b3_core/actions/workflows/pages.yml/badge.svg)](https://wr1.github.io/b3_core/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wr1/b3_core/master/badges/coverage.json)](https://github.com/wr1/b3_core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/wr1/b3_core)](https://github.com/wr1/b3_core/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://wr1.github.io/b3_core/)

Homogenized elastic properties for sandwich-panel cores with sawcuts and
machined grooves. The cuts fill with resin during infusion, altering the
effective stiffness and density of the core; this package predicts the
resulting orthotropic material and returns it as a `b3_mat.OrthotropicMaterial`
so it drops into the wider b3 section / beam pipeline.

## Aims

1. **Curvature-dependent infused cores.** Saw-cut and kerfed cores laid into a
   curved mould open or pinch before infusion, so the cured resin lattice — and
   thus effective stiffness and mass — depends on **local mould curvature**.
2. **Model those properties.** Periodic-BC RVE homogenization with kerf taper
   from κ (`hw(z)`), optional resin halo, FEA-ready orthotropic cards.
3. **Lightweight surrogates for structural FEA.** Map a **vector of
   curvatures** (and halo width) to stiffness/mass for
   **curvature-dependent property assignment** in composite shell/solid models
   without re-homogenizing every station.

```
scored core + local κ  →  homogenize  →  (E, G, ν, ρ)
                              ↓
                     physics / grid surrogate
                              ↓
              κ(s)  →  mass lookup  →  FEA property field
```

## Approach

Structured 3D RVE mesh (PyVista) with cell-level tagging of core vs.
resin-filled groove. Six periodic-BC load cases (εxx, εyy, εzz, εxy, εxz, εyz)
give effective engineering constants (Ex/Ey/Ez, Gxy/Gxz/Gyz, Poisson ratios)
and infused density. Mould curvature (`kx`, `ky`) opens or pinches kerfs via an
**interval-affine wall morph** (`hw(z)`); foam bays become trapezoidal on the
flat FEA RVE (not voxel painting).

**Default backend is MFEM** (omit `backend` or set `"backend": "mfem"`). Also:
**CalculiX** (`ccx`), **FEniCSx**, **numpy**. Orthotropic constituents or
`core.cell_size` (resin halo) auto-route to **numpy**. Optional
`"validate_with_ccx": true` cross-checks any backend against CalculiX.

## Agent skill

Repo-root [`SKILL.md`](SKILL.md) is the source of truth for LLM agents. The same
file is packaged under `b3_core` for installed use:

```bash
b3_core skill              # path to packaged SKILL.md
b3_core skill --stdout     # dump the full skill text
```

```python
from b3_core.skill import read_skill, skill_path
```

## Requirements

- Python ≥ 3.11, `uv`
- PyMFEM (`mfem` package; default backend)
- CalculiX (`ccx`) + `frd2vtu` on PATH if using `backend: ccx` or `validate_with_ccx`
- `treeparse`, `b3_mat` (resolved via `pyproject.toml`)
- `typst` on PATH (only for `b3_core viz datasheet`)

```bash
uv sync --extra dev    # package + pytest + ruff + pre-commit
```

## Development

```bash
make install           # uv sync --extra dev && pre-commit install
make lint              # ruff check src tests
make format            # ruff format + ruff check --fix
make pre-commit        # full hook suite (ruff format/lint + basic files)
make test              # pytest (+ coverage; fail under 72%)
make cov               # same + refresh badges/coverage.json
```

Coverage is measured with `pytest-cov` on the mainline package (optional FEniCSx /
ccx / anim / offline-sweep modules are omitted — see `pyproject.toml`). CI
enforces `--cov-fail-under=72`. Refresh the README badge after a local run:

```bash
make cov
```

GitHub Actions (`.github/workflows/`):

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| **CI** | push / PR to `master` | pre-commit (ruff) + pytest + coverage on 3.11 & 3.12 |
| **Release** | tag `v*` | build sdist/wheel, GitHub Release with notes |
| **Pages** | push of `site/` | deploy prebuilt docs to GitHub Pages |

## Documentation

Rendered docs (static DocKB export): **[https://wr1.github.io/b3_core/](https://wr1.github.io/b3_core/)**

| Source | Role |
|--------|------|
| `docs/*.mdx` | Public guides, concepts, reference (edit these) |
| `public/figures/` | Figures copied into the static site |
| `site/` | Prebuilt HTML for GitHub Pages (`basePath=/b3_core`) |
| `make docs` | Live DocKB server (shared `dockb` runtime) |
| `make docs-build` | Static export → `site/` via `dockb-export` |
| `make docs-preview` | Local preview of the Pages build |

```bash
make docs-build      # needs dockb-export on PATH (shared dockb-runtime)
make docs-preview    # http://localhost:4173/b3_core/docs/
# commit site/ and push → Pages workflow deploys
```

## Release

First release is **v0.1.0**. See [CHANGELOG.md](CHANGELOG.md).

```bash
# after merging to master:
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
# Release workflow builds artifacts and creates the GitHub release
```

## CLI

Homogenisation runs from a YAML or JSON case file:

```bash
uv run b3_core run examples/simple.yaml
uv run b3_core examples/simple.json              # same (run is the default)
uv run b3_core sweep homogenise                  # parametric study
```

Each run writes `run<HASH>.json` with the engineering constants, geometry
metrics, and (optionally) a backend-vs-ccx comparison.

Optional figures and reports live under `b3_core viz` (not required for FEA
handoff). See `b3_core --help` and `b3_core viz --help`.

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

See `examples/curved_panel/` and `examples/mfem_patterns/` for focused demos.
For a unified parametric study (thickness, curvature, groove patterns) with
response curves and gallery renders, see `examples/param_sweeps/`:

```bash
b3_core sweep homogenise --root examples/param_sweeps
# or: make sweep
```

Response curves, gallery renders, and GIFs: `examples/offline/` (not mainline).

## Visualization (`b3_core viz`)

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
uv run b3_core viz view examples/mfem_patterns/two_sided.json --what gallery -o board.png
uv run b3_core viz halo examples/diab_gs30_scored.json -o examples/img
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

`b3_core viz datasheet` renders a one-page report of a case — RVE/geometry, materials
and analysis settings, the internal groove structure with the mesh (plan + side
cross-sections), a 3D isometric of the resin-filled grooves, and the homogenised
engineering constants + 6×6 effective stiffness (via the MFEM backend). Needs the
[`typst`](https://typst.app) binary on PATH.

```bash
uv run b3_core viz datasheet examples/mfem_patterns/two_sided.json -o card.pdf --png card.png
```

```python
from b3_core.datasheet import generate
generate("examples/mfem_patterns/two_sided.json", "card.pdf", out_png="card.png")
```

![Example datasheet](docs/datasheet_example.png)

## Periodic deformation modes

Separate from the datasheet, `b3_core viz deformed` warps the RVE by the true
periodic displacement `u = E·x + w` for each of the six unit-strain load cases
(xx, yy, zz, yz, xz, xy) and renders a 2×3 montage — resin grooves coloured by
displacement magnitude, core translucent. Because the fluctuation `w` is
periodic, opposite faces deform compatibly (the visual check on the periodic BC).

```bash
uv run b3_core viz deformed examples/mfem_patterns/two_sided.json -o modes.png --warp 0.3
```

![Periodic deformation modes](docs/deformed_example.png)

## Offline scripts

Optional workflows (GIF export, explainer MP4, scratch viz, interactive HTML) live
under [`examples/offline/`](examples/offline/README.md) — not part of `make` or
the main `b3_core` subcommands. Example explainer animation:

```bash
uv sync --extra anim
uv run python examples/offline/explainer.py examples/mfem_patterns/two_sided.json
```

```python
from b3_core.viz.animate import render_explainer
render_explainer("examples/mfem_patterns/two_sided.json", "explainer.mp4", gif=True)
```

![Explainer animation](docs/explainer.gif)

## License

MIT
