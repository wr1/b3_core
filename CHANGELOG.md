# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Cache keys now include `CACHE_SCHEMA` 2 and a per-backend solver stamp.
  Entries written before this change are misses. `b3_core cache stats`,
  `cache inspect` (`ls`), and `cache purge --stale` list them and delete the
  stale ones. The package version stays out of the key.
- Run records use schema `b3_core.run/2` and carry a `b3_core` provenance
  block (commit, dirty flag, solver stamp, backend versions). A cache hit
  keeps `solved_with` and refreshes `served_by`. CalculiX ortho cards append
  a `**` comment with that block and do not change the `*ELASTIC` body.
- A curved case (`kx` or `ky` nonzero) no longer falls through to MFEM or
  CalculiX. `auto` requires FEniCSx. `numpy` is an explicit opt-in and
  warns. `mfem` requires `allow_pair_periodicity`. The CalculiX writer
  rejects a z face whose nodes are not transverse pairs.
- `b3_core run --json` prints an agent payload. `--units GPa` scales moduli
  and stiffness. `--no-write` skips the run file. Failures print
  `{"error": {"type", "message", "hint"}}` and exit 1.
- Saw-cut halo distance uses the pitch minimum image, so an offset is a phase
  shift. A pitch that does not tile `dx` or `dy` is rejected unless
  `allow_non_periodic` is set. The physics surrogate kerf floor is
  `mesh.MIN_HW` (`1e-3` mm).
- The κ × cell-size training grid uses MFEM with `allow_pair_periodicity`
  on curved rows when FEniCSx is not installed.
- A run record stores `diagnostics`: raw stiffness asymmetry, positive
  definiteness, Voigt/Reuss bounds, unmatched high-face nodes, periodic
  mismatch, kerf pinch, and per-face ties. `b3_core run --json` prints that
  block. `b3_core run --strict` exits 1 on a warn or a fail.
  `b3_core report ties` prints the coordinate ties without a solve.
  Published stiffness is unchanged. Numpy, MFEM, and CalculiX still
  symmetrise; the raw matrix is kept beside it. FEniCSx does not symmetrise
  and reports its MPC map.
- Through-thickness stations for a kerf taper are added only on an axis
  whose curvature is nonzero. `ky` on a pattern with no y-grooves leaves the
  mesh and the stiffness unchanged.
- The z-face periodic master is the more uniform face, not always `z = 0`.
  A bottom-mouth pinch and its top-mouth mirror then share one discrete
  space. Flat meshes are unchanged. Numpy, MFEM, and CalculiX stamps are
  `*/2026-10-10-uniform-master`. FEniCSx is `fenicsx/2026-10-10-energy`:
  the published matrix is the energy inner product, and a z-face tie whose
  master is still an x or y slave is back-substituted until the chain
  settles. Earlier cache entries miss.
- `b3_core check backends` compares stiffness across backends on a
  curvature list. The flat anchor fails above 0.01 % relative difference.
  Curved growth is flagged and does not fail. `check cross` is the flat
  point. `--refine` densifies `madd` by inserting midpoints.
- Curvature commands accept `--unit 1/mm|1/m|R-mm|R-m` and store 1/mm.
  `kerf_openings` rows include `opens_for` (`k>0` or `k<0`). The datasheet
  and `viz sign` read that row. Signed radius is `1/(1000·k)` metres.
- `b3_core run --json` after a subcommand is the agent payload. A leading
  `--json` still prints the treeparse command schema.
- `b3_core doctor --json` imports the FEniCSx stack and solves a 2×2×2
  cube. `is_fenicsx_available()` does that import, cached.
  `environment-fenicsx.yml` pins the known-good stack. `make env-fenicsx`
  prints the create line and does not touch `~/envs/b3`. CI jobs have
  timeouts, and the FEniCSx job installs from that file in separate pytest
  processes.
- `b3_core sweep grid` writes a curvature grid (`grid.csv`,
  `manifest.json`, and parquet when pyarrow imports). The default backend
  is `fenicsx`. `--workers` above 1 is a process pool with
  `OMP_NUM_THREADS=1` in each child. An inert axis errors unless
  `--allow-inert-axis`.
- `Material.source` and `Material.reference` record neat, infused,
  estimated, or calibrated constituents. Both are stripped from the
  canonical cache JSON.
- The curvature tolerance table in the backends reference is the halo
  mesh. FEniCSx and MFEM agree to better than 6×10⁻⁵ %. Numpy is
  flat-matched and 0.035 % to 0.053 % away at `|k| = 10⁻³` on refine 0.
  One madd refinement widens that to 0.044 % and 0.058 %. The offset is
  in the formulation, so FEniCSx stays the curved reference.
- `b3_core fit` calibrates a case: bounds, residuals, estimate-halo,
  sensitivity, refine, design, a quadratic surface, and `fit run`.
  Exit status is 0 only for `converged`, `rsm_only`, or `no_free_params`.
- `viz datasheet --full` builds the curvature sheet from `data.json`.
  The agent writes `prose.json` only. A missing acceptance stamp needs
  `--draft`. `viz figs` writes the percent-change, heatmap, and density
  figures. Engineering-constant names are `Ex` and `Gxy`, with the
  underscored aliases equal to them.
- `check reproduce`, `check accept`, and `check convergence` score stored
  artifacts. `check accept` fails closed and does not solve. A named
  profile is loaded only when `--profile` is passed.
- `cache export` and `cache import` move a zip of JSON entries plus a
  stamp manifest. Import skips keys that already exist unless
  `--overwrite`.
- `report build` writes a markdown and Typst skeleton from a project
  file. `kerfs map` is a 3×3 open/flat/closed grid. `diagnose bisect`
  separates mesh, field, and wall morph.
- `treeparse` is pinned to `>=0.3.4,<0.4`. The report extra is
  `typst>=0.12`. CI installs with `uv sync --locked`.

## [0.3.1] — 2026-10-07

### Changed

- MFEM and FEniCSx handle orthotropic constituents and the graded resin halo
  via the shared per-Gauss-point 6×6 stiffness (the same constitutive map as
  numpy; MFEM matches it to ~1e-6).
- `auto` prefers FEniCSx when that environment is installed. Its periodic
  constraint is the finite-element projection of the image, including a face
  that is not a tensor grid. Otherwise `auto` prefers MFEM. Both return nodal
  displacements for datasheet, deformed view, and `CoreModel`. Numpy remains
  the last resort and the only source of element strains.
- FEniCSx factors the periodic system once per RVE (MUMPS LU) and reuses that
  factor for the six unit-strain loads.
- Curvature × cell-size sweeps resolve `auto` instead of forcing numpy.
- The halo constitutive map (`P(resin)`, `local_cloud` averaging, phase
  attributes and rule-of-mixtures blending) now lives in one solver-neutral
  module, `b3_core.solvers.sampling`; the numpy and MFEM backends both call it
  instead of each carrying a copy.

## [0.3.0] — 2026-09-29

### Changed (breaking behaviour)

- `homogenize()` is now pure: it returns a `CoreResult` and writes no files.
  Use `homogenize(case, write=True, workdir=...)` or `homogenize_to_disk()` for
  the old file output. Repeated identical calls no longer raise `FileExistsError`.
- Backend default is `"auto"`. An explicit backend that cannot handle the case
  (orthotropic constituents or resin halo) now warns and falls back; this will
  raise `BackendCapabilityError` in 1.0.
- Run files are `run<sha256[:12]>.json` with a namespaced layout
  (`input` / `geometry` / `result`).

### Added

- `Cache` protocol with `MemoryCache` and `DiskCache`; `homogenize(case, cache=...)`.
- `SolverBackend` protocol and registry (`b3_core.solvers`), entry-point group
  `b3_core.solvers`.
- `CoreResult.stiffness` (6×6) and `run_case()` → `RunRecord`.
- Typed `Groove`, `Face`, `Curvature`, `Scoring` models.
- `ky` support in surrogate training sweeps.
- CalculiX `*elastic,type=ortho` card from C_eff (`CoreResult.ccx_ortho()`,
  `CoreModel.ccx_ortho()`, `b3_core run CASE --ccx-ortho out.inp`).
- Git pre-commit hook (`.githooks/pre-commit`): ruff check + format on
  staged `src/` / `tests/` Python (`make install` sets `core.hooksPath`).

### Deprecated

- `cprop` (use `run_case` or `homogenize`), `b3_core.io.*`
  (use `b3_core.solvers.*`), `sweep homogenise` (use `sweep homogenize`),
  `cache_path=` on curvature sweeps (use `cache=`).
  `CpropInput` and `Textile` stay as aliases of `CaseInput` and `CoreCase`.

## [0.2.0] — 2026-09-14

Work since `v0.1.0` (tag `42463eac`, 2026-08-07).

### Added

- **Textile-as-code:** `b3_core.cases` factories (`plain`, `uniaxial`,
  `crossed`, `two_sided`, `curved_panel`, `grid_scored`) plus fluent helpers
  (`.with_curvature`, `.with_halo`, `.with_backend`, `.with_thickness`,
  `.to_json`). `normalize_case` accepts a path, dict, `CpropInput`, or
  `Textile`.
- **GitHub Pages:** prebuilt `site/`, `pages.yml`, `make docs-build` /
  `docs-preview` (`basePath=/b3_core`).
- **Agent-use** docs guide; halo explainer figures and resin-halo rewrite.
- **Coverage gate** + `badges/coverage.json` (mainline ≥90%, badge ~92%);
  `make cov`.
- README badges (CI, Release, Pages, coverage, Python, Ruff, license).
- Dependabot weekly updates for GitHub Actions and Python (`pip`) deps.

### Changed

- README / SKILL rebalanced toward textile-as-code; YAML/JSON-first authoring
  demoted to optional CLI interchange.
- Docs: library aims (curvature-dependent cores and FEA surrogates); static
  export without Dev KB chrome.
- Packaging: `b3_mat` from git (`wr1/b3_mat`); drop machine-local
  `treeparse` path pin (PyPI). `uv sync --extra dev` no longer needs
  `/home/wr1/...` or a sibling `../b3_mat`.
- CI: slim `b3_mat` install; `numba>=0.60` override for mfem on Python 3.12.
- Linguist: mark `site/**` as generated (`.gitattributes`).
- Example cases: generic `grid_scored.json` / `grid_scored_halo.json`
  (geometry and material cards slightly off round datasheet numbers).

### Fixed

- Release workflow: checkout so CHANGELOG `body_path` resolves.
- Docs image URLs for GitHub Pages `basePath`.

### Removed

- Root `foam_schema.md` (design note lives in `notes/plans/`).

## [0.1.0] — 2026-08-07

First public release of **b3_core**: periodic-BC homogenization of grooved
sandwich-panel cores (PVC foam / balsa) with optional mould-curvature kerf
taper and resin-halo grading.

### Added

- **Homogenization pipeline** (`cprop` / `homogenize`) → orthotropic engineering
  constants and infused density as `b3_mat.OrthotropicMaterial`.
- **Backends:** MFEM (default), CalculiX, FEniCSx, numpy; orthotropic or
  `core.cell_size` auto-routes to numpy.
- **Kerf open/close** via interval-affine wall morph
  (`slope = −sign(d)·κ·pitch/2`, `hw(z)`).
- **Resin halo** (Laustsen-style) graded `P(resin)` from foam `cell_size`.
- **Halo + curvature composition** (shared `hw(z)` for morph and ScoreField).
- **Physics surrogate** for mass lookup of stiffness/mass along a curvature
  vector (`b3_core surrogate fit|lookup`, `CorePhysicsSurrogate`).
- **CLI:** `run`, `sweep`, `viz` (gallery, halo, halo-curvature, datasheet,
  deformed), `surrogate`, `skill`.
- **DocKB docs** under `docs/` (concepts, guides, reference).
- **CI:** GitHub Actions lint (pre-commit / ruff) + test matrix (3.11, 3.12).
- **Release workflow** on `v*` tags (sdist/wheel + GitHub Release).

### Notes

- Geometry inputs are **mm**; material moduli and outputs are **SI** (Pa, kg/m³).
- Optional CalculiX / typst / FEniCSx features self-skip when tools are missing.

[0.2.0]: https://github.com/wr1/b3_core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wr1/b3_core/releases/tag/v0.1.0
