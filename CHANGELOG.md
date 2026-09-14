# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CalculiX ``*elastic,type=ortho`` card from C_eff (`CoreResult.ccx_ortho()`,
  `CoreModel.ccx_ortho()`, `b3_core run CASE --ccx-ortho out.inp`).

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
