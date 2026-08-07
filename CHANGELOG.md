# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/wr1/b3_core/releases/tag/v0.1.0
