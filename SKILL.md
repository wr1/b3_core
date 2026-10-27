---
name: b3-core
description: >
  Get homogenized elastic properties and FEA-ready reports for grooved sandwich-
  panel cores, including curvature-dependent kerf open/close and lightweight
  surrogates for mass property lookup. Use when an agent needs effective
  Ex/Ey/Ez, shear moduli, Poisson ratios, infused density, 6x6 stiffness,
  material cards for structural FEA with curvature-dependent core assignment,
  resin halo, or b3_core physics surrogate. Triggers on "homogenize core",
  "grooved core properties", "infused core stiffness", "resin halo",
  "curvature-dependent core", "core surrogate", "b3_core", "cprop". Load via
  `b3_core.skill_path()` (also `b3_core.skill.skill_path()`) or `b3_core skill --stdout`.
---

# b3_core — homogenized properties for FEA

Predicts effective orthotropic stiffness and infused density of PVC/balsa cores
with sawcuts and grooves (resin-filled during infusion), including
curvature-dependent kerf opening and an optional resin halo from foam cell
size. Output feeds structural FEA: SI constants, a `b3_mat.OrthotropicMaterial`,
CalculiX cards, a 6×6 tensor, and a predicted datasheet.

Axes: x = machine direction, y = transverse, z = through-thickness.
Units: geometry in mm, curvature in 1/mm (mould radius R = 1/(1000·k) m),
Python moduli in Pa.

## Agent workflow

1. Define the case in Python (`b3_core.cases` factories or `CpropInput`).
2. `homogenize(case)` → `CoreResult`.
3. Take constants / material / CalculiX card / 6×6 stiffness for the FEA model.
4. Optionally make the predicted (uncalibrated) datasheet.

Run commands yourself; do not ask the user to. Prefer textile-as-code over
authoring YAML.

## Python

```python
from b3_core import grid_scored, homogenize

case = grid_scored(cell_size=0.6).with_curvature(kx=-0.008)  # bottom mouth: -kx opens
r = homogenize(case)            # writes nothing
r.engineering_constants         # Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nu*, rho_infused (SI)
r.material                      # b3_mat.OrthotropicMaterial (Ex, Ey, Ez, ..., rho)
r.stiffness                     # 6x6 Pa, Voigt xx, yy, zz, yz, xz, xy
print(r.ccx_ortho())            # CalculiX *elastic,type=ortho card
```

Other factories: `plain`, `uniaxial`, `crossed`, `two_sided`, `curved_panel`.

## CLI

```bash
b3_core run case.json --json --units GPa --no-write   # agent payload, moduli in GPa
b3_core run case.json --ccx-ortho core.inp
b3_core viz datasheet case.json -o core.pdf --png core.png   # needs typst
```

## Key outputs

- `Exx`, `Eyy`, `Ezz`, `Gxy`, `Gxz`, `Gyz` [Pa]; `nuxy`, `nuxz`, `nuyz`.
- `rho_infused` [kg/m³]; `resin_vf`, `halo_vf`, `effective_resin_vf`.
- `kerf_openings` rows with `opens_for`; read that row for the curvature sign.
- JSON payload: `properties.<name>.{value, unit}`, `diagnostics.checks`
  (`ok`/`warn`/`fail`), `case_hash`, `backend`, `cache_hit`. Errors print
  `{"error": {...}}` and exit non-zero.

## Backends

`auto` uses FEniCSx when installed (baseline; required for curved cases),
otherwise MFEM on flat cases. MFEM and CalculiX are flat cross-checks.
`numpy` is a fallback with a known O(k²) curvature offset.

## Datasheet

`viz datasheet` gives the flat, predicted card: geometry, materials,
constants and the `C_eff` heatmap. Label it "predicted, not calibrated"
unless a fit was applied.

## Quick reference

```bash
b3_core skill --stdout    # load this document
b3_core doctor --json     # FEniCSx import, MPI, MUMPS, 2×2×2 solve
b3_core run case.yaml
b3_core sweep homogenize
b3_core viz view case.json --what gallery -o board.png
b3_core report ties case.json --kx 5e-5 --json
b3_core check backends case.json --points "0,0 5e-5,0" --json
b3_core check cross case.json
b3_core sweep grid case.json --dry-run
b3_core cache stats
```

Skill source: `SKILL.md` at the repo root, packaged as `b3_core/SKILL.md`
(`b3_core.skill.skill_path()`).

## More in docs

- Case fields, grooves, mouth, scoring/halo keys: `docs/reference/input-schema.mdx`
- Resin halo model and `cell_size` forms: `docs/concepts/resin-halo.mdx`, `docs/concepts/halo-curvature.mdx`
- Curvature and kerf taper: `docs/concepts/curvature.mdx`; method: `docs/concepts/method.mdx`
- Cards (ortho, engineering constants, JSON), caching, surrogates: `docs/guides/homogenize.mdx`
- Outputs: `docs/reference/outputs.mdx`; CLI and JSON payload: `docs/reference/cli.mdx`
- Backends: `docs/reference/backends.mdx`; agents: `docs/guides/agent-use.mdx`; figures: `docs/guides/visualization.mdx`
