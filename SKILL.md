---
name: b3-core
description: >
  Get homogenized elastic properties and FEA-ready reports for grooved sandwich-
  panel cores. Use when an agent needs effective Ex/Ey/Ez, shear moduli, Poisson
  ratios, infused density, 6x6 stiffness, or material cards/tables for structural
  FEA (CalculiX, b3_mat, b3 pipeline). Covers grid-scored foam resin halo graded
  by cell size. Triggers on "homogenize core", "grooved core properties",
  "infused core stiffness", "resin halo", "grid-scored foam", "cell size halo",
  "core material for FEA", "b3_core", "cprop". Load via b3_core.skill_path() or
  `b3_core skill --stdout`.
---

# b3_core — homogenized properties for FEA

Predict effective orthotropic stiffness and infused density of PVC/balsa cores
with sawcuts and machined grooves (resin-filled during infusion). Output is
ready for structural FEA: engineering constants in SI (Pa, kg/m³), a
`b3_mat.OrthotropicMaterial`, and optional PDF/PNG datasheet with tables.

**Coordinate system:** x = machine direction, y = transverse, z = through-thickness.

**Source of truth:** this file at the **repo root**. It is also packaged as
`b3_core/SKILL.md` for installed use:

```python
from b3_core.skill import read_skill, skill_path
print(skill_path())   # checkout: …/src/b3_core/SKILL.md → root; wheel: site-packages
```

Or: `b3_core skill` (path) · `b3_core skill --stdout` (full text).

## Agent workflow

```
1. Write a CpropInput case file (YAML or JSON — geometry + core/resin materials)
2. b3_core run case.yaml  →  run<HASH>.json
3. Extract properties → table / JSON / CalculiX card
4. (Optional) b3_core sweep homogenise  or  b3_core viz datasheet …
5. Hand off OrthotropicMaterial or constants to the downstream FEA model
```

Always run commands yourself. Do not ask the user to run them.

## 1. Define the case

Write a YAML or JSON file describing the RVE and constituents. Minimal example
(ungrooved baseline):

```yaml
dx: 50
dy: 50
thickness: 30
xgr: []
ygr: []
core:  { E: 4e9,  nu: 0.3, rho: 100 }
resin: { E: 3.5e9, nu: 0.35, rho: 1100 }
```

Grooves: each row in `xgr` / `ygr` is `[offset, spacing, depth, width]` (mm).
Negative depth = groove opens toward the mould face (curved panels).

| Field | Role |
|-------|------|
| `dx`, `dy`, `thickness` | RVE size [mm] |
| `xgr`, `ygr` | Groove families (empty = plain core) |
| `core`, `resin` | Constituent materials (Pa, kg/m³) |
| `core.cell_size` | Foam cell size [mm] — enables resin halo (see below) |
| `scoring` | Halo tuning: `damage_cells`, `sampling` strategy |
| `curvature` | `{"kx", "ky"}` groove taper for curved panels [1/mm] |
| `face` | `{"thickness": mm}` optional stabilising layer |
| `backend` | `"mfem"` (default), `"ccx"`, `"fenicsx"`, `"numpy"` |
| `validate_with_ccx` | `true` to cross-check against CalculiX |

Example cases ship under `examples/` when developing from source
(`simple.json`, `with_grooves.json`, `mfem_patterns/*.json`,
`diab_gs30_scored.json` for grid-scored foam with halo).

### Resin halo — graded stiffness from foam cell size

Grid-scoring (knife or saw cuts) does more than leave a neat resin-filled kerf:
the cut **opens foam cells** along the groove walls and root. Those opened cells
take up resin during infusion, so the real cross-section is graded:

```
neat resin (kerf)  →  resin-rich zone (opened cells)  →  intact foam
```

Laustsen et al. (2014) CT scans show this interface is **wider than the nominal
slit** and depends on foam density — smaller cells (e.g. H130) give a thinner
resin-rich band than larger cells (e.g. H60). `b3_core` models this explicitly
instead of absorbing it into a calibrated failure strain alone.

**How stiffness is graded.** For each point in the foam (outside the neat kerf),
compute the distance `d` [mm] to the nearest **cut surface** — groove side walls
and groove root only. The unsawn outer top/bottom faces do **not** emit halo
(there is no cut there).

Two surface types carry independent halos; the field takes the **maximum**:

| Surface | Where | Cell state | Default `cell_size` |
|---------|-------|------------|---------------------|
| `saw_cut` | Groove walls + root | Opened by saw/knife | `core.cell_size` |
| `face` | Unsawn `z=0` / `z=thickness` | Closed | `0.25 × saw_cut` reach |

```
P(resin) = max( S_saw(d_to_groove), S_face(d_to_outer_face) )
```

Each `S(d)` is the survival function of that surface's cell-size distribution:
`S(0)=1` at the cut face, `S→0` at reach.

Local stiffness at each Gauss point in foam cells is then a **rule of mixtures**:

```
C_local = P · C_resin + (1 − P) · C_foam
```

So stiffness grades smoothly from full resin at the cut face to bulk foam away
from the groove. Bigger cells ⇒ longer reach ⇒ wider resin-rich band ⇒ **higher
homogenized moduli** (tests confirm `Ezz` increases monotonically with
`cell_size`).

**`core.cell_size` formats** [mm]:

| Form | Meaning | Survival `S(d)` |
|------|---------|-----------------|
| omitted / `null` | No halo — sharp kerf only | — |
| scalar `0.6` | Uniform cells in `[0, cs]` | `S(d) = 1 − d/cs` (linear) |
| `{mean, std, dist}` | Distributed cell size | `lognormal` (default) or `normal` survival, renormalised so `S(0)=1` |

Example — DIAB-style grid-scored PVC with 0.6 mm cells:

```json
{
  "core": {
    "E1": 32e6, "E2": 32e6, "E3": 70e6,
    "G12": 19e6, "G13": 19e6, "G23": 19e6,
    "nu12": 0.3, "nu13": 0.3, "nu23": 0.3,
    "rho": 60,
    "cell_size": 0.6
  },
  "scoring": {
    "damage_cells": 1.0,
    "surfaces": {
      "saw_cut": {},
      "face": { "scale": 0.25 }
    },
    "sampling": { "strategy": "local_cloud", "resolution": 3 }
  }
}
```

Disable the thinner face halo (saw-cut only): `"face": { "enabled": false }`.

**`scoring` block:**

| Key | Role |
|-----|------|
| `damage_cells` | Multiplier on max active surface reach → mesh halo band `s_halo` [mm] (default `1.0`) |
| `surfaces.saw_cut.cell_size` | Override saw-cut halo (default: inherit `core.cell_size`) |
| `surfaces.face.scale` | Face halo reach as fraction of saw-cut (default `0.25`, closed cells) |
| `surfaces.face.cell_size` | Explicit face halo spec; overrides `scale` |
| `surfaces.face.enabled` | `false` → no halo on unsawn top/bottom |
| `sampling.strategy` | `"exact"` — evaluate `P` at each Gauss point; `"local_cloud"` — sub-point cloud + IDW average (smoother) |
| `sampling.resolution` | Sub-points per direction for `local_cloud` (default `3`) |
| `sampling.idw_power` | Inverse-distance weight exponent for `local_cloud` (default `2`) |

**Backend and outputs.** Halo requires per-Gauss-point stiffness, so cases with
`core.cell_size` set auto-route to the **`numpy`** backend (ccx/mfem are
two-phase, isotropic integrators). Results include:

- `resin_vf` — neat kerf resin volume fraction
- `halo_vf` — extra resin from opened cells in the foam band
- `effective_resin_vf` = `resin_vf + halo_vf`
- `rho_infused` — density including halo resin

Use `examples/diab_gs30_scored.json` as the reference halo case. For FEA handoff,
the homogenized engineering constants already embed the graded stiffness — no
separate halo layer is needed in the global model.

## 2. Run homogenization

**CLI (writes `run<HASH>.json` next to the input):**

```bash
b3_core run path/to/case.yaml
b3_core path/to/case.json          # run is the default command
# dev checkout:
uv run b3_core run examples/with_grooves.json
```

**Python (returns typed result for b3 pipeline):**

```python
from b3_core import homogenize, cprop

result = homogenize("case.json")          # CoreResult
mat = result.material                     # b3_mat.OrthotropicMaterial

raw = cprop("case.json")                  # full dict + writes run*.json
```

Default backend is **mfem** (`uv sync --extra mfem`). Use `backend: ccx` when
you need CalculiX (`ccx` + `frd2vtu` on PATH). If `FileExistsError`, delete the
cached `run*.json` or use a fresh output directory.

## 3. Properties for FEA

### Engineering constants (primary handoff)

From `cprop` / `run*.json` or `result.engineering_constants`:

| Key | Meaning | FEA alias |
|-----|---------|-----------|
| `Exx` | E along x [Pa] | `Ex` |
| `Eyy` | E along y [Pa] | `Ey` |
| `Ezz` | E through-thickness [Pa] | `Ez` |
| `Gxy`, `Gxz`, `Gyz` | Shear moduli [Pa] | same |
| `nuxy`, `nuxz`, `nuyz` | Poisson ratios [-] | same |
| `rho_infused` | Effective density [kg/m³] | `rho` |
| `resin_vf` | Resin volume fraction [-] | metadata |
| `area_increase` | Groove surface-area factor [-] | metadata |

`CoreResult.material` maps `Exx→Ex`, `Eyy→Ey`, `Ezz→Ez` plus `rho_infused→rho`.

### Present as a markdown table (agent default)

After a run, format the constants for the user / downstream FEA deck:

```python
from b3_core import homogenize

r = homogenize("case.json")
m = r.material

rows = [
    ("Ex", f"{m.Ex/1e9:.4f} GPa"),
    ("Ey", f"{m.Ey/1e9:.4f} GPa"),
    ("Ez", f"{m.Ez/1e9:.4f} GPa"),
    ("Gxy", f"{m.Gxy/1e9:.4f} GPa"),
    ("Gxz", f"{m.Gxz/1e9:.4f} GPa"),
    ("Gyz", f"{m.Gyz/1e9:.4f} GPa"),
    ("νxy", f"{m.nuxy:.4f}"),
    ("νxz", f"{m.nuxz:.4f}"),
    ("νyz", f"{m.nuyz:.4f}"),
    ("ρ infused", f"{m.rho:.1f} kg/m³"),
    ("resin Vf", f"{r.resin_volume_fraction:.3f}"),
]
```

Show this table in the response. Include GPa for moduli (divide Pa by 1e9).

### JSON material card (for matdb / scripts)

```python
card = {
    "name": "grooved_core_homogenized",
    "Ex": m.Ex, "Ey": m.Ey, "Ez": m.Ez,
    "Gxy": m.Gxy, "Gxz": m.Gxz, "Gyz": m.Gyz,
    "nuxy": m.nuxy, "nuxz": m.nuxz, "nuyz": m.nuyz,
    "rho": m.rho,
}
```

Load into `b3_mat.MaterialDB` or pass to `b3_gx` / laminate builders.

### CalculiX `*elastic,type=engineering constants`

```text
*material,name=core_hom
*elastic,type=engineering constants
Ex,Ey,Ez,nuxy,nuxz,nuyz,Gxy,Gxz,
Gyz,293
```

Substitute SI values from `result.material`. Temperature line is placeholder
(293 K); adjust for the target FEA deck. For full 6×6 `C` tensor use
`CoreModel.from_json(path).stiffness` (Voigt order: xx, yy, zz, yz, xz, xy).

### 6×6 effective stiffness

```python
from b3_core.viz import CoreModel

model = CoreModel.from_json("case.json")
C = model.stiffness          # Pa, 6×6
C_GPa = C * 1e-9             # for tables
```

Use when the downstream solver needs the full anisotropic tensor rather than
engineering constants.

## 4. Reports

### Datasheet (PDF + PNG, publication-ready)

One-page report: RVE/geometry table, materials table, analysis settings,
groove figures, engineering constants, and 6×6 `C_eff` heatmap. Uses MFEM
backend for figures (no ccx solve for the report itself).

```bash
b3_core viz datasheet case.json -o report.pdf --png report.png
```

```python
from b3_core.datasheet import generate
spec = generate("case.json", "report.pdf", out_png="report.png")
# spec.engineering_constants, spec.c_eff_gpa available programmatically
```

Needs `typst` on PATH.

### Terminal comparison table (parametric sweeps)

When developing from source, use `b3_core sweep homogenise` or `make sweep`.
Response curves / gallery / GIFs: `examples/offline/`.

### Viz board (figures only, no PDF)

```bash
b3_core viz view case.json --what gallery -o board.png
```

## 5. Downstream FEA integration

| Target | What to pass |
|--------|--------------|
| `b3_mat` / `b3_gx` laminate | `homogenize(...).material` (`OrthotropicMaterial`) |
| CalculiX solid core layer | Engineering-constants card (§3) or matdb JSON |
| Full anisotropic solid | `CoreModel.stiffness` 6×6 tensor |
| Density in structural model | `rho_infused` from result |

The homogenized properties replace the **core layer** in a larger blade/panel
FEA model. Groove geometry stays in the RVE; the global model sees a smeared
orthotropic solid.

## Quick reference

```bash
b3_core run case.yaml
b3_core sweep homogenise
b3_core viz datasheet case.json -o core.pdf --png core.png
b3_core skill --stdout    # load this document
```

**Units:** input geometry mm; output moduli Pa; present moduli as GPa in tables.
**Backends:** orthotropic constituents or `core.cell_size` (resin halo) auto-route
to `numpy`.