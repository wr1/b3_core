# Foam Surrogate — Phase 1 Feature Schema

**Date:** 2026-07-07
**Status:** AMENDED per Phase 0 findings (no foam-type FEA data exists)

---

## 1. Overview

This document defines the **feature schema** and **target schema** for the foam-type
Mixture-of-Experts (MoE) surrogate model for `b3_core`. The schema is grounded in:

- The `CpropInput` pydantic model from `core/cprop.py` (FEA input)
- The engineering constants produced by the homogenisation pipeline (`cprop()`)
- The 24-dim feature vector expected by `foam_registry.build_foam_feature_vector()`
- The 21-element upper-triangle stiffness tensor (Voigt) used as regression targets

---

## 2. Feature Vector (24-dim)

### 2.1 Justification: 24-dim vs 8 continuous features

The plan (Phase 1 body) specifies "foam_type (categorical) + 8 continuous features".
This document uses the **pre-existing 24-dim space** from `foam_registry.py` and
`b3_micromech` for three reasons:

1. **MoE expert separation.** The 8-dim one-hot foam code (slots 16–23) is the
   routing mechanism — the surrogate must see which expert is active. Without it,
   the model cannot learn per-foam behaviour. This is not optional.
2. **Constituent alignment.** Slots 0–8 (Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf,
   nu_LTf, G_TTf, Vf-dup) are the same 8 physical constituents that b3_micromech
   already uses. Replacing them would require a separate feature-engineering layer
   and break consistency across the b3 surrogate stack.
3. **Geometry extension.** The plan's 8 continuous features map roughly to the
   6 geometry slots (density, relative_density, cell_size, kerf_depth,
   kerf_spacing, curvature) plus 2 constituents — but the full b3_micromech
   constituent set (8 fields) is needed to evaluate design sweeps where fibre
   and matrix properties also vary.

The 8 "continuous features" in the plan refers to the *conceptual* design space
for the geometry+constituents. The actual vector has 16 float slots (8 constituents
+ 1 Vf-dup + 6 normalised geometry + 1 pad) plus 8 foam one-hot = 24 total.
No reduction is recommended.

### 2.2 Structure

| Slot(s)  | Feature            | Units | Notes |
|----------|--------------------|-------|-------|
| 0        | `Vf`               | —     | Fibre volume fraction (0–1) |
| 1        | `E_m`              | Pa    | Matrix Young's modulus |
| 2        | `nu_m`             | —     | Matrix Poisson ratio |
| 3        | `E_Lf`             | Pa    | Fibre longitudinal modulus |
| 4        | `E_Tf`             | Pa    | Fibre transverse modulus |
| 5        | `G_LTf`            | Pa    | Fibre longitudinal-shear modulus |
| 6        | `nu_LTf`           | —     | Fibre longitudinal-transverse Poisson |
| 7        | `G_TTf`            | Pa    | Fibre transverse-shear modulus |
| 8        | `Vf` (dup)         | —     | Duplicate of slot 0 (alignment with surrogate.py) |
| 9        | `density_norm`     | —     | Normalised density = rho_core / foam_max_density |
| 10       | `rel_density`      | —     | Same as density_norm (placeholder for future refinement) |
| 11       | `cell_size_norm`   | —     | Normalised cell size = cell_size_mm / 2.0 |
| 12       | `kerf_depth_norm`  | —     | Kerf depth / core thickness |
| 13       | `kerf_spacing_norm`| —     | Kerf spacing / cell_size, clamped to 5 |
| 14       | `curvature_norm`   | —     | \|curvature\| * thickness, clamped to 2 |
| 15       | `pad`              | —     | Always 0.0 |
| 16–23    | foam one-hot       | —     | 8-dim one-hot encoding of foam type code |

**Total:** 16 float features + 8 foam one-hot = **24-dim**

### 2.2 Pre-processing

- **Log-transform** on modulus features: slots 1, 3, 4, 5, 7 (only applied if value > 0).
  Controlled by `LOG_MODULUS_FEATURE_INDICES` in `foam_registry.py`.
- **Normalisation** is handled at the fixture / dataset-builder level using the foam-specific
  `_FOAM_MAX_DENSITY` map and the global scaling factors (2.0 mm cell size, etc.).

### 2.3 Default feature bounds (for out-of-bounds detection)

See `foam_registry.py` — `DEFAULT_FEATURE_BOUNDS`, shape (16, 2).

| Slot | Low bound | High bound | Meaning |
|------|-----------|------------|---------|
| 0    | 0.0       | 1.0        | Vf |
| 1    | 0.5e9     | 15.0e9     | E_m (Pa) |
| 2    | 0.0       | 0.5        | nu_m |
| 3    | 50.0e9    | 1000.0e9   | E_Lf (Pa) |
| 4    | 5.0e9     | 500.0e9    | E_Tf (Pa) |
| 5    | 1.0e9     | 300.0e9    | G_LTf (Pa) |
| 6    | 0.0       | 0.5        | nu_LTf |
| 7    | 1.0e9     | 300.0e9    | G_TTf (Pa) |
| 8    | 0.0       | 1.0        | Vf dup |
| 9    | 0.0       | 1.0        | norm. density |
| 10   | 0.0       | 1.0        | rel. density |
| 11   | 0.0       | 1.0        | norm. cell size |
| 12   | 0.0       | 1.0        | norm. kerf depth |
| 13   | 0.0       | 5.0        | norm. kerf spacing |
| 14   | 0.0       | 2.0        | norm. curvature |
| 15   | 0.0       | 0.0        | pad (zero) |

---

## 3. Target Space (21-dim)

### 3.1 Structure

The surrogate predicts the **upper triangle of the 6x6 stiffness tensor C** in Voigt notation,
flattened to 21 values:

```
(C_11, C_12, C_13, C_14, C_15, C_16,   # row 0 (i=0), j=0..5
 C_22, C_23, C_24, C_25, C_26,        # row 1 (i=1), j=1..5
 C_33, C_34, C_35,                      # row 2 (i=2), j=2..5
 C_44, C_45,                            # row 3 (i=3), j=3..5
 C_55,                                   # row 4 (i=4), j=4..5
 C_66)                                   # row 5 (i=5), j=5
```

### 3.2 Conversion from FEA output

FEA `cprop()` outputs **engineering constants** (9 values):

```
Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz
```

These must be converted to the full stiffness tensor via the standard orthotropic relations:

```
D_11 = (1 - nu_yz * nu_zy) / (E_y * E_z * Delta)
D_12 = (nu_xy + nu_zy * nu_xz) / (E_y * E_z * Delta)
...
```

where `Delta = 1/(E_x*E_y*E_z) * (1 - nu_xy*nu_yx - nu_yz*nu_zy - nu_zx*nu_xz - 2*nu_yz*nu_zy*nu_xz)`.

#### Justification: 9 engineering constants → 21 stiffness triangle

The plan's Phase 0 says b3_core emits 9 engineering constants, but the surrogate
predicts 21 values (upper-triangle Voigt stiffness). This encoding is required because:

1. **The surrogate is a stiffness regressor, not an engineering-constant regressor.**
   The MoE experts learn per-foam behaviour in the natural constitutive space.
   Predicting stiffness components directly lets the model capture coupled
   anisotropic behaviour (off-diagonal terms like C_13, C_23) that engineering
   constants collapse into Poisson ratios — Poisson ratios lose information about
   the underlying tensor structure.

2. **b3_micromech already uses the stiffness tensor internally.** The existing
   `foam_registry.py` (slots in foam_registry.py) stores and manipulates the full
   (6,6) stiffness matrix. The 21-dim flattened form is the canonical regression
   target used by the MoE surrogate code in `foam_moe.py`. Matching this means
   the dataset builder produces data in the exact format the training pipeline
   expects — no re-encoding step at training time.

3. **Reversibility to engineering constants.** The 21-dim tensor contains all 9
   engineering constants (Exx = 1/D_11 where D = C^{-1}), so any downstream
   consumer can recover the 9 values. The inverse direction (9 → 21) is
   underdetermined: 9 constants leave 12 unknowns in the 6x6 matrix. Only the
   full stiffness tensor enables correct stress-strain prediction under arbitrary
   multi-axial loading.

The `foam_registry.py` module provides:
- `stiffness_to_targets(stiffness_tensor)` — converts (N, 6, 6) to (N, 21)
- `targets_to_stiffness(targets)` — inverse: (N, 21) to (N, 6, 6)
- `relative_frobenius_error(pred, ref)` — per-sample error metric

**Note:** This conversion is a *required step* in the dataset builder. The dataset
builder receives 9 engineering constants from cprop() JSON, converts them to a
full (6,6) stiffness tensor, then flattens to 21 targets. This is already
implemented in `dataset_builder_standalone.py` (`engineering_to_stiffness` →
`stiffness_to_targets`).

### 3.3 Output engineering constants from stiffness tensor

If downstream consumers need engineering constants from predicted stiffness:

```
Exx = 1 / D_11    (where D = C^{-1} is the compliance tensor)
```

---

## 4. Foam Type (Categorical Routing)

### 4.1 Taxonomy

| Code | Type               | Material          | Typical E'  [GPa] | Stiffness Signature          |
|------|--------------------|-------------------|-------------------|------------------------------|
| 0    | pvc_foam_high      | PVC (Divinycell H/HDW) | 0.3–0.6     | Moderate E, low density      |
| 1    | pvc_foam_med       | PVC (Divinycell FM)    | 0.15–0.3    | Lower E, medium density      |
| 2    | pmma_foam          | PMMA (Rohacell WF/IG)  | 0.4–0.7     | Higher Tg, similar to PVC-H  |
| 3    | pet_foam           | PET (Divinagard)       | 0.08–0.25   | Low E, very low density      |
| 4    | aramid_honeycomb   | Nomex paper            | 0.03–0.08   | Honeycomb anisotropy         |
| 5    | balsa_foam         | Balsa wood core        | 2.0–4.0     | High E, high density         |
| 6    | bamboo_foam        | Bamboo fiber core      | 0.5–1.5     | Medium E, somewhat aniso     |
| 7    | generic_foam       | Unknown / fallback     | ?             | Soft blending from others    |

### 4.2 Foam-specific normalisation

Density normalisation uses foam-specific max densities (`_FOAM_MAX_DENSITY` in `foam_registry.py`):

| Foam type              | Max density (kg/m³) |
|------------------------|---------------------|
| pvc_foam_high          | 150.0               |
| pvc_foam_med           | 100.0               |
| pmma_foam              | 140.0               |
| pet_foam               | 75.0                |
| aramid_honeycomb       | 80.0                |
| balsa_foam             | 200.0               |
| bamboo_foam            | 120.0               |
| generic_foam           | 150.0               |

---

## 5. FEA Output Format (JSON)

### 5.1 File naming

Each `cprop()` run writes: `<dirname>/run<md5hash>.json`

The MD5 hash is computed from the string representation of the validated `CpropInput` dict.

### 5.2 Schema of output JSON

```json
{
  "Exx": float,        // Young's modulus x (Pa)
  "Eyy": float,        // Young's modulus y (Pa)
  "Ezz": float,        // Young's modulus z (Pa)
  "Gxy": float,        // Shear modulus xy (Pa)
  "Gxz": float,        // Shear modulus xz (Pa)
  "Gyz": float,        // Shear modulus yz (Pa)
  "nuxy": float,       // Poisson ratio y/x
  "nuxz": float,       // Poisson ratio z/x
  "nuyz": float,       // Poisson ratio z/y
  "resin_vf": float,   // Resin volume fraction (0–1)
  "rho_infused": float, // Infused core density (kg/m³)
  "area_increase": float, // Core surface / ungrooved area ratio
  // ... additional geometry/material specs from CpropInput
  "dx": float,
  "dy": float,
  "thickness": float,
  "xgr": [[offset, spacing, depth, width], ...],
  "ygr": [[...], ...],
  "core": {"E": float, "nu": float, "rho": float},
  "resin": {"E": float, "nu": float, "rho": float},
  "curvature": {"kx": float, "ky": float},
  "backend": "ccx|fenicsx|mfem|numpy",
  "hash": "md5string",
  // ... other CpropInput fields
}
```

---

## 6. Gaps from Phase 0

### 6.1 Missing foam-type FEA data

No FEA runs have been performed for any of the 8 foam types. The existing param sweep
data uses generic isotropic materials only.

### 6.2 Engineering constants → stiffness tensor conversion

The `cprop()` output is engineering constants. The MoE surrogate expects 21-dim stiffness.
The dataset builder **must** perform the conversion.

### 6.3 Provenance

Current `cprop()` JSON includes an MD5 hash of the input but no version, run metadata,
or foam type tagging. The dataset builder will add this via fixture metadata.

---

## 7. FEA Campaign Proposal (to be approved as separate card)

**Status:** PROPOSED — awaiting human go-decision before running any FEA.

### 7.1 Scope

To train a usable MoE surrogate across all 8 foam types, we recommend a campaign of:

- **3 foam types × 3 geometry configs × 3 backend load cases = 27 FEA solves** minimum
  to cover basic training (one per foam type per load case).
- **Full campaign:** 8 types × 5 geometries × 3 load cases = **120 solves** for robust coverage.

### 7.2 Estimated runtime

CCX solves: ~2–5 minutes each (depending on mesh size, ~1000–5000 elements).
FEniCSx/MFEM: ~5–15 minutes each.

Total estimated: **4–30 hours** wall time (sequential CCX).

### 7.3 Risk factors

- Orthotropic foam materials may require the `numpy` backend (slower but no CCX dependency).
- Mesh quality at extreme geometries (high kerf depth/spacing ratios).
- Convergence issues at high curvature values.

### 7.4 Recommendation

Start with a pilot: 2 foam types × 2 geometries × 2 load cases = **8 solves** to validate
the pipeline end-to-end before committing to the full campaign.