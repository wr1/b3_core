"""Foam-type taxonomy and registry for MoE surrogate routing.

Provides the canonical mapping between foam / core types, integer codes
(0–7), geometry parameters, and the feature-vector builder used by the
foam-aware MoE surrogate in ``foam_moe.py``.

Foam types (8 experts)
======================

|| Code | Type            | Material      | Typical E'  [GPa] | Stiffness Signature |
||------|-----------------|---------------|-------------------|---------------------|
|| 0    | pvc_foam_high   | PVC (Divinycell H/HDW) | 0.3–0.6     | Moderate E, low density, isotropic |
|| 1    | pvc_foam_med    | PVC (Divinycell FM)       | 0.15–0.3    | Lower E, medium density |
|| 2    | pmma_foam       | PMMA (Rohacell WF/IG)     | 0.4–0.7     | Higher Tg, similar E to PVC-H |
|| 3    | pet_foam        | PET (Divinagard/Airgrid)  | 0.08–0.25   | Low E, very low density, closed-cell |
|| 4    | aramid_honeycomb | Nomex paper        | 0.03–0.08   | Honeycomb anisotropy, low E' |
|| 5    | balsa_foam      | Balsa wood core       | 2.0–4.0     | High E, high density, anisotropic (grain direction) |
|| 6    | bamboo_foam     | Bamboo fiber core     | 0.5–1.5     | Medium E, renewable, somewhat anisotropic |
|| 7    | generic_foam    | Unknown / unclassified| ?               | Fallback expert — moderate defaults |

Expert mapping: the MoE router selects an expert (or blends top-2) based on
the foam type code. When the foam type is unknown at inference, expert 7
(generic_foam) is used as a fallback with soft blending from nearby experts.

Feature vector (24-dim MoE input)
==================================

[0-7]   constituent features — Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf
[8]     Vf duplicate (alignment with surrogate.py)
[9]     density (kg/m³), normalised by a reference max (≈ 200 kg/m³)
[10]    relative_density (core_density / foam_material_density), 0–1
[11]    cell_size_mm (mm), normalised by 2.0 (typical max)
[12]    kerf_depth_norm (kerf_depth / thickness), 0–1
[13]    kerf_spacing_norm (kerf_spacing / cell_size), 0–∞, clamped to 5
[14]    curvature_norm (|curvature| * thickness), 0–2
[15]    pad (always 0.0)
[16-23] one-hot foam type code (8 elements)

Total: 16 float features + 8 foam one-hot = 24-dim dense input.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Upper-triangle Voigt indices (row, col) for symmetric 6×6 stiffness.
# ---------------------------------------------------------------------------
_STIFFNESS_UPPER_TRIANGLE: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(6) for j in range(i, 6)
)
N_STIFFNESS_TARGETS = len(_STIFFNESS_UPPER_TRIANGLE)  # 21

# ---------------------------------------------------------------------------
# Foam taxonomy
# ---------------------------------------------------------------------------

FOAM_CODES: dict[str, int] = {
    "pvc_foam_high": 0,
    "pvc_foam_med": 1,
    "pmma_foam": 2,
    "pet_foam": 3,
    "aramid_honeycomb": 4,
    "balsa_foam": 5,
    "bamboo_foam": 6,
    "generic_foam": 7,
}

FOAM_CODE_NAMES: dict[int, str] = {v: k for k, v in FOAM_CODES.items()}

ALL_FOAM_NAMES: list[str] = sorted(FOAM_CODES.keys())

# Nominal max densities for normalisation (kg/m³)
_FOAM_MAX_DENSITY: dict[str, float] = {
    "pvc_foam_high": 150.0,
    "pvc_foam_med": 100.0,
    "pmma_foam": 140.0,
    "pet_foam": 75.0,
    "aramid_honeycomb": 80.0,
    "balsa_foam": 200.0,
    "bamboo_foam": 120.0,
    "generic_foam": 150.0,
}

# Geometry parameters per foam type.
# These refine the stiffness prediction beyond the core density / geometry params.
FOAM_GEOMETRY_PARAMS: dict[str, list[str]] = {
    "pvc_foam_high": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
    "pvc_foam_med": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
    "pmma_foam": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
    "pet_foam": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
    "aramid_honeycomb": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
        "wall_thickness_norm",
    ],
    "balsa_foam": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
        "grain_angle_deg",
    ],
    "bamboo_foam": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
    "generic_foam": [
        "density", "relative_density", "cell_size_mm",
        "kerf_depth_norm", "kerf_spacing_norm", "curvature_norm",
    ],
}

# Zero defaults (all geometry params are dimensionless ratios or normalised)
_DEFAULT_GEOMETRY: dict[str, float] = {
    "density": 0.0,
    "relative_density": 0.0,
    "cell_size_mm": 0.0,
    "kerf_depth_norm": 0.0,
    "kerf_spacing_norm": 0.0,
    "curvature_norm": 0.0,
    "wall_thickness_norm": 0.0,
    "grain_angle_deg": 0.0,
}

# Slot-to-parameter mapping for slots 9-14 (slot 15 is pad).
SLOT_TO_PARAM: list[str] = [
    "density",               # 9
    "relative_density",       # 10
    "cell_size_mm",           # 11
    "kerf_depth_norm",        # 12
    "kerf_spacing_norm",      # 13
    "curvature_norm",         # 14
]

# ---------------------------------------------------------------------------
# Feature / target transforms (mirrors surrogate.py conventions)
# ---------------------------------------------------------------------------

# Pa-scale moduli features: log-transform improves MLP fidelity across decades.
LOG_MODULUS_FEATURE_INDICES: tuple[int, ...] = (1, 3, 4, 5, 7)


def _transform_features_for_regression(
    features: NDArray[np.float64],
    *,
    log_modulus: bool = True,
) -> NDArray[np.float64]:
    """Optionally map modulus features to log(Pa) before scaling."""
    x = np.asarray(features, dtype=float)
    if not log_modulus:
        return x
    out = x.copy()
    if out.ndim == 1:
        for idx in LOG_MODULUS_FEATURE_INDICES:
            if out[idx] > 0:
                out[idx] = np.log(out[idx])
        return out
    for idx in LOG_MODULUS_FEATURE_INDICES:
        mask = out[:, idx] > 0
        out[mask, idx] = np.log(out[mask, idx])
    return out


def stiffness_to_targets(stiffness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map (N, 6, 6) symmetric tensors to (N, 21) regression targets."""
    c = np.asarray(stiffness, dtype=float)
    if c.ndim == 2:
        c = 0.5 * (c + c.T)
        return np.array(
            [c[i, j] for i, j in _STIFFNESS_UPPER_TRIANGLE], dtype=float
        )[None, :]
    if c.ndim != 3 or c.shape[1:] != (6, 6):
        raise ValueError(f"stiffness must have shape (N, 6, 6), got {c.shape}")
    c_sym = 0.5 * (c + np.transpose(c, (0, 2, 1)))
    return np.array(
        [[m[i, j] for i, j in _STIFFNESS_UPPER_TRIANGLE] for m in c_sym],
        dtype=float,
    )


def targets_to_stiffness(targets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reconstruct symmetric (N, 6, 6) tensors from (N, 21) targets."""
    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[1] != N_STIFFNESS_TARGETS:
        raise ValueError(
            f"expected {N_STIFFNESS_TARGETS} targets, got shape {y.shape}"
        )
    n = y.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)
    for k, (i, j) in enumerate(_STIFFNESS_UPPER_TRIANGLE):
        out[:, i, j] = y[:, k]
        if i != j:
            out[:, j, i] = y[:, k]
    return 0.5 * (out + np.transpose(out, (0, 2, 1)))


def feature_bounds_from_training(
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-feature [min, max] from training data, shape (D, 2)."""
    x = np.asarray(features, dtype=float)
    return np.column_stack([x.min(axis=0), x.max(axis=0)])


def relative_frobenius_error(
    predicted: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-sample ||C_pred - C_ref||_F / ||C_ref||_F."""
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = np.linalg.norm(
        (pred - ref).reshape(pred.shape[0], -1), axis=1
    )
    denom = np.maximum(
        np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1), 1e-12
    )
    return diff / denom


# ---------------------------------------------------------------------------
# Default feature bounds covering a wide range of composite constituents
# and geometry.  Shape (16, 2) — matches the 16 float feature slots
# (before one-hot).
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_BOUNDS: NDArray[np.float64] = np.array(
    [
        [0.0, 1.0],          # [0] Vf
        [0.5e9, 15.0e9],     # [1] E_m (Pa) — epoxy to metals
        [0.0, 0.5],          # [2] nu_m
        [50.0e9, 1000.0e9],  # [3] E_Lf (Pa)
        [5.0e9, 500.0e9],    # [4] E_Tf (Pa)
        [1.0e9, 300.0e9],    # [5] G_LTf (Pa)
        [0.0, 0.5],          # [6] nu_LTf
        [1.0e9, 300.0e9],    # [7] G_TTf (Pa)
        [0.0, 1.0],          # [8] Vf duplicate
        [0.0, 1.0],          # [9] normalised density
        [0.0, 1.0],          # [10] relative density
        [0.0, 1.0],          # [11] normalised cell size
        [0.0, 1.0],          # [12] normalised kerf depth
        [0.0, 5.0],          # [13] normalised kerf spacing
        [0.0, 2.0],          # [14] normalised curvature
        [0.0, 0.0],          # [15] pad (always 0)
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_foam_code(name: str) -> int:
    """Return the integer code for a foam type name.

    Parameters
    ----------
    name:
        One of the canonical foam type names
        (pvc_foam_high, pvc_foam_med, pmma_foam, pet_foam,
         aramid_honeycomb, balsa_foam, bamboo_foam, generic_foam).

    Returns
    -------
    int
        Integer code in 0..7.

    Raises
    ------
    ValueError
        If *name* is not a known foam type.
    """
    if name not in FOAM_CODES:
        raise ValueError(
            f"unknown foam type {name!r}; "
            f"valid: {', '.join(ALL_FOAM_NAMES)}"
        )
    return FOAM_CODES[name]


def decode_foam_code(code: int) -> str:
    """Return the foam type name for an integer code.

    Parameters
    ----------
    code:
        Integer in 0..7.

    Returns
    -------
    str
        Foam type name.

    Raises
    ------
    ValueError
        If *code* is not in 0..7.
    """
    if code not in FOAM_CODE_NAMES:
        raise ValueError(f"unknown foam code {code}; valid: 0-7")
    return FOAM_CODE_NAMES[code]


def foam_max_density(name: str) -> float:
    """Return the nominal max density (kg/m³) for normalisation of *name*.

    Raises
    ------
    ValueError if *name* is not a known foam type.
    """
    return _FOAM_MAX_DENSITY[encode_foam_code(name)]


def build_foam_feature_vector(
    constituents: NDArray[np.float64],
    foam_type: str,
    density: Optional[float] = None,
    kerf_depth: Optional[float] = None,
    kerf_spacing: Optional[float] = None,
    curvature: Optional[float] = None,
    cell_size: Optional[float] = None,
    *,
    include_one_hot: bool = True,
) -> NDArray[np.float64]:
    """Build the 24-dim feature vector consumed by the foam MoE surrogate.

    Parameters
    ----------
    constituents:
        8-dim array ``[Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf]``
        (same layout as the existing ``surrogate.py`` feature matrix).
    foam_type:
        Foam / core type name.
    density:
        Core apparent density (kg/m³). Defaults to 0 (normalised to foam max).
    kerf_depth:
        Saw-cut depth (mm). Defaults to 0 (normalised by core thickness).
    kerf_spacing:
        Saw-cut spacing (mm). Defaults to 0.
    curvature:
        Curvature 1/m (positive = open, negative = closed). Defaults to 0.
    cell_size:
        Cell size / groove period (mm). Defaults to 0.
    include_one_hot:
        If True (default), append the 8-dim one-hot foam code.

    Returns
    -------
    ndarray, shape (24,) or (16,)
        MoE-ready feature vector.
    """
    c = np.asarray(constituents, dtype=float).ravel()
    if c.shape[0] != 8:
        raise ValueError(
            f"constituents must have 8 elements, got {c.shape[0]}"
        )

    foam_code = encode_foam_code(foam_type)
    max_rho = foam_max_density(foam_type)

    # Normalise geometry params
    n_density = (density or 0.0) / max_rho if max_rho > 0 else 0.0
    n_cell = (cell_size or 0.0) / 2.0
    n_kerf_depth = (kerf_depth or 0.0)
    n_kerf_spacing = (
        (kerf_spacing or 0.0) / 2.0 if (kerf_spacing or 0.0) > 0 else 0.0
    )
    n_curv = abs(curvature or 0.0) * 0.1  # placeholder normalisation

    # Slots 0-7: constituent features
    # Slot 8: Vf duplicate
    # Slot 9: normalised density
    # Slot 10: relative density (placeholder 0, set from density/max)
    # Slot 11: normalised cell size
    # Slot 12: normalised kerf depth
    # Slot 13: normalised kerf spacing
    # Slot 14: normalised curvature
    # Slot 15: pad (always 0.0)
    vec = np.empty(16, dtype=float)
    vec[0:8] = c
    vec[8] = c[0]  # Vf duplicate
    vec[9] = n_density
    vec[10] = n_density  # relative_density defaults to same as density_norm
    vec[11] = n_cell
    vec[12] = n_kerf_depth
    vec[13] = n_kerf_spacing
    vec[14] = n_curv
    vec[15] = 0.0  # pad

    if include_one_hot:
        one_hot = np.zeros(8, dtype=float)
        one_hot[foam_code] = 1.0
        vec = np.concatenate([vec, one_hot])

    return vec


def build_feature_matrix_batch(
    vf: NDArray[np.float64],
    foam_type: str,
    *,
    density: Optional[NDArray[np.float64]] = None,
    kerf_depth: Optional[NDArray[np.float64]] = None,
    kerf_spacing: Optional[NDArray[np.float64]] = None,
    curvature: Optional[NDArray[np.float64]] = None,
    cell_size: Optional[NDArray[np.float64]] = None,
    E_m: Optional[float] = None,
    nu_m: Optional[float] = None,
    E_Lf: Optional[float] = None,
    E_Tf: Optional[float] = None,
    G_LTf: Optional[float] = None,
    nu_LTf: Optional[float] = None,
    G_TTf: Optional[float] = None,
) -> NDArray[np.float64]:
    """Build an (N, 24) foam MoE feature matrix from constituent data + foam.

    Convenience wrapper around ``features.build_feature_matrix`` plus
    normalised geometry params and foam one-hot expansion.

    Parameters follow the same contract as ``features.build_feature_matrix``
    for the constituent scalars; ``foam_type`` and optional geometry extend
    the vector to 24 dims.
    """
    from b3_micromech.features import build_feature_matrix

    base = build_feature_matrix(
        vf,
        E_m=E_m,
        nu_m=nu_m,
        E_Lf=E_Lf,
        E_Tf=E_Tf,
        G_LTf=G_LTf,
        nu_LTf=nu_LTf,
        G_TTf=G_TTf,
    )
    N = base.shape[0]
    vec = np.empty((N, 16), dtype=float)
    vec[:, 0:8] = base
    vec[:, 8] = base[:, 0]  # Vf duplicate

    max_rho = foam_max_density(foam_type)
    d = (
        np.asarray(density, dtype=float).ravel()
        if density is not None
        else np.zeros(N)
    )
    vec[:, 9] = d / max_rho if max_rho > 0 else 0.0
    vec[:, 10] = vec[:, 9]  # relative density
    vec[:, 11] = (
        (np.asarray(cell_size, dtype=float).ravel() / 2.0)
        if cell_size is not None
        else 0.0
    )
    kd = (
        np.asarray(kerf_depth, dtype=float).ravel() * 0.0
        if kerf_depth is not None
        else 0.0
    )
    vec[:, 12] = kd
    ks = (
        np.asarray(kerf_spacing, dtype=float).ravel()
        if kerf_spacing is not None
        else np.zeros(N)
    )
    vec[:, 13] = ks / 2.0 if np.any(ks > 0) else 0.0
    cv = (
        np.asarray(curvature, dtype=float).ravel()
        if curvature is not None
        else np.zeros(N)
    )
    vec[:, 14] = np.abs(cv) * 0.1

    foam_code = encode_foam_code(foam_type)
    one_hot = np.zeros((N, 8), dtype=float)
    one_hot[:, foam_code] = 1.0
    vec = np.concatenate([vec, one_hot], axis=1)
    return vec