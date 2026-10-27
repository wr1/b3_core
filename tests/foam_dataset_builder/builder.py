"""Foam surrogate dataset builder — standalone version.

Collects FEA results from run*.json output files into (features, targets) arrays
suitable for training the foam MoE surrogate.

Standalone: no b3_core import dependency. Inlines only the registry functions
needed (feature vector builder, foam taxonomy, stiffness converters).

Usage
-----

.. code-block:: python

    from dataset_builder_standalone import FoamDatasetBuilder

    builder = FoamDatasetBuilder(fixture_dir="examples/foam_synth/")
    builder.scan()
    X, y = builder.to_arrays()

Dependencies
------------

- numpy
- (optional) jsonschema for validation

Output schema
-------------

    features  : (N, 24)  float64  — MoE-ready feature vectors
    targets   : (N, 21)  float64  — upper-triangle stiffness tensor (Voigt)
    provenance : list[dict] — one entry per sample with foam_type, backend, geometry, etc.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Foam taxonomy (from foam_registry.py)
# ---------------------------------------------------------------------------

FOAM_CODES: Dict[str, int] = {
    "pvc_foam_high": 0,
    "pvc_foam_med": 1,
    "pmma_foam": 2,
    "pet_foam": 3,
    "aramid_honeycomb": 4,
    "balsa_foam": 5,
    "bamboo_foam": 6,
    "generic_foam": 7,
}

FOAM_CODE_NAMES: Dict[int, str] = {v: k for k, v in FOAM_CODES.items()}
ALL_FOAM_NAMES: List[str] = sorted(FOAM_CODES.keys())

_FOAM_MAX_DENSITY: Dict[str, float] = {
    "pvc_foam_high": 150.0,
    "pvc_foam_med": 100.0,
    "pmma_foam": 140.0,
    "pet_foam": 75.0,
    "aramid_honeycomb": 80.0,
    "balsa_foam": 200.0,
    "bamboo_foam": 120.0,
    "generic_foam": 150.0,
}

# Upper-triangle Voigt indices for symmetric 6x6 stiffness
_STIFFNESS_UPPER_TRIANGLE: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in range(6) for j in range(i, 6)
)
N_STIFFNESS_TARGETS = len(_STIFFNESS_UPPER_TRIANGLE)  # 21

LOG_MODULUS_FEATURE_INDICES: Tuple[int, ...] = (1, 3, 4, 5, 7)

# ---------------------------------------------------------------------------
# Feature bounds
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_BOUNDS: NDArray[np.float64] = np.array(
    [
        [0.0, 1.0],  # [0] Vf
        [0.5e9, 15.0e9],  # [1] E_m (Pa)
        [0.0, 0.5],  # [2] nu_m
        [50.0e9, 1000.0e9],  # [3] E_Lf (Pa)
        [5.0e9, 500.0e9],  # [4] E_Tf (Pa)
        [1.0e9, 300.0e9],  # [5] G_LTf (Pa)
        [0.0, 0.5],  # [6] nu_LTf
        [1.0e9, 300.0e9],  # [7] G_TTf (Pa)
        [0.0, 1.0],  # [8] Vf duplicate
        [0.0, 1.0],  # [9] normalised density
        [0.0, 1.0],  # [10] relative density
        [0.0, 1.0],  # [11] normalised cell size
        [0.0, 1.0],  # [12] normalised kerf depth
        [0.0, 5.0],  # [13] normalised kerf spacing
        [0.0, 2.0],  # [14] normalised curvature
        [0.0, 0.0],  # [15] pad (always 0)
    ],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENGINEERING_CONSTANTS = (
    "Exx",
    "Eyy",
    "Ezz",
    "Gxy",
    "Gxz",
    "Gyz",
    "nuxy",
    "nuxz",
    "nuyz",
)
REQUIRED_FIELDS = set(ENGINEERING_CONSTANTS) | {"hash"}


# ---------------------------------------------------------------------------
# Transform functions (standalone copies of foam_registry helpers)
# ---------------------------------------------------------------------------


def _transform_features_for_regression(
    features: NDArray[np.float64], *, log_modulus: bool = True
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


def encode_foam_code(name: str) -> int:
    if name not in FOAM_CODES:
        raise ValueError(
            f"unknown foam type {name!r}; valid: {', '.join(ALL_FOAM_NAMES)}"
        )
    return FOAM_CODES[name]


def decode_foam_code(code: int) -> str:
    if code not in FOAM_CODE_NAMES:
        raise ValueError(f"unknown foam code {code}; valid: 0-7")
    return FOAM_CODE_NAMES[code]


def foam_max_density(name: str) -> float:
    return _FOAM_MAX_DENSITY[name]


def stiffness_to_targets(stiffness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map (N, 6, 6) symmetric tensors to (N, 21) regression targets."""
    c = np.asarray(stiffness, dtype=float)
    if c.ndim == 2:
        c = 0.5 * (c + c.T)
        return np.array([c[i, j] for i, j in _STIFFNESS_UPPER_TRIANGLE], dtype=float)[
            None, :
        ]
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
        raise ValueError(f"expected {N_STIFFNESS_TARGETS} targets, got shape {y.shape}")
    n = y.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)
    for k, (i, j) in enumerate(_STIFFNESS_UPPER_TRIANGLE):
        out[:, i, j] = y[:, k]
        if i != j:
            out[:, j, i] = y[:, k]
    return 0.5 * (out + np.transpose(out, (0, 2, 1)))


def relative_frobenius_error(
    predicted: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-sample ||C_pred - C_ref||_F / ||C_ref||_F."""
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = np.linalg.norm((pred - ref).reshape(pred.shape[0], -1), axis=1)
    denom = np.maximum(np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1), 1e-12)
    return diff / denom


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
    """Build the 24-dim feature vector consumed by the foam MoE surrogate."""
    c = np.asarray(constituents, dtype=float).ravel()
    if c.shape[0] != 8:
        raise ValueError(f"constituents must have 8 elements, got {c.shape[0]}")

    foam_code = encode_foam_code(foam_type)
    max_rho = foam_max_density(foam_type)

    # The normalisation below uses raw physical values (density kg/m3,
    # kerf_depth mm, kerf_spacing mm, curvature 1/m, cell_size mm).
    # NOTE: this function does NOT receive thickness, so kerf_depth_norm
    # and curvature_norm cannot be fully normalised here. The _extract_features
    # caller pre-computes these and passes them as raw values.
    n_density = (density or 0.0) / max_rho if max_rho > 0 else 0.0
    n_cell = (cell_size or 0.0) / 2.0
    kerf_spacing_val = kerf_spacing or 0.0
    n_kerf_spacing = (
        min(kerf_spacing_val / cell_size, 5.0)
        if (kerf_spacing_val > 0 and cell_size > 0)
        else 0.0
    )
    # kerf_depth and curvature are assumed pre-normalised by the caller
    n_kerf_depth = kerf_depth or 0.0
    n_curv = abs(curvature or 0.0)

    # Build feature vector using pre-normalised values (passed as raw params)
    # build_foam_feature_vector uses the same names internally, but we pass
    # the already-normalised values so the result is correct.
    vec = np.empty(16, dtype=float)
    vec[0:8] = constituents
    vec[8] = constituents[0]  # Vf duplicate
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


def engineering_to_stiffness(
    Exx: float,
    Eyy: float,
    Ezz: float,
    Gxy: float,
    Gxz: float,
    Gyz: float,
    nuxy: float,
    nuxz: float,
    nuyz: float,
) -> NDArray[np.float64]:
    """Convert 9 engineering constants to a 6x6 symmetric stiffness tensor."""
    S = np.zeros((6, 6), dtype=np.float64)
    S[0, 0] = 1.0 / Exx
    S[1, 1] = 1.0 / Eyy
    S[2, 2] = 1.0 / Ezz
    S[3, 3] = 1.0 / Gxy
    S[4, 4] = 1.0 / Gxz
    S[5, 5] = 1.0 / Gyz
    S[0, 1] = S[1, 0] = -nuxy / Exx
    S[0, 2] = S[2, 0] = -nuxz / Exx
    S[1, 2] = S[2, 1] = -nuyz / Eyy
    C = np.linalg.inv(S)
    return C


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """A single (features, targets, provenance) record extracted from one JSON."""

    index: int
    foam_type: str
    features: NDArray[np.float64]  # (24,)
    targets: NDArray[np.float64]  # (21,)
    stiffness_raw: NDArray[np.float64]  # (6, 6) before flattening
    backend: str
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


class FoamDatasetBuilder:
    """Collect FEA result JSONs into (features, targets) arrays for MoE training.

    Parameters
    ----------
    scan_dirs : list[str]
        Directories to scan for run*.json or fixture JSON files.
    validate_bounds : bool, default True
        If True, flag samples whose features fall outside DEFAULT_FEATURE_BOUNDS.
    """

    def __init__(
        self,
        scan_dirs: Optional[List[str]] = None,
        validate_bounds: bool = True,
    ):
        self.scan_dirs = scan_dirs or []
        self.validate_bounds = validate_bounds
        self.records: List[SampleRecord] = []
        self._scan_results: List[Dict[str, Any]] = []

    def scan(self, dirs: Optional[List[str]] = None) -> int:
        """Scan directories for FEA result JSON files.

        Returns the number of JSON files found.
        """
        dirs = dirs or self.scan_dirs
        self._scan_results = []
        total = 0

        for d in dirs:
            if not os.path.isdir(d):
                logger.warning("scan dir not found: %s", d)
                continue
            json_files = sorted(glob(os.path.join(d, "*.json")))
            for jf in json_files:
                fname = os.path.basename(jf)
                total += 1
                self._scan_results.append(
                    {
                        "file": jf,
                        "name": fname,
                        "size": os.path.getsize(jf),
                    }
                )
                logger.info("found: %s (%d bytes)", fname, os.path.getsize(jf))

        logger.info("scan complete: %d JSON files in %d dirs", total, len(dirs))
        return total

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load and validate a single FEA result JSON."""
        with open(filepath, "r") as fh:
            data = json.load(fh)
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            logger.warning("skipping %s: missing fields %s", filepath, missing)
            return {}
        return data

    def _extract_features(self, data: Dict[str, Any]) -> Optional[NDArray[np.float64]]:
        """Build the 24-dim feature vector from an FEA result dict."""
        meta = data.get("_meta", {})
        foam_type = meta.get("foam_type", "generic_foam")

        try:
            encode_foam_code(foam_type)
        except ValueError:
            logger.warning("unknown foam type %s, using generic_foam", foam_type)
            foam_type = "generic_foam"

        # Default constituent values
        Vf = 0.6
        E_m = 3.0e9
        nu_m = 0.35
        E_Lf = 230.0e9
        E_Tf = 10.0e9
        G_LTf = 7.0e9
        nu_LTf = 0.28
        G_TTf = 4.0e9

        thickness = data.get("thickness", 30.0)
        core = data.get("core", {})
        xgr = data.get("xgr", [])
        ygr = data.get("ygr", [])

        kerf_depth = 0.0
        if xgr and len(xgr[0]) >= 4:
            kerf_depth = xgr[0][2]
        elif ygr and len(ygr[0]) >= 4:
            kerf_depth = ygr[0][2]

        kerf_spacing = 0.0
        if xgr and len(xgr[0]) >= 4:
            kerf_spacing = xgr[0][1]
        elif ygr and len(ygr[0]) >= 4:
            kerf_spacing = ygr[0][1]

        curvature = 0.0
        curv = data.get("curvature", {})
        if "kx" in curv:
            curvature = curv["kx"]
        elif "ky" in curv:
            curvature = curv["ky"]

        max_rho = _FOAM_MAX_DENSITY.get(foam_type, 150.0)
        density_raw = core.get("rho", 100.0)
        cell_size = 1.0  # typical foam cell size

        # Pre-normalise geometry params (the registry builder doesn't receive thickness)
        n_density = density_raw / max_rho if max_rho > 0 else 0.0
        n_cell = cell_size / 2.0
        n_kerf_depth = kerf_depth / thickness if thickness > 0 else 0.0
        n_kerf_spacing = (
            min(kerf_spacing / cell_size, 5.0)
            if (kerf_spacing > 0 and cell_size > 0)
            else 0.0
        )
        n_curv = abs(curvature) * thickness  # |curv| * thickness

        feature_vec = build_foam_feature_vector(
            constituents=np.array(
                [Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf], dtype=np.float64
            ),
            foam_type=foam_type,
            density=n_density,
            kerf_depth=n_kerf_depth,
            kerf_spacing=n_kerf_spacing,
            curvature=n_curv,
            cell_size=n_cell,
            include_one_hot=True,
        )
        return feature_vec

    def _extract_targets(self, data: Dict[str, Any]) -> Optional[NDArray[np.float64]]:
        """Convert engineering constants to 21-dim stiffness targets."""
        try:
            Exx = data["Exx"]
            Eyy = data["Eyy"]
            Ezz = data["Ezz"]
            Gxy = data["Gxy"]
            Gxz = data["Gxz"]
            Gyz = data["Gyz"]
            nuxy = data["nuxy"]
            nuxz = data["nuxz"]
            nuyz = data["nuyz"]
        except KeyError as e:
            logger.warning("missing engineering constant %s", e)
            return None

        C = engineering_to_stiffness(Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz)
        targets = stiffness_to_targets(C)
        # stiffness_to_targets returns (1, 21) for single tensor; squeeze to (21,)
        if targets.ndim == 2 and targets.shape[0] == 1:
            targets = targets.ravel()
        return targets

    def _check_bounds(self, features: NDArray[np.float64]) -> List[str]:
        """Check features against DEFAULT_FEATURE_BOUNDS. Returns list of violations."""
        violations = []
        for i, (low, high) in enumerate(DEFAULT_FEATURE_BOUNDS):
            if low != high:  # skip pad (always 0)
                if features[i] < low or features[i] > high:
                    violations.append(
                        f"slot {i}: {features[i]:.6g} outside [{low:.2e}, {high:.2e}]"
                    )
        return violations

    def build(self, files: Optional[List[str]] = None) -> int:
        """Extract (features, targets, provenance) from scanned or specified files."""
        file_list = files or [r["file"] for r in self._scan_results]
        self.records = []
        idx = 0
        failed = 0

        for filepath in file_list:
            data = self._load_json(filepath)
            if not data:
                failed += 1
                continue

            features = self._extract_features(data)
            if features is None:
                logger.warning("failed to extract features from %s", filepath)
                failed += 1
                continue

            targets = self._extract_targets(data)
            if targets is None:
                logger.warning("failed to extract targets from %s", filepath)
                failed += 1
                continue

            out_of_bounds = []
            if self.validate_bounds:
                out_of_bounds = self._check_bounds(features)

            meta = data.get("_meta", {})
            provenance = {
                "file": filepath,
                "hash": data.get("hash", ""),
                "foam_type": meta.get("foam_type", "unknown"),
                "backend": meta.get("backend", data.get("backend", "unknown")),
                "date": meta.get("date", ""),
                "provenance_note": meta.get("provenance", ""),
                "thickness": data.get("thickness"),
                "xgr_count": len(data.get("xgr", [])),
                "ygr_count": len(data.get("ygr", [])),
                "out_of_bounds": out_of_bounds,
            }

            record = SampleRecord(
                index=idx,
                foam_type=provenance["foam_type"],
                features=features,
                targets=targets,
                stiffness_raw=None,
                backend=provenance["backend"],
                provenance=provenance,
            )

            self.records.append(record)
            idx += 1

        logger.info("build complete: %d samples, %d failed", idx, failed)
        return idx

    def to_arrays(
        self,
        log_modulus: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (features, targets) arrays ready for training."""
        if not self.records:
            raise RuntimeError("no records; call scan() then build() first")

        features_list = []
        targets_list = []

        for rec in self.records:
            feat = rec.features.copy()
            if log_modulus:
                for idx_slot in LOG_MODULUS_FEATURE_INDICES:
                    if feat[idx_slot] > 0:
                        feat[idx_slot] = np.log(feat[idx_slot])
            features_list.append(feat)
            targets_list.append(rec.targets)

        X = np.stack(features_list, axis=0)
        y = np.stack(targets_list, axis=0)
        return X, y

    @property
    def provenance(self) -> List[Dict[str, Any]]:
        return [rec.provenance for rec in self.records]

    @property
    def foam_types(self) -> NDArray[np.object_]:
        return np.array([rec.foam_type for rec in self.records])

    @property
    def sample_count(self) -> int:
        return len(self.records)

    @property
    def out_of_bounds_samples(self) -> List[SampleRecord]:
        return [r for r in self.records if r.provenance.get("out_of_bounds")]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_foam_dataset(
    scan_dirs: Optional[List[str]] = None,
    validate_bounds: bool = True,
) -> Tuple["FoamDatasetBuilder", NDArray[np.float64], NDArray[np.float64]]:
    """Convenience: scan, build, and return (features, targets)."""
    builder = FoamDatasetBuilder(scan_dirs=scan_dirs, validate_bounds=validate_bounds)
    builder.scan()
    builder.build()
    X, y = builder.to_arrays()
    return builder, X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["."]
    builder, X, y = build_foam_dataset(scan_dirs=dirs)
    print(f"\nDataset: {builder.sample_count} samples")
    print(f"Features: {X.shape}")
    print(f"Targets: {y.shape}")
    print(f"\nFoam types found: {np.unique(builder.foam_types)}")
    if builder.out_of_bounds_samples:
        print(f"\nWARNING: {len(builder.out_of_bounds_samples)} out-of-bounds samples")
    print("\nProvenance:")
    for p in builder.provenance:
        print(
            f"  {os.path.basename(p['file'])}: foam_type={p['foam_type']}, backend={p['backend']}"
        )
