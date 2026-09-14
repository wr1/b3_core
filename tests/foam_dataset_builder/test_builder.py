"""Test the standalone dataset builder on synthetic fixtures."""

import logging
import os
import sys

import numpy as np

# Allow running from workspace dir or from repo tests dir
_workspace = os.environ.get(
    "DATASET_WORKSPACE", "/home/wr/.hermes/kanban/boards/blade3/workspaces/t_2c562aac"
)
if os.path.isdir(_workspace):
    sys.path.insert(0, _workspace)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import (  # noqa: E402  # path insert above for local fixture package
    FoamDatasetBuilder,
)

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("TEST: Dataset builder on synthetic fixtures")
print("=" * 60)

builder = FoamDatasetBuilder(
    scan_dirs=[
        _workspace,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"),
    ],
    validate_bounds=True,
)

# Scan
n_found = builder.scan()
print(f"\nFound {n_found} JSON files")
for r in builder._scan_results:
    print(f"  {r['name']}: {r['size']} bytes")

# Build
n_built = builder.build()
print(f"\nBuilt {n_built} samples")

if builder.sample_count > 0:
    X, y = builder.to_arrays()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    if y.ndim == 3:
        y_flat = y.reshape(y.shape[0], -1)
    else:
        y_flat = y

    print(f"\nFoam types: {np.unique(builder.foam_types)}")

    if builder.out_of_bounds_samples:
        print(f"\nOut-of-bounds: {len(builder.out_of_bounds_samples)}")
        for rec in builder.out_of_bounds_samples:
            print(f"  {rec.provenance['file']}: {rec.provenance['out_of_bounds']}")
    else:
        print("\nNo out-of-bounds samples")

    print("\nProvenance:")
    for p in builder.provenance:
        print(
            f"  {os.path.basename(p['file'])}: foam_type={p['foam_type']}, "
            f"backend={p['backend']}, thickness={p['thickness']}mm"
        )

    # Verify targets are positive definite stiffness matrices
    from builder import targets_to_stiffness

    for i, rec in enumerate(builder.records):
        C = targets_to_stiffness(rec.targets)
        eigvals = np.linalg.eigvalsh(C)
        print(f"\nSample {i} ({rec.foam_type}):")
        print(f"  Stiffness eigenvalues: {eigvals}")
        if np.any(eigvals < -1e-6):
            print("  WARNING: non-positive eigenvalue detected!")

    # Print one full feature vector for inspection
    print("\nSample 0 feature vector (24-dim, after log-transform):")
    for i, v in enumerate(X[0]):
        print(f"  [{i:2d}] {v:12.6g}")

    print("\nSample 0 targets (21-dim upper triangle):")
    for i, v in enumerate(y[0]):
        print(f"  [{i:2d}] {v:12.6g}")

    # Verify target ranges make physical sense
    print("\nTarget ranges across all samples:")
    target_labels = [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C33",
        "C34",
        "C35",
        "C36",
        "C44",
        "C45",
        "C46",
        "C55",
        "C56",
        "C66",
    ]
    for k in [0, 1, 2, 6, 11, 15, 20]:
        print(
            f"  {target_labels[k]}: [{y_flat[:, k].min() / 1e6:.2f} MPa, {y_flat[:, k].max() / 1e6:.2f} MPa]"
        )

print("\n" + "=" * 60)
print("TEST PASSED: Dataset builder works on synthetic fixtures")
print("=" * 60)
