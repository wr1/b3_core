"""Foam-type Mixture of Experts surrogate package.

Provides foam-type specific experts for the b3_core homogenisation pipeline.
When the core material is a foam (or honeycomb) the MoE router selects
foam-specific neural-network experts that have been trained on foam-type
FEA data, delivering better stiffness predictions than a single global model.

Submodules
----------
foam_registry
    Foam type taxonomy, codes, geometry parameters, feature vector builder.
foam_experts
    PyTorch expert MLPs, shared encoder, router, and dynamic loading.
foam_moe
    High-level MoE wrapper: training, inference, save/load, integration.

Usage
-----

>>> from b3_core.core.foam_moe import FoamMoE, build_foam_feature_vector
>>> # Train from FEA data
>>> moe = FoamMoE.train(features, stiffness, foam_codes=foam_codes)
>>> # Predict
>>> pred = moe.predict_single(features, foam_type="pvc_foam_high")
"""

from __future__ import annotations

from b3_core.core.foam_registry import (
    ALL_FOAM_NAMES,
    DEFAULT_FEATURE_BOUNDS,
    FOAM_CODES,
    FOAM_CODE_NAMES,
    N_STIFFNESS_TARGETS,
    SLOT_TO_PARAM,
    build_foam_feature_vector,
    build_feature_matrix_batch,
    decode_foam_code,
    encode_foam_code,
    feature_bounds_from_training,
    foam_max_density,
    relative_frobenius_error,
    stiffness_to_targets,
    targets_to_stiffness,
)

from b3_core.core.foam_experts import (
    FoamExpertMLP,
    FoamExpertLoader,
    FoamMoEModel,
    SharedEncoder,
    create_experts,
    FoamRouter,
)

from b3_core.core.foam_moe import (
    FoamMoE,
    foam_type_from_material,
    save_hf_format,
)


__all__ = [
    # Foam registry
    "FOAM_CODES",
    "FOAM_CODE_NAMES",
    "ALL_FOAM_NAMES",
    "N_STIFFNESS_TARGETS",
    "SLOT_TO_PARAM",
    "DEFAULT_FEATURE_BOUNDS",
    "encode_foam_code",
    "decode_foam_code",
    "foam_max_density",
    "build_foam_feature_vector",
    "build_feature_matrix_batch",
    "stiffness_to_targets",
    "targets_to_stiffness",
    "relative_frobenius_error",
    "feature_bounds_from_training",
    # Expert models
    "FoamExpertMLP",
    "FoamExpertLoader",
    "SharedEncoder",
    "FoamRouter",
    "create_experts",
    "FoamMoEModel",
    # High-level API
    "FoamMoE",
    "foam_type_from_material",
    "save_hf_format",
]