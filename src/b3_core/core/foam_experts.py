"""Foam-specific expert models and dynamic loading for the MoE surrogate.

Each expert is a MLP that predicts the upper-triangle stiffness targets
(21 values) given the shared encoder output (32-dim). Experts are
specialized per foam / core type and can be loaded dynamically from
disk via the ``FoamExpertLoader`` utility.

Expert layout
=============

Each expert: Dense(64→64, ReLU, BN, Do=0.15) → Dense(64→64, ReLU, BN) → Dense(64→21, linear)
Total per expert: ~10K params. 8 experts → ~85K params total (with encoder).

Dynamic loading support
=======================

The FoamExpertLoader can save / load individual expert weights from
separate files, enabling:

* Pre-trained foam-specific experts (e.g., PVC foam expert trained on
  Divinycell data) to be loaded on demand.
* Incremental training: add new foam types without retraining all experts.
* Model versioning: swap out individual expert weights.

Usage example
=============

>>> from b3_core.core.foam_experts import (
...     FoamExpertMLP,
...     create_experts,
...     FoamExpertLoader,
... )
>>> # Create fresh experts (or load pre-trained)
>>> loader = FoamExpertLoader("/path/to/experts")
>>> experts, encoder, router = loader.load_all()
>>> # Or build from scratch:
>>> from b3_core.core.foam_experts import create_experts, SharedEncoder, MoERouter
>>> encoder = SharedEncoder(input_dim=24)
>>> experts = create_experts(hidden_dim=64, n_experts=8)
>>> router = MoERouter(input_dim=32, n_experts=8)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from b3_core.core.foam_registry import (
    ALL_FOAM_NAMES,
    FOAM_CODE_NAMES,
    encode_foam_code,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch model components (conditional on torch availability)
# ---------------------------------------------------------------------------

def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for MoE foam experts. "
            "Install with: pip install torch"
        )


class FoamExpertMLP(nn.Module):
    """Per-foam-type MLP expert.

    Architecture:
        Dense(64 → 64, ReLU, BatchNorm, Dropout 0.15)
        → Dense(64 → 64, ReLU, BatchNorm)
        → Dense(64 → 21, linear)

    Outputs the 21 upper-triangle elements of a symmetric 6×6 stiffness
    tensor in Voigt notation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 21,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.do1 = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: (N, hidden_dim) encoded features.

        Returns
        -------
        (N, output_dim) stiffness targets.
        """
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.layer2(x)))
        return self.layer3(x)


class SharedEncoder(nn.Module):
    """Shared feature encoder for all foam experts.

    Architecture:
        Dense(input_dim → 64, ReLU, BN) → Dropout(0.1)
        → Dense(64 → 64, ReLU, BN) → Dropout(0.1)
        → Dense(64 → 32, ReLU, BN)

    The encoder learns cross-foam-type representations from the universal
    constituent features (Vf, moduli, etc.).
    """

    def __init__(
        self,
        input_dim: int = 24,
        encoder_dims: tuple[int, ...] = (64, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, out_dim in enumerate(encoder_dims):
            layers.append(nn.Linear(prev_dim, out_dim))
            if i < len(encoder_dims) - 1:  # no BN on last layer
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = out_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FoamRouter(nn.Module):
    """Soft router: maps encoder output → foam-type gate weights.

    Produces a gate weight vector of length ``n_experts`` via a linear
    projection followed by Softmax. Supports both soft (all experts) and
    hard (deterministic by foam code) routing modes.
    """

    def __init__(
        self,
        input_dim: int = 32,
        n_experts: int = 8,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, n_experts)

    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 2,
        foam_codes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return gate weights and optional hard expert indices.

        Parameters
        ----------
        x: (N, 32) encoder output.
        top_k: Number of experts to weight (default 2).
        foam_codes: (N,) int tensor of foam codes, or None.
            When provided, uses deterministic hard-masking: only the
            expert matching each foam code receives non-zero weight.

        Returns
        -------
        gates: (N, n_experts) normalised gate weights.
        expert_indices: (N,) or None — deterministic expert index per sample.
        """
        raw = self.projection(x)  # (N, 8)

        if foam_codes is not None:
            # Hard mask: zero out non-matching experts, renormalise.
            codes = foam_codes.long()
            mask = F.one_hot(codes, num_classes=raw.shape[1]).float()
            masked = raw + torch.log(mask + 1e-9)
            gates = F.softmax(masked, dim=1)
            expert_indices = codes
        else:
            gates = F.softmax(raw, dim=1)  # (N, 8)

            if top_k <= 1:
                # Top-1: deterministic assignment
                expert_indices = torch.argmax(gates, dim=1)
            else:
                # Top-k: select top-k gates, zero the rest, renormalise.
                top_values, top_indices = torch.topk(gates, top_k, dim=1)
                sparse = torch.zeros_like(gates)
                sparse.scatter_(1, top_indices, top_values)
                gates = sparse / (sparse.sum(dim=1, keepdim=True) + 1e-9)
                expert_indices = None  # soft blend

        return gates, expert_indices


def create_experts(
    hidden_dim: int = 64,
    n_experts: int = 8,
    output_dim: int = 21,
) -> nn.ModuleList:
    """Create a list of ``n_experts`` FoamExpertMLP instances.

    Parameters
    ----------
    hidden_dim: Hidden layer size (default 64).
    n_experts: Number of foam experts (default 8).
    output_dim: Number of stiffness targets (default 21).

    Returns
    -------
    nn.ModuleList of FoamExpertMLP instances.
    """
    return nn.ModuleList([
        FoamExpertMLP(hidden_dim=hidden_dim, output_dim=output_dim)
        for _ in range(n_experts)
    ])


# ---------------------------------------------------------------------------
# Dynamic expert loading / saving
# ---------------------------------------------------------------------------

@dataclass
class ExpertMeta:
    """Metadata for a saved foam expert.

    Attributes
    ----------
    foam_type: Canonical foam type name.
    foam_code: Integer code (0–7).
    hidden_dim: Hidden layer size.
    output_dim: Number of output targets.
    version: Model version string.
    state_key: Key used in the saved state dict.
    """
    foam_type: str
    foam_code: int
    hidden_dim: int = 64
    output_dim: int = 21
    version: str = "1.0.0"

    @property
    def state_key(self) -> str:
        return f"expert_{self.foam_code}"

    def to_dict(self) -> dict:
        return {
            "foam_type": self.foam_type,
            "foam_code": self.foam_code,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpertMeta":
        return cls(**d)


class FoamExpertLoader:
    """Utility for loading / saving individual foam experts on disk.

    Supports:

    * Saving each expert to a separate file: ``experts/<foam_type>.pt``
    * Loading a specific expert by foam type or code.
    * Loading all experts at once (building encoder + router if present).
    * Versioned checkpoints with metadata JSON.

    Directory structure
    -------------------

    .. code-block:: text

        <save_dir>/
        ├── experts/
        │   ├── pvc_foam_high.pt
        │   ├── pvc_foam_med.pt
        │   ├── ...
        │   └── generic_foam.pt
        ├── shared_encoder.pt
        ├── router.pt
        └── metadata.json

    Usage
    -----

    >>> loader = FoamExpertLoader("/path/to/experts")
    >>> # Save
    >>> loader.save_expert(0, expert_module, encoder, router)
    >>> # Load all
    >>> experts, encoder, router = loader.load_all()
    >>> # Load specific expert
    >>> expert = loader.load_expert("pvc_foam_high")
    """

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.expert_dir = self.save_dir / "experts"
        self.expert_dir.mkdir(exist_ok=True)
        self._load_metadata()

    def _load_metadata(self) -> None:
        meta_path = self.save_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                self.metadata = json.load(fh)
        else:
            self.metadata = {
                "version": "1.0.0",
                "loaded_experts": [],
            }

    def _save_metadata(self) -> None:
        with open(self.save_dir / "metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)

    def save_expert(
        self,
        foam_code: int,
        expert_module: nn.Module,
        encoder: Optional[nn.Module] = None,
        router: Optional[nn.Module] = None,
    ) -> str:
        """Save an expert to disk.

        Parameters
        ----------
        foam_code: Integer code (0–7).
        expert_module: FoamExpertMLP instance to save.
        encoder: Optional SharedEncoder (save only once; overwrites).
        router: Optional FoamRouter (save only once; overwrites).

        Returns
        -------
        Path string of the saved expert file.
        """
        _check_torch()

        foam_type = FOAM_CODE_NAMES.get(foam_code, f"foam_{foam_code}")
        expert_path = self.expert_dir / f"{foam_type}.pt"

        meta = ExpertMeta(
            foam_type=foam_type,
            foam_code=foam_code,
        )
        torch.save(
            {
                "state_dict": expert_module.state_dict(),
                "metadata": meta.to_dict(),
            },
            str(expert_path),
        )

        if foam_type not in self.metadata.get("loaded_experts", []):
            self.metadata.setdefault("loaded_experts", []).append(foam_type)
            self._save_metadata()

        # Save encoder / router if provided
        if encoder is not None:
            torch.save(expert_module.state_dict() if hasattr(expert_module, 'state_dict') else {},
                       str(self.save_dir / "shared_encoder.pt"))
        if router is not None:
            torch.save(router.state_dict(), str(self.save_dir / "router.pt"))

        logger.info("Saved foam expert %s (code %d) to %s", foam_type, foam_code, expert_path)
        return str(expert_path)

    def load_expert(
        self,
        foam_type_or_code: str | int,
        device: str = "cpu",
    ) -> tuple[FoamExpertMLP, ExpertMeta]:
        """Load a single expert by foam type name or code.

        Parameters
        ----------
        foam_type_or_code: Foam type name (str) or integer code (0–7).
        device: Torch device.

        Returns
        -------
        Tuple of (FoamExpertMLP instance, ExpertMeta).
        """
        _check_torch()

        if isinstance(foam_type_or_code, int):
            foam_type = FOAM_CODE_NAMES.get(foam_type_or_code, str(foam_type_or_code))
        else:
            foam_type = foam_type_or_code
            if foam_type not in FOAM_CODE_NAMES.values():
                raise ValueError(
                    f"unknown foam type {foam_type!r}; "
                    f"valid: {', '.join(ALL_FOAM_NAMES)}"
                )

        expert_path = self.expert_dir / f"{foam_type}.pt"
        if not expert_path.exists():
            raise FileNotFoundError(
                f"No saved expert found for {foam_type!r} at {expert_path}. "
                f"Save it first with FoamExpertLoader.save_expert()."
            )

        payload = torch.load(str(expert_path), map_location=device, weights_only=False)
        state_dict = payload["state_dict"]
        meta = ExpertMeta.from_dict(payload["metadata"])

        # Determine hidden/output dims from state dict
        params = list(state_dict.values())
        if params:
            weight = params[0]
            if weight.ndim == 2:
                hidden_dim = weight.shape[1]
            else:
                hidden_dim = 64  # fallback
        else:
            hidden_dim = 64

        expert = FoamExpertMLP(hidden_dim=hidden_dim)
        expert.load_state_dict(state_dict)
        expert.to(device)
        expert.eval()

        return expert, meta

    def load_all(
        self, device: str = "cpu"
    ) -> tuple[nn.ModuleList, Optional[SharedEncoder], Optional[FoamRouter]]:
        """Load all saved experts, plus encoder and router if present.

        Returns
        -------
        (experts, encoder, router):
            - experts: nn.ModuleList of FoamExpertMLP instances.
            - encoder: SharedEncoder or None (if not saved).
            - router: FoamRouter or None (if not saved).
        """
        _check_torch()

        experts = nn.ModuleList()
        for foam_type in ALL_FOAM_NAMES:
            expert_path = self.expert_dir / f"{foam_type}.pt"
            if not expert_path.exists():
                logger.warning(
                    "No expert saved for %s — creating empty placeholder", foam_type
                )
                continue

            try:
                expert, _ = self.load_expert(foam_type, device=device)
                experts.append(expert)
            except Exception as e:
                logger.warning("Failed to load expert %s: %s", foam_type, e)

        # Load encoder
        encoder = None
        enc_path = self.save_dir / "shared_encoder.pt"
        if enc_path.exists():
            try:
                payload = torch.load(str(enc_path), map_location=device, weights_only=False)
                state_dict = payload if isinstance(payload, dict) else payload.get("state_dict", {})
                if state_dict:
                    encoder = SharedEncoder(input_dim=24)
                    encoder.load_state_dict(state_dict)
                    encoder.to(device)
                    encoder.eval()
            except Exception as e:
                logger.warning("Failed to load encoder: %s", e)

        # Load router
        router = None
        router_path = self.save_dir / "router.pt"
        if router_path.exists():
            try:
                state_dict = torch.load(str(router_path), map_location=device, weights_only=True)
                if state_dict:
                    router = FoamRouter(input_dim=32, n_experts=8)
                    router.load_state_dict(state_dict)
                    router.to(device)
                    router.eval()
            except Exception as e:
                logger.warning("Failed to load router: %s", e)

        return experts, encoder, router

    def has_expert(self, foam_type_or_code: str | int) -> bool:
        """Check whether an expert is saved for the given foam type."""
        if isinstance(foam_type_or_code, int):
            foam_type = FOAM_CODE_NAMES.get(foam_type_or_code, str(foam_type_or_code))
        else:
            foam_type = foam_type_or_code
        return (self.expert_dir / f"{foam_type}.pt").exists()

    def available_experts(self) -> list[str]:
        """Return list of foam type names that have saved experts."""
        return sorted([
            p.stem for p in self.expert_dir.glob("*.pt")
        ])


# ---------------------------------------------------------------------------
# Convenience: full MoE model assembly
# ---------------------------------------------------------------------------

class FoamMoEModel(nn.Module):
    """Complete foam-type Mixture of Experts model.

    Combines a shared encoder, foam-type router, and per-foam experts into
    a single forward-pass module.

    Usage
    -----

    >>> encoder = SharedEncoder(input_dim=24)
    >>> experts = create_experts(n_experts=8)
    >>> router = FoamRouter(input_dim=32, n_experts=8)
    >>> model = FoamMoEModel(encoder, experts, router)
    >>> # Forward: features (N,24), foam_codes (N,) or None
    >>> out = model(features, foam_codes=foam_codes)  # (N, 21)
    """

    def __init__(
        self,
        encoder: nn.Module,
        experts: nn.ModuleList,
        router: nn.Module,
    ):
        super().__init__()
        self.shared_encoder = encoder
        self.experts = experts
        self.router = router

    def forward(
        self,
        x: torch.Tensor,
        foam_codes: Optional[torch.Tensor] = None,
        top_k: int = 2,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: (N, input_dim) feature vectors.
        foam_codes: (N,) int tensor or None.
        top_k: Number of experts to blend (default 2).

        Returns
        -------
        (N, 21) stiffness targets.
        """
        encoded = self.shared_encoder(x)  # (N, 32)
        gates, _ = self.router(
            encoded, top_k=top_k, foam_codes=foam_codes
        )

        # Expert outputs: (N, n_experts, 21)
        expert_outs = torch.stack(
            [expert(encoded) for expert in self.experts], dim=1
        )

        # Weighted sum: (N, 1, 21) * (N, n_experts, 21) → (N, 21)
        return (gates.unsqueeze(2) * expert_outs).sum(dim=1)