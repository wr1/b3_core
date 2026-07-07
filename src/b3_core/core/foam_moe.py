"""High-level foam-type Mixture of Experts surrogate for b3_core.

This module provides the complete MoE pipeline for foam / core type routing:
feature engineering, training, inference, save/load, and integration with
the b3_core cprop pipeline.

Architecture summary
====================

The foam MoE uses the same 24-dim feature vector as the weave MoE in
b3_micromech but replaces the 8-weave one-hot with 8-foam-type codes:

    [0-7]   constituent features (Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)
    [8]     Vf duplicate (surrogate alignment)
    [9]     normalised density (core density / max density)
    [10]    relative density (core density / foam material density)
    [11]    normalised cell size (cell_size / 2mm)
    [12]    normalised kerf depth (kerf_depth / thickness → placeholder)
    [13]    normalised kerf spacing
    [14]    normalised curvature (|curvature| * 0.1)
    [15]    pad (always 0.0)
    [16-23] one-hot foam type code (8 elements)

The foam type code determines which expert(s) handle the prediction:

    0  pvc_foam_high   — PVC high-density (Divinycell H/HDW)
    1  pvc_foam_med    — PVC medium-density (Divinycell FM)
    2  pmma_foam       — PMMA (Rohacell)
    3  pet_foam        — PET (Divinagard)
    4  aramid_honeycomb — Nomex honeycomb
    5  balsa_foam      — Balsa wood core
    6  bamboo_foam     — Bamboo fiber core
    7  generic_foam    — Fallback / unknown

Each expert is a 3-layer MLP (64→64→21) trained on foam-specific FEA data.
A shared encoder (24→64→64→32) learns universal constituent representations,
and a soft router selects expert(s) based on input features and foam code.

Usage
=====

Training from FEA data:

>>> from b3_core.core.foam_moe import FoamMoE
>>> features = ...  # (N, 24) arrays
>>> stiffness = ...  # (N, 6, 6) arrays
>>> foam_codes = ...  # (N,) int array of foam codes
>>> moe = FoamMoE.train(features, stiffness, foam_codes, save_dir="./foam_moe_models")
>>> pred = moe.predict(features, foam_codes=foam_codes)  # (N, 6, 6)
>>> moe.save("./foam_moe_models/checkpoint.pt")

Inference (loaded from disk):

>>> moe = FoamMoE.load("./foam_moe_models/checkpoint.pt", device="cpu")
>>> single_pred = moe.predict_single(feature_vector, foam_type="pvc_foam_high")

Integration with b3_core homogenisation:

>>> from b3_core import homogenize
>>> from b3_core.core.foam_moe import FoamMoE, foam_type_from_material
>>> moe = FoamMoE.load("./foam_moe_models/latest.pt")
>>> case = load_case("examples/simple.json")
>>> foam_type = foam_type_from_material(case["core"])
>>> feature_vec = build_foam_feature_vector(..., foam_type=foam_type)
>>> pred = moe.predict_single(feature_vec, foam_type=foam_type)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from b3_core.core.foam_registry import (
    ALL_FOAM_NAMES,
    DEFAULT_FEATURE_BOUNDS,
    FOAM_CODE_NAMES,
    build_foam_feature_vector,
    build_feature_matrix_batch,
    encode_foam_code,
    feature_bounds_from_training,
    relative_frobenius_error,
    stiffness_to_targets,
    targets_to_stiffness,
    _transform_features_for_regression,
)
from b3_core.core.foam_experts import (
    FoamExpertMLP,
    FoamExpertLoader,
    FoamMoEModel,
    SharedEncoder,
    create_experts,
    FoamRouter,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch data utilities
# ---------------------------------------------------------------------------

class _CheckDataset(Dataset):
    """PyTorch Dataset for foam MoE training.

    Stores (features, targets, foam_codes) tensors.
    """

    def __init__(
        self,
        features: NDArray[np.float32],
        targets: NDArray[np.float32],
        foam_codes: Optional[NDArray[np.int64]] = None,
    ):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        if foam_codes is not None:
            self.foam_codes = torch.as_tensor(foam_codes, dtype=torch.int64)
        else:
            self.foam_codes = None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "features": self.features[idx],
            "targets": self.targets[idx],
        }
        if self.foam_codes is not None:
            item["foam_codes"] = self.foam_codes[idx]
        return item


class _WeightedDataset(_CheckDataset):
    """Dataset that returns per-sample weights alongside features/targets."""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        foam_codes: torch.Tensor,
        sample_weights: NDArray[np.float64] | None = None,
    ):
        super().__init__(features.cpu().numpy(), targets.cpu().numpy(), foam_codes.cpu().numpy())
        self.sample_weights = (
            torch.as_tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.sample_weights is not None:
            item["weight"] = self.sample_weights[idx]
        return item


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _torch_device(device: str) -> torch.device:
    """Return a torch.device, falling back to cpu if device is invalid."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_epoch(
    model: "FoamMoE",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 0.01,
) -> dict[str, float]:
    """Train for one epoch. Returns metric dict."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        feats = batch["features"].to(device)
        tgts = batch["targets"].to(device)
        codes = batch.get("foam_codes", None)
        if codes is not None:
            codes = codes.to(device)

        optimizer.zero_grad()
        pred = model(feats, foam_codes=codes)

        # MSE loss
        mse_loss = F.mse_loss(pred, tgts)

        # Entropy regularisation (encourage diversity in gating)
        if codes is None:
            # Only compute entropy when not using hard mask
            encoded = model.shared_encoder(feats)
            gates, _ = model.router(encoded, top_k=model.top_k, foam_codes=None)
            entropy = -(gates * torch.log(gates + 1e-9)).sum(dim=1).mean()
            loss = mse_loss + kl_weight * entropy
        else:
            loss = mse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * feats.shape[0]
        n_samples += feats.shape[0]

    return {"train_loss": total_loss / max(n_samples, 1)}


def _evaluate(
    model: "FoamMoE",
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model. Returns metric dict."""
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_tgts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            tgts = batch["targets"].to(device)
            codes = batch.get("foam_codes", None)
            if codes is not None:
                codes = codes.to(device)

            pred = model(feats, foam_codes=codes)
            all_preds.append(pred.cpu())
            all_tgts.append(tgts.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    tgts = torch.cat(all_tgts, dim=0).numpy()

    # MSE
    mse = float(np.mean((preds - tgts) ** 2))

    # Relative Frobenius error (convert targets back to stiffness)
    err = relative_frobenius_error(
        targets_to_stiffness(preds), targets_to_stiffness(tgts)
    )
    return {
        "val_mse": mse,
        "val_mean_frob": float(err.mean()),
        "val_max_frob": float(err.max()),
    }


# ---------------------------------------------------------------------------
# Feature bounds
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_BOUNDS: NDArray[np.float64] = np.array([
    [0.0, 1.0],          # [0] Vf
    [0.5e9, 15.0e9],     # [1] E_m (Pa)
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
], dtype=float)


# ---------------------------------------------------------------------------
# Public FoamMoE wrapper (dataclass + high-level API)
# ---------------------------------------------------------------------------

@dataclass
class FoamMoE:
    """High-level foam-type MoE surrogate wrapping a PyTorch module.

    Provides training, inference, save/load, and a ``SurrogateModel``
    adapter for use within the b3_core pipeline.

    Attributes
    ----------
    module: PyTorch FoamMoEModel.
    feature_bounds: Per-feature [min, max] bounds, shape (16, 2).
    top_k: Default number of experts to blend (2).
    log_modulus_features: Whether to log-transform modulus features.
    device: Torch device.
    _feature_scaler: Per-feature StandardScaler.
    _target_scaler: Per-target StandardScaler.
    _feature_input_dim: Input dimension (24).
    """

    module: Any = field(default=None, repr=False)  # FoamMoEModel
    feature_bounds: NDArray[np.float64] = field(default=DEFAULT_FEATURE_BOUNDS)
    top_k: int = 2
    log_modulus_features: bool = True
    device: str = "cpu"
    _feature_scaler: Any = field(default=None, repr=False)
    _target_scaler: Any = field(default=None, repr=False)
    _feature_input_dim: int = 24

    # -------------------------------------------------------------------
    # Class factory: train from scratch
    # -------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        features: NDArray[np.float64],
        stiffness: NDArray[np.float64],
        *,
        foam_codes: Optional[NDArray[np.int64]] = None,
        hidden_dim: int = 64,
        encoder_dims: tuple[int, ...] = (64, 64, 32),
        top_k: int = 2,
        n_experts: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 1000,
        patience: int = 100,
        batch_size: int = 64,
        log_modulus_features: bool = True,
        seed: int = 0,
        device: str = "cpu",
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> "FoamMoE":
        """Train a new foam MoE from (features, stiffness) data.

        Parameters
        ----------
        features: (N, 24) float array — full MoE feature vectors.
        stiffness: (N, 6, 6) float array — homogenised stiffness tensors.
        foam_codes: (N,) int array of foam codes (0-7), or None.
            When None, the router uses soft top-k blending.
        hidden_dim: Hidden dimension for encoder and experts (default 64).
        top_k: Number of experts to blend by default (default 2).
        n_experts: Number of expert MLPs (default 8).
        learning_rate: AdamW learning rate (default 1e-4).
        weight_decay: AdamW weight decay (default 1e-5).
        max_epochs: Maximum training epochs (default 1000).
        patience: Early stopping patience (default 100).
        batch_size: DataLoader batch size (default 64).
        log_modulus_features: Log-transform modulus features (default True).
        seed: Random seed (default 0).
        device: Torch device (default "cpu").
        val_split: Fraction for validation (default 0.1).
        test_split: Fraction for test (default 0.1).

        Returns
        -------
        Trained FoamMoE instance.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        x = np.asarray(features, dtype=float)
        c = np.asarray(stiffness, dtype=float)

        # Handle single sample
        if x.ndim == 1:
            x = x[None, :]
        if c.ndim == 2:
            c = c[None, :, :]

        if c.ndim != 3 or c.shape[1:] != (6, 6):
            raise ValueError(f"stiffness must have shape (N, 6, 6), got {c.shape}")

        N = x.shape[0]
        if N < 10:
            raise ValueError(f"need at least 10 training samples, got {N}")

        # Transform features
        x_model = _transform_features_for_regression(x, log_modulus=log_modulus_features)

        # Scale features
        feature_scaler = StandardScaler()
        x_scaled = feature_scaler.fit_transform(x_model)

        # Targets
        y = stiffness_to_targets(c)
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y)

        # Foam codes
        if foam_codes is not None:
            fc = np.asarray(foam_codes, dtype=np.int64).ravel()
            if fc.shape[0] != N:
                raise ValueError(f"foam_codes length ({fc.shape[0]}) != features ({N})")
        else:
            fc = None

        # Split: train / val / test
        if fc is not None:
            classes = np.unique(fc)
            train_idx, val_idx, test_idx = [], [], []
            for cls in classes:
                cls_idx = np.where(fc == cls)[0]
                n = len(cls_idx)
                n_val = max(1, int(n * val_split / (val_split + test_split)))
                n_test = max(1, int(n * test_split / (val_split + test_split)))
                perm = rng.permutation(n)
                train_idx.extend(cls_idx[perm[n_val + n_test:]].tolist())
                val_idx.extend(cls_idx[perm[:n_val]].tolist())
                test_idx.extend(cls_idx[perm[n_val:n_val + n_test]].tolist())
        else:
            indices = rng.permutation(N)
            n_val = max(1, int(N * val_split))
            n_test = max(1, int(N * test_split))
            val_idx = indices[:n_val]
            test_idx = indices[n_val:n_val + n_test]
            train_idx = indices[n_val + n_test:]

        # Build datasets
        dev = _torch_device(device)
        train_dataset = _CheckDataset(
            x_scaled[train_idx], y_scaled[train_idx]
        )
        if fc is not None:
            train_dataset.foam_codes = torch.tensor(
                fc[train_idx], dtype=torch.int64, device=dev
            )
        val_dataset = _CheckDataset(x_scaled[val_idx], y_scaled[val_idx])
        if fc is not None:
            val_dataset.foam_codes = torch.tensor(
                fc[val_idx], dtype=torch.int64, device=dev
            )
        test_dataset = _CheckDataset(x_scaled[test_idx], y_scaled[test_idx])
        if fc is not None:
            test_dataset.foam_codes = torch.tensor(
                fc[test_idx], dtype=torch.int64, device=dev
            )

        # Build model
        input_dim = x_scaled.shape[1]

        encoder = SharedEncoder(input_dim, encoder_dims)
        experts = create_experts(hidden_dim=hidden_dim, n_experts=n_experts)
        router = FoamRouter(encoder_dims[-1], n_experts)

        moe_model = FoamMoEModel(encoder, experts, router)
        moe_model.to(dev)

        # Optimizer
        optimizer = torch.optim.AdamW(
            moe_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        patience_counter = 0

        for epoch in range(max_epochs):
            train_metrics = _train_epoch(
                moe_model,
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                optimizer,
                dev,
            )

            val_metrics = _evaluate(
                moe_model,
                DataLoader(val_dataset, batch_size=batch_size),
                dev,
            )

            val_loss = val_metrics["val_mse"]

            if epoch % 50 == 0 or epoch == max_epochs - 1:
                logger.info(
                    f"Epoch {epoch:5d} | train_loss: {train_metrics['train_loss']:.6f} "
                    f"| val_mse: {val_loss:.6f} | val_frob: {val_metrics['val_mean_frob']:.4%}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in moe_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (best val_mse: {best_val_loss:.6f})"
                    )
                    break

        # Load best state
        moe_model.load_state_dict(best_state)

        feat_bounds = feature_bounds_from_training(x)

        return cls(
            module=moe_model,
            feature_bounds=feat_bounds,
            top_k=top_k,
            log_modulus_features=log_modulus_features,
            device=device,
            _feature_scaler=feature_scaler,
            _target_scaler=target_scaler,
            _feature_input_dim=input_dim,
        )

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def predict(
        self,
        features: NDArray[np.float64],
        foam_codes: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.float64]:
        """Predict stiffness tensors.

        Parameters
        ----------
        features: (N, D) or (D,) feature vector(s).
        foam_codes: (N,) int array of foam codes, or None.
            When provided, uses deterministic expert routing by foam type.

        Returns
        -------
        (N, 6, 6) or (6, 6) symmetric stiffness tensor(s).
        """
        _check_torch()

        x = np.asarray(features, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]

        x_model = _transform_features_for_regression(
            x, log_modulus=self.log_modulus_features
        )

        if self._feature_scaler is not None:
            x_scaled = self._feature_scaler.transform(x_model)
        else:
            x_scaled = x_model

        dev = _torch_device(self.device)
        x_t = torch.tensor(x_scaled, dtype=torch.float32, device=dev)

        foam_t = None
        if foam_codes is not None:
            foam_t = torch.tensor(
                np.asarray(foam_codes, dtype=np.int64), dtype=torch.int64
            ).to(dev)

        self.module.eval()
        with torch.no_grad():
            preds_21 = self.module(x_t, foam_codes=foam_t, top_k=self.top_k)

        preds_np = preds_21.cpu().numpy()
        if self._target_scaler is not None:
            preds_np = self._target_scaler.inverse_transform(preds_np)

        result = targets_to_stiffness(preds_np)

        if single:
            return result[0]
        return result

    def predict_with_gate(
        self,
        features: NDArray[np.float64],
        foam_codes: Optional[NDArray[np.int64]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict stiffness with gate weights for interpretability.

        Returns
        -------
        stiffness: (N, 6, 6) predicted stiffness tensor(s).
        gates: (N, 8) gate weights per expert.
        """
        _check_torch()

        x = np.asarray(features, dtype=float)
        single = x.ndim == 1
        if single:
            x = x[None, :]

        x_model = _transform_features_for_regression(
            x, log_modulus=self.log_modulus_features
        )

        if self._feature_scaler is not None:
            x_scaled = self._feature_scaler.transform(x_model)
        else:
            x_scaled = x_model

        dev = _torch_device(self.device)
        x_t = torch.tensor(x_scaled, dtype=torch.float32, device=dev)

        foam_t = None
        if foam_codes is not None:
            foam_t = torch.tensor(
                np.asarray(foam_codes, dtype=np.int64), dtype=torch.int64
            ).to(dev)

        self.module.eval()
        with torch.no_grad():
            encoded = self.module.shared_encoder(x_t)
            gates, expert_indices = self.module.router(
                encoded, top_k=self.top_k, foam_codes=foam_t
            )
            preds_21 = self.module(x_t, foam_codes=foam_t, top_k=self.top_k)

        preds_np = preds_21.cpu().numpy()
        if self._target_scaler is not None:
            preds_np = self._target_scaler.inverse_transform(preds_np)

        stiffness = targets_to_stiffness(preds_np)
        gates_np = gates.cpu().numpy()

        if single:
            return stiffness[0], gates_np[0]
        return stiffness, gates_np

    def predict_single(
        self,
        features: NDArray[np.float64],
        foam_type: Optional[str] = None,
    ) -> NDArray[np.float64]:
        """Predict a single feature vector.

        Parameters
        ----------
        features: 24-dim feature vector.
        foam_type: Optional foam type name for interpretability logging.

        Returns
        -------
        (6, 6) stiffness tensor.
        """
        result = self.predict(features)
        if result.ndim == 3:
            return result[0]
        return result

    # -------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the full foam MoE surrogate to disk.

        Saves:
        - model state dict (PyTorch)
        - feature bounds
        - scaler parameters
        - metadata (top_k, log_modulus_features, input_dim)

        Parameters
        ----------
        path: Path to save to (.pt or .pth extension recommended).
        """
        _check_torch()

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.module.state_dict()

        scalers = {
            "feature_scaler_mean": self._feature_scaler.mean_.tolist()
            if self._feature_scaler is not None else None,
            "feature_scaler_scale": self._feature_scaler.scale_.tolist()
            if self._feature_scaler is not None else None,
            "target_scaler_mean": self._target_scaler.mean_.tolist()
            if self._target_scaler is not None else None,
            "target_scaler_scale": self._target_scaler.scale_.tolist()
            if self._target_scaler is not None else None,
        }

        metadata = {
            "top_k": self.top_k,
            "log_modulus_features": self.log_modulus_features,
            "feature_input_dim": self._feature_input_dim,
            "n_experts": len(self.module.experts),
            "feature_bounds": self.feature_bounds.tolist(),
            "version": "1.0.0",
        }

        torch.save(
            {
                "state_dict": state_dict,
                "scalers": scalers,
                "metadata": metadata,
            },
            str(out),
        )

        logger.info("Saved foam MoE surrogate to %s", out)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "FoamMoE":
        """Load a foam MoE surrogate from disk.

        Parameters
        ----------
        path: Path to saved model (.pt or .pth).
        device: Device to load on (default "cpu").

        Returns
        -------
        Loaded FoamMoE instance.
        """
        _check_torch()

        from sklearn.preprocessing import StandardScaler

        payload = torch.load(str(path), map_location=device, weights_only=False)

        state_dict = payload["state_dict"]
        scalers = payload["scalers"]
        metadata = payload["metadata"]

        input_dim = metadata["feature_input_dim"]
        n_experts = metadata["n_experts"]
        top_k = metadata["top_k"]

        encoder_dims = (64, 64, 32)
        encoder = SharedEncoder(input_dim, encoder_dims)
        experts = create_experts(n_experts=n_experts)
        router = FoamRouter(encoder_dims[-1], n_experts)

        moe_model = FoamMoEModel(encoder, experts, router)
        moe_model.load_state_dict(state_dict)
        moe_model.to(device)

        # Reconstruct scalers
        feature_scaler = None
        if scalers["feature_scaler_mean"] is not None:
            fs = StandardScaler()
            fs.mean_ = np.array(scalers["feature_scaler_mean"])
            fs.scale_ = np.array(scalers["feature_scaler_scale"])
            fs.var_ = fs.scale_ ** 2
            fs.n_features_in_ = len(fs.mean_)
            feature_scaler = fs

        target_scaler = None
        if scalers["target_scaler_mean"] is not None:
            ts = StandardScaler()
            ts.mean_ = np.array(scalers["target_scaler_mean"])
            ts.scale_ = np.array(scalers["target_scaler_scale"])
            ts.var_ = ts.scale_ ** 2
            ts.n_features_in_ = len(ts.mean_)
            target_scaler = ts

        feature_bounds = np.array(metadata["feature_bounds"], dtype=float)

        return cls(
            module=moe_model,
            feature_bounds=feature_bounds,
            top_k=top_k,
            log_modulus_features=metadata["log_modulus_features"],
            device=device,
            _feature_scaler=feature_scaler,
            _target_scaler=target_scaler,
            _feature_input_dim=input_dim,
        )

    # -------------------------------------------------------------------
    # SurrogateModel adapter (b3_tex compatible)
    # -------------------------------------------------------------------

    def as_predict_callable(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return a predict(features) → (6, 6) function for b3_tex.micromodels.SurrogateModel.

        Returns
        -------
        predict: Callable that maps 8-dim features to (6, 6) stiffness.
        """
        def predict(features: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.predict_single(features)
        return predict


# ---------------------------------------------------------------------------
# Helper: foam type detection from b3_core material definitions
# ---------------------------------------------------------------------------

def foam_type_from_material(
    material: dict,
) -> str:
    """Detect foam type from a b3_core CpropInput Material dict.

    Examines the material dict's fields (name, E1, rho, cell_size) and
    returns a canonical foam type name. Falls back to 'generic_foam'
    when the foam type cannot be determined.

    Parameters
    ----------
    material: dict matching the CpropInput Material structure:
        - 'name': optional material name string
        - 'E1': if present, material is orthotropic
        - 'rho': density (kg/m³)
        - 'cell_size': foam cell size specification (mm)

    Returns
    -------
    str: foam type name (one of ALL_FOAM_NAMES).

    Examples
    --------
    >>> foam_type_from_material({"E1": 0.32e9, "rho": 80.0, "name": "Divinycell H250"})
    'pvc_foam_high'
    >>> foam_type_from_material({"rho": 200.0, "E1": 2.0e9, "E2": 4.0e9})
    'balsa_foam'
    """
    name = material.get("name", "").lower()

    # Check orthotropic indicators (balsa, honeycomb, bamboo have orthotropic E)
    if material.get("E1") is not None:
        # Orthotropic — could be balsa (high E), honeycomb (low E), or bamboo
        E1 = material["E1"]
        rho = material.get("rho", 0)

        if rho > 100 and E1 > 1e9:
            return "balsa_foam"
        elif rho < 100 and E1 < 0.1e9:
            return "aramid_honeycomb"
        elif rho > 80 and E1 > 0.5e9:
            return "bamboo_foam"

    # Check by name patterns
    if "divinycell" in name or "pvc" in name:
        if "hdw" in name or "h100" in name or "h250" in name or "h45" in name or "h80" in name:
            return "pvc_foam_high"
        return "pvc_foam_med"
    elif "roham" in name or "rohamcell" in name or "rohacell" in name or "pmma" in name:
        return "pmma_foam"
    elif "divinagard" in name or "airgrid" in name or "pet" in name:
        return "pet_foam"
    elif "nomex" in name or "honeycomb" in name or "aramid" in name:
        return "aramid_honeycomb"
    elif "balsa" in name or "corewood" in name:
        return "balsa_foam"
    elif "bamboo" in name or "bam" in name:
        return "bamboo_foam"

    # Check by density heuristics (if no name match)
    rho = material.get("rho", 0)
    if rho > 150 and material.get("cell_size") is not None:
        return "pvc_foam_high"
    elif rho > 80:
        return "pvc_foam_med"
    elif rho > 50:
        return "pet_foam"
    else:
        return "generic_foam"


# ---------------------------------------------------------------------------
# Convenience: export for HuggingFace-compatible format
# ---------------------------------------------------------------------------

def save_hf_format(
    model: FoamMoE,
    output_dir: str | Path,
    foam_registry: dict[str, int] | None = None,
) -> None:
    """Save the foam MoE in HuggingFace-compatible format.

    Creates:
    - model.pt: full torch state dict + metadata
    - config.json: architecture config
    - foam_registry.json: code↔name mapping

    Parameters
    ----------
    model: Trained FoamMoE instance.
    output_dir: Directory to save to.
    foam_registry: Optional foam code↔name mapping.
        Defaults to ``foam_registry.FOAM_CODE_NAMES``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model.pt"
    if not model_path.exists():
        model.save(model_path)

    config = {
        "model_type": "foam_moe_surrogate",
        "input_dim": model._feature_input_dim,
        "output_dim": 21,
        "top_k": model.top_k,
        "log_modulus_features": model.log_modulus_features,
        "feature_bounds": model.feature_bounds.tolist(),
        "n_experts": len(model.module.experts),
        "encoder_dims": (64, 64, 32),
        "hidden_dim": 64,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    registry = foam_registry or {
        FOAM_CODE_NAMES[k]: k for k in FOAM_CODE_NAMES
    }
    (out / "foam_registry.json").write_text(json.dumps(registry, indent=2))

    logger.info("Saved HF-compatible foam MoE format to %s", out)