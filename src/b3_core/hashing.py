"""Canonical case hash. The package version is not part of the key."""

from __future__ import annotations

import hashlib
import json

from b3_core.models import CaseInput

CACHE_SCHEMA = 1


def canonical_json(case: CaseInput) -> str:
    """Stable JSON of a validated case. Key order and ``_note`` keys are gone."""
    payload = case.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def case_hash(case: CaseInput, *, backend: str) -> str:
    """SHA-256 of schema, resolved backend name, and canonical JSON."""
    body = f"b3_core-case/{CACHE_SCHEMA}/{backend}/{canonical_json(case)}"
    return hashlib.sha256(body.encode()).hexdigest()
