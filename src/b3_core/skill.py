"""Agent skill document shipped inside the b3_core package."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def skill_path() -> Path:
    """Absolute path to the packaged ``SKILL.md``."""
    return Path(resources.files("b3_core") / "SKILL.md")


def read_skill() -> str:
    """Return the full ``SKILL.md`` text."""
    return skill_path().read_text(encoding="utf-8")