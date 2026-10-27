"""The only file an agent writes for a datasheet."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Prose(BaseModel):
    """Limits are the customer-sheet budget. Extra keys are an error."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str = Field(max_length=400)
    intended_use: str = Field(max_length=160)
    limitations: str = Field(max_length=160)
    notes: str = ""
