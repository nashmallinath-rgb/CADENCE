"""Pydantic contracts shared by every CADENCE module.

Names and fields follow docs/ARCHITECTURE.md section 2. Do not rename them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Resource usage of one call, or the sum of several calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    units: dict[str, float] = Field(default_factory=dict)  # e.g. {"images": 1, "audio_seconds": 4.2}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        units = dict(self.units)
        for key, value in other.units.items():
            units[key] = units.get(key, 0.0) + value
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            usd=self.usd + other.usd,
            units=units,
        )


class Example(BaseModel):
    id: str
    input: dict[str, Any]
    reference: Any | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class GenRequest(BaseModel):
    role: str  # key of configs/models.yaml roles
    messages: list[dict[str, Any]]
    params: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    kind: Literal["text", "code", "image", "audio"]
    text: str | None = None
    path: str | None = None
    usage: Usage = Field(default_factory=Usage)


class Score(BaseModel):
    quality: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    judged: bool = False
    escalated: bool = False
    usage: Usage = Field(default_factory=Usage)


class Candidate(BaseModel):
    id: str
    template: str
    parent_ids: list[str] = Field(default_factory=list)
    generation: int = 0


class Objectives(BaseModel):
    """Quality is maximized; cost and length are minimized."""

    quality: float = Field(ge=0.0, le=1.0)
    cost_usd_per_1k: float = Field(ge=0.0)
    length_tokens: int = Field(ge=0)


class Evaluated(BaseModel):
    candidate: Candidate
    objectives: Objectives
    n_examples: int
    escalation_ratio: float = 0.0
