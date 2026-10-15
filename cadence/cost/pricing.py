"""Pricing table loader and USD calculation.

Prices live in configs/pricing.yaml. Every entry there is a PLACEHOLDER until you
verify it against the provider's current pricing page (see the `verify` flag).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from cadence.config import CONFIGS
from cadence.schemas import Usage


class ModelPrice(BaseModel):
    input_per_mtok: float = Field(default=0.0, ge=0.0)  # USD per 1,000,000 input tokens
    output_per_mtok: float = Field(default=0.0, ge=0.0)  # USD per 1,000,000 output tokens
    units: dict[str, float] = Field(default_factory=dict)  # USD per unit, e.g. {"images": 0.03}
    verify: bool = True  # True until the user has checked this entry against the provider


class Pricing(BaseModel):
    default: ModelPrice
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    def for_model(self, model: str) -> ModelPrice:
        """Unknown models fall back to the (deliberately pessimistic) default."""
        return self.models.get(model, self.default)

    def price_tokens(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.for_model(model)
        return input_tokens * p.input_per_mtok / 1e6 + output_tokens * p.output_per_mtok / 1e6

    def price(self, usage: Usage, model: str) -> float:
        """USD for a usage record: token cost plus any per-unit cost (images, audio seconds)."""
        p = self.for_model(model)
        total = self.price_tokens(model, usage.input_tokens, usage.output_tokens)
        for unit, count in usage.units.items():
            total += count * p.units.get(unit, 0.0)
        return total

    def unverified(self) -> list[str]:
        return sorted(name for name, p in self.models.items() if p.verify)


def load_pricing(path: Path | None = None) -> Pricing:
    target = path or (CONFIGS / "pricing.yaml")
    return Pricing.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")))
