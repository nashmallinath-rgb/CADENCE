"""Pydantic loaders for YAML configs. No optimization logic."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class ModelRoles(BaseModel):
    generator: str
    reflector: str
    cheap_judge: str
    strong_judge: str
    second_family_judge: str
    image_gen: str
    tts: str
    asr_or_audio_judge: str


class ModelsConfig(BaseModel):
    roles: ModelRoles


class BudgetConfig(BaseModel):
    max_usd_per_run: float = Field(gt=0)
    max_calls_per_run: int = Field(gt=0)
    max_tokens_per_run: int = Field(gt=0)


def load_models_config(path: Path | None = None) -> ModelsConfig:
    target = path or (CONFIGS / "models.yaml")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(data)


def load_budget_config(path: Path | None = None) -> BudgetConfig:
    target = path or (CONFIGS / "budget.yaml")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    return BudgetConfig.model_validate(data)
