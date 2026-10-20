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


class CascadeConfig(BaseModel):
    tau: float = Field(default=0.70, ge=0.0, le=1.0)
    delta: float = Field(default=2.0, ge=0.0)
    cheap_trials: int = Field(default=3, ge=1)
    strong_trials: int = Field(default=3, ge=1)
    kappa_min: float = Field(default=0.60, ge=-1.0, le=1.0)  # judge flagged below this


class LLMSettings(BaseModel):
    cache_dir: str = ".cache/llm"
    max_retries: int = Field(default=4, ge=0)
    est_output_tokens: int = Field(default=512, ge=0)
    rpm: dict[str, float] = Field(default_factory=dict)


class RunConfig(BaseModel):
    seed: int = 0
    max_gen: int = Field(default=8, ge=1)
    children_per_gen: int = Field(default=4, ge=1)
    minibatch: int = Field(default=16, ge=1)
    max_archive: int = Field(default=20, ge=1)
    final_sample: int = Field(default=64, ge=1)
    workers: int = Field(default=4, ge=1)  # parallel example evaluations (provider rpm still throttles)
    parents_per_gen: int = Field(default=2, ge=1)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_cascade_config(path: Path | None = None) -> CascadeConfig:
    target = path or (CONFIGS / "cascade.yaml")
    return CascadeConfig.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")))


def load_run_config(path: Path | None = None) -> RunConfig:
    target = path or (CONFIGS / "run.yaml")
    return RunConfig.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")))
