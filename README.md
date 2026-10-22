# CADENCE

CADENCE is a solo research prototype for **automated, closed-loop prompt optimization**. It jointly optimizes prompts on three objectives (quality, token/API cost, prompt length) and keeps a **Pareto-optimal archive**.

Pipeline (not implemented yet): Input → Prompt Generator (DSPy, reflective evolution) → Cost-aware Eval Cascade → LLM Judge → Judge Validator (kappa / ICC / bias; flag if kappa < 0.60) → Multi-Objective Evaluator → Pareto Archive → loop while budget remains → held-out Benchmarking (exact-match + dual-judge).

Four modality streams share one orchestrator and one Pareto schema: (1) text/reasoning, (2) code and web UI, (3) image generation, (4) audio/TTS.

LLM access is **Gemini and OpenAI (via LiteLLM; Groq optional)**, always through LiteLLM.

**Status:** phase-0 scaffolding only. No optimization logic yet.

## Setup (Windows / PowerShell)

Python **3.12** is required (`py -3.12`). Do not use 3.14.

```powershell
cd path\to\CADENCE
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` and `OPENAI_API_KEY`. Do not commit `.env`.

Smoke-test providers (prints OK/FAIL and model names, never key values):

```powershell
python -m cadence.cli check-setup
# or
python scripts\check_setup.py
```

Tests (mock client, no network):

```powershell
python -m pytest
```

## Folder map

```
cadence/           package
  llm/             LiteLLM wrapper, cache, retry, mock client
  cost/            token counting and pricing
  pareto/          non-dominated sorting and archive
  judge/           rubric judge, ensembling
  cascade/         cheap → strong routing
  validator/       kappa, ICC, bias checks
  orchestrator/    DSPy reflective loop
  tasks/           Task interface; text/, code_web/, image/, audio/
  tracking/        MLflow
  benchmark/       held-out eval, dual judge
  cli.py
configs/           models.yaml, budget.yaml
data/              datasets (raw/ is gitignored)
scripts/           check_setup.py
tests/             pytest, mock-only
docs/              notes
```

## Phase checklist

- [x] Phase 0 — repo hygiene, package stubs, configs, venv, check-setup
- [x] Phase 1 — LiteLLM wrapper (cache, retry, throttle, budget)
- [x] Phase 2 — cost accounting and pricing table
- [x] Phase 3 — Pareto archive / non-dominated sorting
- [x] Phase 4 — judges, cascade, validator
- [x] Phase 5 — DSPy orchestrator loop
- [x] Phase 6 — modality tasks (text, code/web, image, audio)
- [x] Phase 7 — MLflow tracking
- [x] Phase 8 — held-out benchmark + dual-judge
- [ ] Phase 9 — first closed-loop run on a tiny budget
