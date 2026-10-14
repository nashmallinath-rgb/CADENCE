# CADENCE: Phase tasks for Cursor

## How to work (read once per phase)
- One phase = one branch (`phase-N-name`) = one PR. Do not merge; do not touch `main`.
- Read only the sections of `docs/SPEC.md` and `docs/ARCHITECTURE.md` named in the phase. Don't explore the rest of the repo.
- Test first. Everything runs offline via `MockClient`. Run `pytest` before every commit. Small commits.
- Reuse existing scaffold files; extend, don't rewrite. Follow `.cursor/rules/cadence.mdc`.
- Final reply must be short: what was built, `pytest` result, `git diff main --stat`, open questions. Then stop.

Paste for each phase: **"Implement Phase N from docs/TASKS.md."**

---
## Phase 1: Core infra (`phase-1-core`) | Day 1-2
Read: ARCHITECTURE §2, llm, cost, config keys.
Build: `LLMClient` (LiteLLM wrapper, diskcache, tenacity retry, RPM throttle), `BudgetGuard`, `MockClient`, `configs/pricing.yaml` (placeholders marked VERIFY), cost functions, `cascade.yaml` and `run.yaml` with defaults, pydantic contracts in `cadence/schemas.py`.
Tests: cache hit avoids a second call; budget raises `BudgetExceeded` before overshoot; retry on transient error; mock is deterministic; `cost_per_1k` matches the formula by hand on a fixture; `template_length` stable.

## Phase 2: Pareto (`phase-2-pareto`) | Day 2
Read: SPEC §2, ARCHITECTURE pareto.
Build: dominance, non-dominated sort, `ParetoArchive` (add/prune/serialize/hypervolume).
Tests: hand-made 3D point sets with known fronts; dominated points rejected; duplicates handled; JSON round-trip; prune keeps extremes.

## Phase 3: Judge, cascade, validator (`phase-3-judge`) | Day 3-4
Read: ARCHITECTURE judge, cascade, validator; SPEC O2, O3.
Build: `Judge` (JSON output, trials, reference-free), `Cascade`, `agreement.py`, `bias.py`, `calibrate.py`.
Tests (mock judge with scripted noise levels): kappa/ICC match sklearn/known values; noisy judge gets flagged at kappa < 0.60; escalation happens only below tau/above delta; auto-scored tasks never call a judge; spend accounting is correct.

## Phase 4: Task interface + text stream (`phase-4-text`) | Day 4-5
Read: ARCHITECTURE §2 Task, tasks; SPEC §3 stream 1.
Build: reconcile `tasks/base.py` with the contract; `ToyTask`; loaders for GSM8K, HotpotQA, IFBench, SST-2 (HF `datasets`, split into opt/heldout/calib by seed); exact-match/numeric-answer metrics; fixtures in `tests/fixtures/`.
Tests: loaders on fixtures; split disjointness; GSM8K answer extraction; metrics edge cases. Real downloads only in a manual `scripts/fetch_data.py`.

## Phase 5: Orchestrator + tracking (`phase-5-loop`) | Day 5-7
Read: ARCHITECTURE orchestrator, tracking.
Build: `generator.py` (reflective rewrite, mutation, crossover; DSPy optional per rules), `evaluate.py` (cascade over minibatch -> `Objectives`), `loop.py`, MLflow logger, run directory writer, CLI `run --mock`.
Tests: **e2e on ToyTask**: front size grows or hypervolume increases across generations; budget stops the loop; resume-safe run dir; deterministic with a seed. Mark `@pytest.mark.e2e`, under 60 s.
Milestone (Day 7): `python -m cadence.cli run --stream text --task toy --mock` prints a Pareto menu.

## Phase 5b: Live smoke (manual, needs keys) | Day 7-8
Fill `.env`, verify model names in `configs/models.yaml`, run `check-setup`, then one tiny real run (`--budget-usd 0.5`) on SST-2 and GSM8K. Fix what breaks; record real prices in `pricing.yaml`.

## Phase 6: Code and web UI stream (`phase-6-codeweb`) | Day 8-9
Read: SPEC §3 stream 2.
Build: `tasks/code_web/`: ~30 task specs in `data/tasks/code_web.jsonl` (id, brief, required elements/behaviours), validators (HTML parse with BeautifulSoup/html5lib, required-element checks, JS syntax check via `node --check` if available, otherwise skip with a flag), quality = weighted checklist; optional judge for design quality.
Tests: validators on good/bad HTML fixtures; scoring is deterministic; missing node degrades gracefully.

## Phase 7: Image and audio streams (`phase-7-media`) | Day 9-10
Read: SPEC §3 streams 3-4; ARCHITECTURE llm (image/speech/transcribe).
Build: `LLMClient.image/speech/transcribe` (use LiteLLM if it supports the provider; otherwise one thin adapter with `google-genai`, adding the dependency); `tasks/image` (50 concept prompts, vision-judge rubric); `tasks/audio` (50 spoken-instruction prompts, TTS, ASR round trip, WER via jiwer, chars-per-second ratio); per-unit pricing in `pricing.yaml`; media saved under the run dir.
Tests: mock media (tiny generated PNG/WAV); WER maths; ratio calc; cost with units.

## Phase 8: Benchmark and reports (`phase-8-benchmark`) | Day 11-12
Read: SPEC O4, ARCHITECTURE benchmark.
Build: `heldout.py`, `dual_judge.py` (inter-judge kappa, rank agreement, self-preference gap), `report.py`, `plots.py` (3D Pareto plus 2D projections), CLI `benchmark` and `report`.
Tests: report contains every required section on a mock run; plots written; dual-judge stats on scripted judges.
Manual: real runs on all four streams within `budget.yaml` caps.

## Phase 9: Polish (`phase-9-polish`) | Day 13-14
README (Windows quickstart, architecture diagram, how to reproduce), demo script `scripts/demo.ps1`, final cleanup, ensure `pytest` and `-m e2e` pass from a fresh clone.
