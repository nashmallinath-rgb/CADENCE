# CADENCE: Master Implementation Handoff for Session B (Phase 7 media + Phase 8 benchmark)

Paste everything below the line into a fresh Claude Code session that has the CADENCE repo checked out. It is
self-contained. Deadline: working prototype on **Oct 28**. Today is Oct 21, so budget about 5 working days for both
phases. Do Phase 7 first, but keep Phase 8's data needs in mind (section 3).

------------------------------------------------------------------------------------------------------------------

## 0. Who you are working with and how

You are the correctness-critical builder for Phases 7 and 8 of CADENCE, a solo student project (Nash, final-year
CSE-AIML) building "An Automated Framework for Multi-Objective, Cost-Calibrated, Artifact-Resistant Prompt
Optimization". Nash is informal and wants direct, honest assessments, including pushback when something is a bad idea.

Other workers touch the same repo at the same time:
- **Continue.dev (local, Gemini)** builds Phase 6 (code/web stream: `cadence/tasks/code_web/`, `data/tasks/code_web.jsonl`)
  and **owns `cadence/benchmark/plots.py`** (3D Pareto + 2D projections) and extra edge-case tests. Spec:
  `docs/continue/PHASE6_CODEWEB_SPEC.md`. **Do not write plots.py.** Call its interface (section 4.4).
- **The coordination thread** verifies integration and will run live (keyed) checks.
You cannot reach the user's API keys or their machine. Everything you build must be fully testable offline with
`MockClient`; live runs are manual and the user does them. Where a live fact is unknown (a model name, a price),
ship a probe script plus config keys and say exactly what the user must run.

## 1. State of the repo (read these first, in this order)

1. `docs/SPEC.md` (objectives, streams, definition of done), `docs/ARCHITECTURE.md` (contracts, modules, config keys,
   CLI), `docs/TASKS.md` (Phase 7 and 8 entries), `.continue/rules/cadence.md` (house rules, they apply to you too).
2. Code that exists and works (Phases 1-5, **154 tests green offline**):
   - `cadence/schemas.py`: `Usage` (supports `+`, has `units: dict[str, float]`), `Example`, `GenRequest`, `Output`
     (`kind` text|code|image|audio, `text`, `path`, `usage`), `Score`, `Candidate`, `Objectives`, `Evaluated`.
   - `cadence/llm/`: `LLMClient.complete(req) -> (text, Usage)` with disk cache, tenacity retries, per-provider RPM
     throttle (thread-safe), `BudgetGuard` (check before call, record after; cache hits are free), `MockClient`
     (deterministic, role-aware, scriptable, optional budget). `LLMClient.from_config(budget=...)`.
   - `cadence/cost/`: `Pricing.price(usage, model)` already adds per-unit cost from `pricing.yaml` `units:`;
     `cost_per_1k(usages)`, `template_length(template)`.
   - `cadence/judge/`: `Judge(client, role, trials).score(ex, response_text, rubric) -> JudgeResult` (reference-free,
     lenient JSON parse; `.score_0_10`, `.confidence_0_1`, `.spread`, `.usage`), `PairwiseJudge`.
   - `cadence/cascade/cascade.py`: `Cascade.from_config(client, cfg, cheap_role=..., strong_role=...)`;
     `.score(task, ex, out) -> Score`; auto-scores when `task.automatic_score` is not None, else cheap judge, then
     strong judge if mean confidence < tau or spread > delta. It passes `out.text` to the judge.
   - `cadence/validator/`: `cohen_kappa(a, b, weights)`, `icc_2_1`, `binarize(scores, threshold=5.0)`,
     `position_bias`, `verbosity_bias`, `calibrate(judge, items, kappa_min)`, `calibrate_with_recheck`,
     `CalibItem`, `JudgeHealth`.
   - `cadence/orchestrator/`: `Generator` (reflector role), `Evaluator` (threaded; deployment cost = generator usage
     only), `optimize(task, client, cascade, settings, run_dir, tracker=, kappa=, resume=, log=)`, run directory
     (`config.json state.json archive.json archive_search.json trace.jsonl menu.md`). `trace.jsonl` events:
     `evaluated`, `generation` (front_size, hypervolume, best_quality, escalation_ratio, spend_usd, calls, kappa...),
     `skipped`, `stop`.
   - `cadence/tasks/`: `Task` ABC (frozen contract, see `base.py`), `ToyTask`, text tasks, `registry.TASKS`.
   - `cadence/cli.py`: `check-setup`, `smoke`, `run`. `cadence/smoke.py` already builds calibration sets and
     runs judge health checks; reuse it for the `calibrate` command.
   - `scripts/probe_models.py` / `list_models.py`: the pattern for live probes (urllib with a User-Agent, prints real errors).
3. Models today (`configs/models.yaml`): generator `gemini/gemini-3.5-flash-lite`, reflector `gemini/gemini-3.1-flash-lite`,
   cheap_judge `groq/openai/gpt-oss-20b`, strong_judge `groq/openai/gpt-oss-120b`, second_family_judge
   `groq/qwen/qwen3.8-27b`. `image_gen`, `tts`, `asr_or_audio_judge` are **unverified placeholders** (Imagen 3 etc.
   may be gone or paid). Free tiers only; Gemini ~10 RPM, Groq ~15 RPM and ~8K tokens/min per model.
4. Known facts to design around: Python 3.12 target (user's machine also ran 3.14 fine); Windows/PowerShell;
   LF endings; no torch/CLIP; tests never touch the network; no secrets anywhere.

## 2. Git and coordination rules

- Branches: `phase-7-media`, then `phase-8-benchmark`. Base: `main` if the user has merged the Phase 5 PR,
  otherwise `phase-5-loop`. Ask which in your first message if you cannot tell.
- Shared hot spots are **append-only**: `cadence/cli.py`, `cadence/tasks/registry.py`, `cadence/config.py`,
  `configs/*.yaml`, `pyproject.toml`, `requirements.txt`. Continue appends to registry and tests too.
- You may add: new files, new roles to `models.yaml` + `ModelRoles` (with defaults so old configs load), the
  `google-genai` dependency if LiteLLM cannot do Gemini TTS/image (pin the version, update both
  `pyproject.toml` and `requirements.txt`). Do not change frozen schema fields; adding fields with defaults is
  allowed only where this document says so.
- Frequent small commits, imperative subjects. The sandbox cannot push to GitHub (403), so deliver each phase as a
  **git bundle** (`git bundle create ../cadence-phase-7.bundle <base>..phase-7-media`) via `SendUserFile`, with the
  exact PowerShell apply commands (`git fetch <bundle> branch:branch`, checkout, `python -m pytest`).
  Ignore the stop-hook warning about unpushed commits while GitHub access is missing.
- Never print or request keys. Never claim a live behaviour you did not observe; label unverified things.

## 3. Execution order (do not skip steps)

**Phase 7, branch `phase-7-media`**
1. Read the files in section 1. Run `python -m pytest` (expect 154 passed) before changing anything.
2. Config groundwork: new roles, `vision_judge`, `vision_judge_strong`, `asr` (see 4.1); `ModelRoles` defaults;
   `pricing.yaml` per-unit entries with `verify: true`; tests that old and new configs both load.
3. `LLMClient.image/speech/transcribe` plus `MockClient` equivalents, media cache, budget handling (4.1). Tests first.
4. `scripts/probe_media.py` (4.1). The user runs it live to choose real model names.
5. Vision judging path through `Judge` and `Cascade` (4.2). Existing tests must pass unchanged.
6. `ImageTask` with 50 prompts (4.2). Registry append. Tests.
7. `AudioTask` with 50 prompts (4.3). Registry append. Tests.
8. CLI: let `run` pass a media directory to media tasks and choose the media cascade roles (4.5).
9. E2E: `python -m cadence.cli run --stream image --task image --mock --no-mlflow` and the same for audio finish and
   print a Pareto menu; add `@pytest.mark.e2e` tests (offline, under 60 s each).
10. Full suite, review your own diff critically, bundle, deliver.

**Phase 8, branch `phase-8-benchmark`** (start from the finished phase-7 branch; the data contract in 5.1 must be
settled before you write anything)
11. Calibrate command and loop integration (6.1).
12. `heldout.py` and the persisted outputs file (5.1-5.2).
13. `dual_judge.py` (5.3).
14. `report.py` (5.4), calling Continue's `plots.py` (5.5).
15. CLI `benchmark` and `report`; wire `run` to auto-run report at the end (5.6).
16. E2E on ToyTask and on a judged mock task, full suite, bundle, deliver.

## 4. Phase 7 specification: image and audio streams

### 4.1 LLM layer extension (`cadence/llm/`)
- Add to `LLMClient` and `MockClient`, same interface on both:
  - `image(req: GenRequest) -> tuple[bytes, Usage]` (PNG bytes; `Usage.units["images"] = 1`).
  - `speech(req: GenRequest) -> tuple[bytes, Usage]` (WAV bytes; `Usage.units["audio_seconds"]` from the real duration).
  - `transcribe(audio: bytes, req: GenRequest) -> tuple[str, Usage]` (`units["audio_seconds"]`).
  Choose the signature shapes that fit, keep them identical on both clients, and document them.
- Use LiteLLM where it supports the provider (`litellm.image_generation`, `speech`, `transcription`; Groq serves
  Whisper for ASR on the free tier). Where it does not (likely Gemini image/TTS), write **one thin adapter module** in
  `cadence/llm/` using `google-genai`. All provider imports stay inside `cadence/llm/`.
- Every media call must: use the cache (key = hash of model + request + seed; cache the bytes on disk under the
  cache dir, not in the JSON text store; cached calls return the original Usage and cost nothing), go through the
  RPM throttle and retries, and be checked by `BudgetGuard.check` **before** the call with a per-unit USD estimate
  from `pricing.yaml` (the guard currently estimates from tokens; extend `check` usage additively, do not break
  existing signatures), then `record`ed after. Thread-safe.
- `MockClient` media: deterministic tiny PNG (generate with the standard library, no Pillow dependency unless it is
  already installed) and tiny WAV (`wave` module) whose duration scales with input length; scriptable like `complete`.
  `transcribe` mock should return the spoken text with optional scripted corruption so WER tests are meaningful.
- Pricing: per-unit prices live in `pricing.yaml` under `units:` (`images`, `audio_seconds`); tokens as today. All
  new entries `verify: true`. Cost objective for media tasks = generator-stage `Usage.usd` including units.
- `scripts/probe_media.py`: for each of `image_gen`, `tts`, `asr`, `vision_judge`: one tiny live call, print the
  real error text, a PASS/FAIL line and measured latency; if several candidate model names are plausible, try each
  in turn (copy the style of `probe_models.py`). Free tier only. Deliver with instructions; the user will run it and
  paste results. If no free image-generation model works, the image stream stays mock-validated and you say so
  plainly in the report; never silently switch to a paid model.

### 4.2 Image stream (`cadence/tasks/image/`)
- Data: `data/tasks/image_prompts.jsonl`, 50 records: `id` (`img-001`...), `category` (>= 5 distinct: object, scene,
  portrait, abstract, diagram, ...), `concept` (6-40 words, a visual concept), `must_include` (2-4 short phrases that
  each appear in the concept, case-insensitive). Hand-written, varied, no real people or copyrighted characters.
- `ImageTask(Task)`: `name="image"`, `stream="image"`. Template = a prompt-prefix instruction; the image-generation
  prompt is `template + "\n\n" + concept` (single-stage, one image call per example). Splits via `select`.
  `format_input` returns the concept. `generate` calls `client.image`, writes the PNG under
  `media_dir/<candidate-hash>/<example-id>.png` and returns `Output(kind="image", path=..., usage=...)`.
  Constructor accepts `media_dir: Path | None` (default `ROOT/"runs"/"_media"`) and `records`/`path` like `TextTask`.
- Quality: `automatic_score` returns `None` (needs the vision judge). `rubric(ex)` returns a rubric on **prompt
  alignment** (are the `must_include` concepts visible), **aesthetic quality**, and **absence of artifacts**, 0-10.
  Reference-free by construction (there is no reference answer).
- Vision judging: extend `Judge.score` **additively** with an optional `media` argument carrying the image (as a
  base64 `image_url` content part in the user message), and make `Cascade.score` pass `out` for non-text outputs.
  Existing tests and the text path must not change. The cascade for media tasks is built with
  `Cascade.from_config(client, cfg, cheap_role="vision_judge", strong_role="vision_judge_strong")`
  (Groq text judges cannot see images; vision judges are Gemini vision-capable models, probed by the user).
  `calibration_rubric`, `degraded_prompt` and `corrupt_output` are optional for image; if you implement them, a
  degraded prompt (for example "reply with a blank white image") gives real low-quality samples.
- Known limitation to state in the report: with single-stage generation the per-image cost is constant across
  candidates, so the front for this stream is mostly quality versus prompt length. Do not hide this.

### 4.3 Audio stream (`cadence/tasks/audio/`)
- Data: `data/tasks/audio_prompts.jsonl`, 50 records: `id` (`aud-001`...), `text` (8-40 words, plain ASCII sentences
  ending in punctuation, spoken instructions such as "Turn left at the next junction and stop near the blue gate.").
- `AudioTask(Task)`: `name="audio"`, `stream="audio"`. Template = a delivery directive (e.g. "Speak calmly at a
  steady pace"). TTS input is `template + ": " + text`; the output WAV goes under `media_dir/...`. If the directive
  gets read aloud, WER rises, which is a legitimate signal.
- Quality is **automatic** (never needs the judge in this phase): transcribe the WAV with the `asr` role, compute
  `wer = jiwer.wer(normalised_text, normalised_transcript)` (jiwer is installed; lower-case, strip punctuation),
  `accuracy = max(0, 1 - wer)`, and `cps = len(text) / audio_seconds`. Decision made by the coordinator, adopt it
  unless you find a flaw and say why: `quality = 0.8 * accuracy + 0.2 * rate_score`, where `rate_score = 1.0` for
  `10 <= cps <= 18`, decaying linearly to 0 at `cps <= 4` or `cps >= 30`. Put `wer`, `cps` and `rate_score` in
  `Score.components` (the cascade only keeps `auto`; extend additively if needed so these reach `trace.jsonl`).
  Cost objective = TTS (`audio_seconds` units) plus ASR is NOT deployment cost: count only the TTS stage.
- Tests: WER maths on known pairs (identical = 0, one substitution, empty reference), `cps` and `rate_score`
  boundaries, cost with units, mock TTS->ASR round trip, determinism.

### 4.4 Interfaces Continue provides (do not implement)
`cadence/benchmark/plots.py`: `load_members(archive_json) -> list[Evaluated]`,
`plot_pareto_3d(members, out_path) -> Path`, `plot_pareto_2d(members, out_prefix) -> list[Path]`. Import lazily
inside `report.py` and degrade gracefully (note "plots unavailable") if it is missing, so you are never blocked.

### 4.5 CLI integration (append only)
`run` must, for the image and audio streams, create `runs/<run_id>/media/`, pass it to the task, select the vision
roles for the cascade (image), and otherwise behave as today. Registry gets `"image": ImageTask`, `"audio": AudioTask`
(append lines; Continue appends `"code_web"`; resolve trivial merge conflicts by keeping all lines).

## 5. Phase 8 specification: benchmark, dual judge, report

### 5.1 Data contract (decide this first, everything reads it)
`benchmark --run <id>` writes into `runs/<id>/`:
- `heldout.jsonl`: one line per (member, example): `candidate_id`, `example_id`, `output_text` (or `output_path`),
  `auto_score` (float or null), `primary_score` (0-1, what the cascade returned), `judged`, `escalated`,
  `rubric` (the exact rubric string used, needed to re-score), `input` (task input as given to the model),
  `gen_usd`.
- `heldout_summary.json`: per member: heldout quality, 95% bootstrap CI (seeded), cost per 1k, length, search-time
  quality (from `archive_search.json`), n examples, escalation ratio, plus the task name and seed.
- `dual_judge.json` (5.3), `calibration.json` (6.1), then `report.md`, plots, `pareto_3d.png`.

### 5.2 `cadence/benchmark/heldout.py`
- `evaluate_heldout(task, client, cascade, members, n, seed, workers, log)`: the **same** held-out examples for every
  member (`task.load("heldout", n, seed)`), evaluated through the existing `Evaluator` plumbing (reuse, do not
  duplicate), but keeping per-example outputs for the file above.
- **Exact-match anchoring**: where `automatic_score` is not None, the anchor is that score and no judge is called.
  For judged tasks the primary score comes from the cascade and is labelled as judged. Report these separately; never
  blend an anchored task's judge score into its number.
- Budget: `BudgetExceeded` mid-run writes what finished, marks the summary `partial: true`, exits cleanly.
- Report search-time vs held-out quality gap per member; a large positive gap is the overfitting/artifact signal the
  project's title promises to resist. Rank correlation (Spearman) between search-time and held-out quality too.

### 5.3 `cadence/benchmark/dual_judge.py` (inter-judge agreement and self-preference)
- Input: judged rows of `heldout.jsonl` only (anchored rows have ground truth, no judge needed).
- Re-score each judged output with `second_family_judge` (reference-free, same rubric, same trials as the primary).
  Compute: Cohen's kappa on `binarize(scores, 5.0)`, quadratic-weighted kappa on 0-10 scores, ICC(2,1) across the two
  judges, per-member mean score from each judge and **Spearman/Kendall rank agreement of members** between judges.
- **Same-family self-preference gap**: add role `same_family_judge` (a Gemma model on the Gemini API, free tier;
  the user verifies the name with the probe script; the generator is Gemini so Gemma is the same vendor family).
  Gap = mean(same-family score) minus mean(other-family judges' score) on identical outputs, with a paired seeded
  bootstrap 95% CI. State the caveat that Gemma and Gemini are the same vendor, not identical models.
- Flag: if inter-judge kappa < `kappa_min` (0.60), say so prominently; the judge ranking is not trustworthy.
- If judge roles are missing or the run has no judged rows, write `dual_judge.json` with `skipped: true` and the
  reason; never crash. Tests with scripted judges (agreeing, disagreeing, biased) pin the maths.

### 5.4 `cadence/benchmark/report.py` -> `runs/<id>/report.md`
Required sections (SPEC definition of done), each present even if empty ("not available: reason"):
1. Run header: task, stream, seed, models per role, date, settings, git commit if available.
2. Pareto menu (from `archive.json`), with held-out quality and CI next to search-time quality.
3. Judge reliability: kappa vs ground truth, ICC, flagged status, position and verbosity bias (from `calibration.json`).
4. Cascade: escalation ratio over generations (from `trace.jsonl`) and what it saved versus always-strong.
5. Spend versus budget (calls, tokens, USD from the `generation` events and `budget.yaml`).
6. Held-out benchmark and exact-match anchor results; the search-vs-held-out gap.
7. Dual-judge results and self-preference gap.
8. Streams demonstrated (one line per stream with run ids if multiple runs are given) and honest limitations
   (unverified prices, constant image cost, small n, free-tier models).
- Pure function of the run directory contents; deterministic; no network; tolerant of missing files. Also
  `report --run <id>` regenerates it. Tests: all headings present on a mock ToyTask run, missing-file tolerance, numbers
  match fixtures.

### 5.5 Plots
After the report data is ready, call Continue's `plot_pareto_3d(members, runs/<id>/pareto_3d.png)` and
`plot_pareto_2d(members, runs/<id>/pareto)`; embed the paths in `report.md`.

### 5.6 CLI (append)
`benchmark --run <id> [--n N] [--mock]`, `report --run <id>`, and `calibrate --task <name> [--n N] [--mock]`.
After `run` finishes, it should print the report path; `run --report` (default on) calls report generation without
the benchmark, so the DoD ("run writes archive, trace, report, 3D plot") holds. Exit codes: 0 ok, 1 handled failure.

## 6. Calibration and loop wiring (do early in Phase 8)

### 6.1 `calibrate` and `kappa` in the loop
- `python -m cadence.cli calibrate --task <name>`: build the calibration set from the task's `calib` split
  (genuine, degraded-prompt and corrupted outputs, as `cadence/smoke.py` does), run `calibrate_with_recheck` for the
  cheap and strong judges (and second-family when present), write `runs/<id or calibration>/calibration.json` with
  kappa, ICC, parse failures, per-source accuracy, bias probes and flagged status.
- `run` should run calibration first for **judged** tasks (skip when the task is fully auto-scored), pass the minimum
  kappa into `optimize(kappa=...)` so it is logged each generation, and print a loud warning when flagged. A flagged
  judge is a warning plus the doubled-trials recheck already implemented, not an abort.
- Honest caveat to keep in the report: the previous calibration had n=12 and mostly easy corrupted items; the
  confidence-based cascade cannot detect a consistently wrong judge; say so unless you strengthen the set.

## 7. Quality bar and acceptance

- `python -m pytest` green offline in under 60 s, including `-m e2e`. Old tests untouched and passing. New public
  functions each get a normal, an edge and a failure test.
- No network or keys in tests; deterministic by seed; thread-safe where the evaluator uses threads.
- Failure behaviour: provider hiccup costs one example, not the run (the evaluator tolerates up to 25% failures).
- Before delivering each phase: review your own diff for budget-guard bypasses, cache key correctness (media bytes!),
  thread safety, rubric leakage of references, and paths that break on Windows (`pathlib`, no hard-coded slashes).
- Deliver per phase: bundle, apply commands, test summary line, the list of things the user must verify live
  (model names, prices, `probe_media.py` output), and any deviation from this document with the reason.
- Final summary in plain language: what works, what is mock-only, what is unverified. No overclaiming.

## 8. First message you should send back

Confirm the base branch, say `pytest` result before changes, list which of this document's decisions you disagree
with (if any, with reasons), then start step 2 of section 3.
