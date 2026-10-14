# CADENCE: Specification

Automated closed-loop prompt optimization that returns a **Pareto-optimal archive of prompts** over three objectives, with a calibrated LLM judge and a cost-aware evaluation cascade.

## 1. Objectives and the gaps they close
| Gap | Objective |
|---|---|
| G1 Single-objective bias | **O1** Jointly optimize quality, token cost, prompt length; output a Pareto archive |
| G2 Unvalidated judges | **O2** Calibrate judges with chance-corrected Cohen's kappa and ICC; check position/verbosity bias |
| G3 Naive evaluation budgets | **O3** Two-tier cascade: cheap judge first, strong judge only under uncertainty |
| G4 Benchmark artifacts | **O4** Held-out evaluation with exact-match anchoring and dual-judge (two model families) cross-validation |

## 2. Definitions
- **Candidate** = a prompt template string with `{placeholders}` filled from an `Example`.
- **Objective vector** per candidate: `[quality, cost, length]`.
  - `quality` in [0,1], stream-specific (exact match, structural validity, judge score, WER-based, ...). Maximize.
  - `cost` = USD per 1,000 executions: `1000 * (avg_input_tokens * P_in + avg_output_tokens * P_out)`. For image/audio, add per-unit price (per image, per audio second). Use provider-reported usage, never estimates, when available. Minimize.
  - `length` = token count of the template using tiktoken `cl100k_base` (documented proxy, since Gemini/Grok tokenizers are not available). Minimize.
- **Dominance**: A dominates B if A is no worse on all three and strictly better on at least one (after flipping quality to a minimization axis).
- **Judge reliability**: kappa below 0.60 flags the judge; the response is more ensemble trials, then re-check.

## 3. Modality streams (one orchestrator, one Pareto schema)
| Stream | Tasks | Quality metric | Cost/length focus |
|---|---|---|---|
| 1 Text & reasoning | GSM8K, HotpotQA, IFBench, SST-2 (cascade/judge validation) | exact match; rubric judge for open-ended | short tokens |
| 2 Code & web UI | ~30 hand-written full-stack HTML/JS task specs | structural validity (HTML parse, required elements, JS syntax check) plus optional judge | heavy output tokens |
| 3 Image | ~50 visual concept prompts | vision-judge rubric (prompt alignment, aesthetics); CLIP is deferred | prompt length vs per-image cost |
| 4 Audio/TTS | ~50 spoken-instruction prompts | ASR round trip (WER via jiwer), judge for clarity/adherence | chars-per-second ratio, per-second cost |

Dataset ids (verify on HuggingFace before relying on them): `openai/gsm8k` (main), `hotpotqa/hotpot_qa` (distractor), `stanfordnlp/sst2`, IFBench from AllenAI. Tests never touch the network; they use tiny fixtures in `tests/fixtures/`.

## 4. Constraints
- Windows + PowerShell, Python 3.12 only, LF line endings.
- Providers: **Gemini and Grok (xAI) only**, always through LiteLLM and `cadence.llm`. No direct SDK calls except where LiteLLM lacks image/audio support (then add one adapter in `cadence/llm/`).
- No secrets in code, logs or git. `.env` only.
- **Mock-first**: every module is fully testable offline via `MockClient`. Real-API runs are manual, gated by `budget.yaml` caps.
- Determinism: all randomness takes a seed from config.
- No torch/CLIP in v1.

## 5. Prototype definition of done
1. `pytest` passes offline, including an end-to-end test (`-m e2e`) that runs the whole loop on `ToyTask` in under 60 s and shows the Pareto front improving.
2. `python -m cadence.cli run --stream <text|code_web|image|audio> --task <name> --budget-usd N` runs with real keys and writes a run directory (archive, trace, report, 3D Pareto plot).
3. `python -m cadence.cli benchmark --run <id>` re-scores the archive on the held-out split with exact-match anchoring and a second-family judge, and reports inter-judge kappa.
4. Report lists: Pareto menu, judge kappa/ICC, escalation ratio, spend vs budget, all four streams demonstrated.

## 6. Non-goals
UI beyond the MLflow UI and a static HTML/Markdown report, fine-tuning, multi-user, torch-based metrics, deployment.
