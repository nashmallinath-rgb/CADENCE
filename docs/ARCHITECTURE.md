# CADENCE: Architecture

Extend the existing scaffold (`cadence/llm/client.py`, `cadence/config.py`, `cadence/tasks/base.py`, `cadence/check_setup.py`, `cadence/cli.py`). Do not rewrite them wholesale; reconcile with the contracts below.

## 1. Layout
```
cadence/
  llm/        client.py cache.py budget.py mock.py     # all model calls go here
  cost/       pricing.py tracker.py                    # configs/pricing.yaml
  pareto/     dominance.py archive.py
  judge/      judge.py prompts.py
  cascade/    cascade.py
  validator/  agreement.py bias.py calibrate.py
  orchestrator/ generator.py evaluate.py loop.py
  tasks/      base.py toy.py text/ code_web/ image/ audio/
  tracking/   mlflow_logger.py
  benchmark/  heldout.py dual_judge.py report.py plots.py
  cli.py
configs/  models.yaml budget.yaml pricing.yaml cascade.yaml run.yaml
runs/<run_id>/  config.json archive.json trace.jsonl report.md pareto.png   (gitignored)
```

## 2. Data contracts (pydantic v2; keep these names)
```python
class Usage(BaseModel): input_tokens:int=0; output_tokens:int=0; usd:float=0.0; units:dict[str,float]={}  # e.g. {"images":1,"audio_seconds":4.2}
class Example(BaseModel): id:str; input:dict[str,Any]; reference:Any|None=None; meta:dict[str,Any]={}
class GenRequest(BaseModel): role:str; messages:list[dict]; params:dict={}      # role keys map to configs/models.yaml
class Output(BaseModel): kind:Literal["text","code","image","audio"]; text:str|None=None; path:str|None=None; usage:Usage
class Score(BaseModel): quality:float; components:dict[str,float]={}; judged:bool=False; escalated:bool=False; usage:Usage=Usage()
class Candidate(BaseModel): id:str; template:str; parent_ids:list[str]=[]; generation:int=0
class Objectives(BaseModel): quality:float; cost_usd_per_1k:float; length_tokens:int
class Evaluated(BaseModel): candidate:Candidate; objectives:Objectives; n_examples:int; escalation_ratio:float
```
```python
class Task(ABC):
    name: str; stream: Literal["text","code_web","image","audio"]
    def load(self, split: Literal["opt","heldout","calib"], n: int|None=None, seed:int=0) -> list[Example]: ...
    def build_request(self, template: str, ex: Example) -> GenRequest: ...
    def generate(self, client: "LLMClient", req: GenRequest) -> Output: ...      # default = text chat; image/audio override
    def automatic_score(self, ex: Example, out: Output) -> float | None: ...     # exact match etc.; None means "needs judge"
    def rubric(self, ex: Example) -> str | None: ...                              # for the judge
    def seed_prompt(self) -> str: ...
```

## 3. Modules
**llm.LLMClient**: `complete(req) -> (text, Usage)`, `image(req)`, `speech(req)`, `transcribe(req)`. Wraps LiteLLM. Disk cache (diskcache, key = hash of model+messages+params+seed), tenacity retry with backoff, per-provider RPM throttle, `BudgetGuard` that raises `BudgetExceeded` before a call would exceed caps in `budget.yaml`. `MockClient` has the same interface: deterministic, role-aware, scriptable (`MockClient(script=...)`), zero cost. Select with `CADENCE_MOCK=1` or `--mock`.

**cost**: `price(usage, model)` from `configs/pricing.yaml` (placeholders marked `VERIFY`), `cost_per_1k(usages)`, `template_length(template)` via tiktoken. Pricing values must be checked by the user against current provider pages.

**pareto**: `dominates(a, b)`, `non_dominated(list)`, `ParetoArchive(add(evaluated) -> bool, members, to_json/from_json, hypervolume(ref))`. Quality is maximized, cost and length minimized. Use `pymoo.NonDominatedSorting` or a plain O(n^2) loop; hypervolume via `moocore`. Archive never exceeds `max_archive` (crowding-distance prune).

**judge**: `Judge(role, trials)`. `score(ex, out, rubric) -> JudgeResult(score_0_10, confidence_0_1, trials:list, usage)`. Structured JSON output, parsed with `json_repair`. Reference-free by default (judge must not see the reference answer).

**cascade**: `Cascade(cheap, strong, tau, delta)`. Per (candidate, example): (1) `task.automatic_score` if not None, no judge needed; (2) else cheap judge with k trials; escalate to strong if `mean_confidence < tau` or `score_spread > delta`; return `Score(escalated=...)`. Track escalation ratio and spend.

**validator**: `agreement.py`: `cohen_kappa(a, b, weights)` (sklearn), `icc_2_1(matrix)` (numpy). `bias.py`: `position_bias(judge, pairs)` (swap-order flip rate), `verbosity_bias(scores, lengths)` (Spearman). `calibrate.py`: builds a calibration set from outputs whose correctness is known via exact match (judge rates reference-free, kappa vs. ground truth), plus optional hand-labelled items. If kappa < 0.60: return `JudgeHealth(flagged=True)`, orchestrator raises `trials` and re-checks once.

**orchestrator**:
```
archive = ParetoArchive(); evaluate(seed); archive.add(...)
while budget.ok() and gen < max_gen:
    parents  = select(archive.members, k)                       # include seed if archive empty
    traces   = worst_examples(parents)                          # low-scoring (input, output, score)
    children = generator.propose(parents, traces, n)            # reflective rewrite + mutation + crossover (LLM role "reflector")
    for c in children: archive.add(evaluate(c, minibatch))      # cascade on a minibatch
    tracker.log(gen, archive, spend, escalation_ratio, kappa)
final: re-evaluate archive members on a larger opt sample; write run dir
```
DSPy: keep behind `generator.py`. Use it where it helps (signatures for the reflection step, GEPA-style feedback). If it fights `cadence.llm` (budget/cache), implement reflection with plain `client.complete` calls instead. Don't let DSPy bypass the budget guard.

**tasks**: `ToyTask` (offline): quality rises with instruction keywords present in the template, cost and length rise with verbosity, so a trade-off front exists. Used by the e2e test and by `--mock` demos. Real tasks: text (gsm8k, hotpotqa, ifbench, sst2), code_web (jsonl specs + validators), image (gen + vision judge), audio (TTS + ASR round trip).

**benchmark**: `heldout.py` evaluates each archive member on the `heldout` split with `automatic_score` as the anchor where available; `dual_judge.py` re-scores judged outputs with the second-family model, reports inter-judge kappa, rank agreement and a same-family self-preference gap; `report.py` writes `report.md`; `plots.py` writes 3D Pareto (matplotlib) and 2D projections.

**tracking**: MLflow with `sqlite:///mlflow.db` (gitignored). Log params, per-generation metrics (front size, hypervolume, spend, escalation ratio, kappa), artifacts (archive.json, report, plots).

## 4. Config keys (names fixed)
`models.yaml` roles: generator, reflector, cheap_judge, strong_judge, second_family_judge, vision_judge, image_gen, tts, asr. `budget.yaml`: max_usd_per_run, max_calls_per_run, max_tokens_per_run. `cascade.yaml`: tau, delta, cheap_trials, strong_trials. `run.yaml`: seed, max_gen, children_per_gen, minibatch, max_archive, final_sample.

## 5. CLI
`check-setup`, `run --stream --task --budget-usd [--mock]`, `calibrate --task`, `benchmark --run`, `report --run`.
