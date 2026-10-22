"""The closed optimisation loop (docs/ARCHITECTURE.md, orchestrator).

    evaluate(seed) -> archive
    each generation: pick parents from the archive -> reflector proposes children -> evaluate children on a minibatch
                     -> archive.add -> log
    stop on max_gen or budget; then re-evaluate the archive on a larger sample and write the run directory.

Run directory (resume-safe: files are replaced atomically, state.json is written last each generation):
    config.json  state.json  archive.json  archive_search.json  trace.jsonl  menu.md
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cadence.cascade import Cascade
from cadence.llm.budget import BudgetExceeded
from cadence.orchestrator.evaluate import EvalResult, EvaluationFailed, Evaluator, Trace, worst_traces
from cadence.orchestrator.generator import Generator, template_hash
from cadence.pareto import ParetoArchive
from cadence.pareto.dominance import crowding_distance, to_min_vector
from cadence.schemas import Candidate, Evaluated, Example, Objectives
from cadence.tasks.base import Task
from cadence.tracking import NullTracker, Tracker


class LoopSettings(BaseModel):
    seed: int = 0
    max_gen: int = Field(default=8, ge=0)
    children_per_gen: int = Field(default=4, ge=1)
    minibatch: int = Field(default=16, ge=1)
    max_archive: int = Field(default=20, ge=1)
    final_sample: int = Field(default=64, ge=1)
    workers: int = Field(default=4, ge=1)
    parents_per_gen: int = Field(default=2, ge=1)


class OptimizationResult(BaseModel):
    run_dir: str
    stopped_reason: str  # "max_gen" | "budget:<kind>"
    generations_done: int
    archive: list[Evaluated]
    hv_history: list[float]
    final_reevaluated: bool
    resumed: bool = False


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def reference_point(seed_eval: Evaluated) -> Objectives:
    """Fixed hypervolume reference derived from the seed, so hypervolume is comparable across generations."""
    o = seed_eval.objectives
    return Objectives(quality=0.0, cost_usd_per_1k=max(10 * o.cost_usd_per_1k, 1e-6),
                      length_tokens=int(8 * o.length_tokens + 64))


def minibatch_for(pool: list[Example], gen: int, size: int) -> list[Example]:
    """Rotating window over the opt split: a new minibatch each generation, deterministic."""
    if not pool:
        raise ValueError("task returned no optimisation examples")
    start = (gen * size) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(min(size, len(pool)))]


def select_parents(members: list[Evaluated], k: int, rng: random.Random) -> list[Evaluated]:
    """Sample k parents without replacement, favouring isolated (high crowding distance) front members."""
    if len(members) <= k:
        return list(members)
    d = crowding_distance([to_min_vector(m.objectives) for m in members])
    finite = [x for x in d if math.isfinite(x)]
    cap = 2 * max(finite) if finite and max(finite) > 0 else 1.0
    weights = [(cap if not math.isfinite(x) else x) + 1e-6 for x in d]
    pool, w, chosen = list(range(len(members))), weights, []
    for _ in range(k):
        i = rng.choices(range(len(pool)), weights=w)[0]
        chosen.append(members[pool.pop(i)])
        w = [w[j] for j in range(len(w)) if j != i]
    return chosen


def pareto_menu(members: list[Evaluated], max_template_chars: int = 90) -> str:
    """Human-readable menu: best quality first, each with its price and size."""
    rows = sorted(members, key=lambda m: (-m.objectives.quality, m.objectives.cost_usd_per_1k))
    lines = ["| # | quality | $ / 1k runs | tokens | gen | template |", "|---|---|---|---|---|---|"]
    for i, m in enumerate(rows, 1):
        o, t = m.objectives, " ".join(m.candidate.template.split())
        t = t if len(t) <= max_template_chars else t[: max_template_chars - 3] + "..."
        lines.append(f"| {i} | {o.quality:.3f} | {o.cost_usd_per_1k:.4f} | {o.length_tokens} | "
                     f"{m.candidate.generation} | {t} |")
    return "\n".join(lines)


def optimize(task: Task, client: Any, cascade: Cascade, settings: LoopSettings, run_dir: Path, *,
             tracker: Tracker | None = None, kappa: float | None = None, resume: bool = False,
             log: Callable[[str], None] = print) -> OptimizationResult:
    tracker = tracker or NullTracker()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = getattr(client, "budget", None)
    evaluator = Evaluator(task, client, cascade, workers=settings.workers)
    generator = Generator(client, seed=settings.seed,
                          task_hint=f"{task.name}. Seed prompt: {task.seed_prompt()}")
    cfg = {"task": task.name, "settings": settings.model_dump()}
    state_p, arch_p, cfg_p, trace_p = (run_dir / n for n in ("state.json", "archive.json", "config.json", "trace.jsonl"))

    archive = ParetoArchive(settings.max_archive)
    traces: dict[str, list[Trace]] = {}
    seen: set[str] = set()
    hv_history: list[float] = []
    ref: Objectives | None = None
    gen_done = -1
    resumed = False

    if resume and state_p.exists():
        old = json.loads(cfg_p.read_text(encoding="utf-8"))
        strip = lambda c: {"task": c["task"], "settings": {k: v for k, v in c["settings"].items() if k != "max_gen"}}  # noqa: E731
        if strip(old) != strip(cfg):  # extending max_gen is the point of resuming; anything else changes the search
            raise ValueError("cannot resume: run directory was created with a different task/settings")
        _atomic_write(cfg_p, json.dumps(cfg, indent=2, sort_keys=True))
        st = json.loads(state_p.read_text(encoding="utf-8"))
        if not arch_p.exists():
            raise ValueError(f"cannot resume: {arch_p.name} is missing from {run_dir} (state.json exists, so the run is incomplete)")
        archive = ParetoArchive.from_json(arch_p.read_text(encoding="utf-8"), settings.max_archive)
        seen, hv_history, gen_done = set(st["seen"]), list(st["hv_history"]), int(st["gen_done"])
        ref = Objectives.model_validate(st["ref"])
        resumed = True
        log(f"resuming after generation {gen_done} ({len(archive)} archive members)")
    elif state_p.exists() and not resume:
        raise FileExistsError(f"{run_dir} already holds a run; pass resume=True or choose a new run id")
    else:
        _atomic_write(cfg_p, json.dumps(cfg, indent=2, sort_keys=True))

    def emit(event: dict[str, Any]) -> None:
        with trace_p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def persist(gen: int) -> None:
        _atomic_write(arch_p, archive.to_json())
        _atomic_write(state_p, json.dumps({"gen_done": gen, "seen": sorted(seen), "hv_history": hv_history,
                                          "ref": ref.model_dump() if ref else None}))

    def record(res: EvalResult, gen: int) -> bool:
        added = archive.add(res.evaluated)
        traces[res.evaluated.candidate.id] = res.traces
        emit({"event": "evaluated", "gen": gen, "id": res.evaluated.candidate.id,
              "parents": res.evaluated.candidate.parent_ids, "template": res.evaluated.candidate.template,
              "objectives": res.evaluated.objectives.model_dump(), "n_errors": res.n_errors, "in_archive": added,
              "n_examples": res.evaluated.n_examples, "n_judged": res.n_judged, "n_escalated": res.n_escalated,
              "judge_usd": res.judge_usd, "generation_usd": res.generation_usd})
        return added

    tracker.start(run_dir.name, {**cfg["settings"], "task": task.name, "resumed": resumed})
    opt_pool = task.load("opt", seed=settings.seed)
    stopped = "max_gen"
    status = "FINISHED"
    try:
        # generation 0: the seed prompt
        if gen_done < 0:
            seed_c = Candidate(id="seed", template=task.seed_prompt(), generation=0)
            seen.add(template_hash(seed_c.template))
            try:
                res = evaluator.evaluate(seed_c, minibatch_for(opt_pool, 0, settings.minibatch))
            except BudgetExceeded as exc:
                raise RuntimeError(f"budget exhausted before the seed prompt could be evaluated ({exc})") from exc
            record(res, 0)
            ref = reference_point(res.evaluated)
            hv_history.append(archive.hypervolume(ref))
            gen_done = 0
            persist(0)
            _log_generation(log, tracker, emit, 0, archive, ref, hv_history, budget, [res], kappa, 1, 1)

        assert ref is not None
        for gen in range(gen_done + 1, settings.max_gen + 1):
            if budget is not None and not budget.ok():
                stopped = "budget:exhausted"
                break
            rng = random.Random(settings.seed * 1_000_003 + gen)
            parents = select_parents(archive.members, settings.parents_per_gen, rng)
            try:
                children = generator.propose(parents, traces_for(parents, traces), settings.children_per_gen, gen, seen)
            except BudgetExceeded as exc:
                stopped = f"budget:{exc.kind}"
                break
            batch = minibatch_for(opt_pool, gen, settings.minibatch)
            results: list[EvalResult] = []
            n_added = 0
            for child in children:
                try:
                    res = evaluator.evaluate(child, batch)
                except BudgetExceeded as exc:
                    stopped = f"budget:{exc.kind}"
                    break
                except EvaluationFailed as exc:
                    emit({"event": "skipped", "gen": gen, "id": child.id, "reason": str(exc)})
                    continue
                results.append(res)
                n_added += record(res, gen)
            hv_history.append(archive.hypervolume(ref))
            if stopped != "max_gen":  # budget hit mid-generation: keep what finished, then stop
                persist(gen)
                _log_generation(log, tracker, emit, gen, archive, ref, hv_history, budget, results, kappa,
                                len(children), n_added)
                gen_done = gen
                break
            persist(gen)
            _log_generation(log, tracker, emit, gen, archive, ref, hv_history, budget, results, kappa,
                            len(children), n_added)
            gen_done = gen

        # final: re-evaluate the archive on a larger sample for honest objectives
        _atomic_write(run_dir / "archive_search.json", archive.to_json())
        reevaluated = False
        if archive.members:
            final_batch = opt_pool[: settings.final_sample]
            try:
                final = ParetoArchive(settings.max_archive)
                for m in archive.members:
                    r = evaluator.evaluate(m.candidate, final_batch)
                    final.add(r.evaluated)
                archive, reevaluated = final, True
            except (BudgetExceeded, EvaluationFailed) as exc:
                emit({"event": "final_reeval_skipped", "reason": str(exc)})
                log(f"final re-evaluation skipped: {exc}")
        persist(gen_done)
        _atomic_write(run_dir / "menu.md", pareto_menu(archive.members) + "\n")
        emit({"event": "stop", "reason": stopped, "generations": gen_done, "final_reevaluated": reevaluated,
              "time": time.time()})
        for name in ("archive.json", "trace.jsonl", "menu.md", "config.json"):
            tracker.log_artifact(run_dir / name)
    except BaseException:
        status = "FAILED"
        raise
    finally:
        tracker.end(status)
    return OptimizationResult(run_dir=str(run_dir), stopped_reason=stopped, generations_done=gen_done,
                              archive=archive.members, hv_history=hv_history, final_reevaluated=reevaluated,
                              resumed=resumed)


def traces_for(parents: list[Evaluated], traces: dict[str, list[Trace]]) -> dict[str, list[Trace]]:
    return {p.candidate.id: worst_traces(traces.get(p.candidate.id, []), 4) for p in parents}


def _log_generation(log: Callable[[str], None], tracker: Tracker, emit: Callable[[dict[str, Any]], None], gen: int,
                    archive: ParetoArchive, ref: Objectives, hv_history: list[float], budget: Any,
                    results: list[EvalResult], kappa: float | None, n_children: int, n_added: int) -> None:
    judged = sum(r.n_judged for r in results)
    esc = sum(r.n_escalated for r in results)
    best = max((m.objectives.quality for m in archive.members), default=0.0)
    metrics = {
        "front_size": float(len(archive)), "hypervolume": hv_history[-1], "best_quality": best,
        "escalation_ratio": esc / judged if judged else 0.0,
        "spend_usd": budget.spent_usd if budget is not None else None,
        "calls": float(budget.calls) if budget is not None else None, "kappa": kappa,
        "children": float(n_children), "children_added": float(n_added),
    }
    tracker.log_generation(gen, metrics)
    emit({"event": "generation", "gen": gen, **metrics})
    log(f"gen {gen}: front={len(archive)} hv={hv_history[-1]:.4g} best_q={best:.3f} "
        f"added={n_added}/{n_children}" + (f" spend=${budget.spent_usd:.4f}" if budget is not None else ""))
