"""Evaluate one candidate prompt on a minibatch: generate, score through the cascade, aggregate to Objectives.

Objectives:
- quality: mean cascade score over the examples that evaluated successfully.
- cost_usd_per_1k: DEPLOYMENT cost, i.e. the generator's usage only (judge spend is optimisation overhead, tracked
  separately in `EvalResult.judge_usd`).
- length_tokens: token count of the template.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from cadence.cascade import Cascade
from cadence.cost import cost_per_1k, template_length
from cadence.llm.budget import BudgetExceeded
from cadence.schemas import Candidate, Evaluated, Example, Objectives, Output, Score
from cadence.tasks.base import Task

MAX_ERROR_RATE = 0.25  # more failed examples than this and the evaluation is not trusted


class EvaluationFailed(RuntimeError):
    """Too many examples failed (provider outage, parse errors). The loop skips the candidate."""


class Trace(BaseModel):
    """One (input, output, score) record. The reflector learns from the worst ones."""

    example_id: str
    input: str
    output: str
    quality: float
    judged: bool = False
    rationale: str = ""
    components: dict[str, float] = Field(default_factory=dict)


class EvalResult(BaseModel):
    evaluated: Evaluated
    traces: list[Trace] = Field(default_factory=list)
    n_errors: int = 0
    n_judged: int = 0
    n_escalated: int = 0
    judge_usd: float = 0.0
    generation_usd: float = 0.0


def _clip(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def _one(task: Task, client: Any, cascade: Cascade, template: str, ex: Example) -> tuple[Output, Score]:
    out = task.generate(client, task.build_request(template, ex))
    return out, cascade.score(task, ex, out)


class Evaluator:
    def __init__(self, task: Task, client: Any, cascade: Cascade, workers: int = 4) -> None:
        self.task, self.client, self.cascade, self.workers = task, client, cascade, max(1, workers)

    def evaluate(self, cand: Candidate, examples: Sequence[Example]) -> EvalResult:
        if not examples:
            raise ValueError("evaluate needs at least one example")
        done: list[tuple[Example, Output, Score]] = []
        errors = 0
        budget_hit: BudgetExceeded | None = None

        def work(ex: Example) -> tuple[Example, Output, Score]:
            out, sc = _one(self.task, self.client, self.cascade, cand.template, ex)
            return ex, out, sc

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(work, ex) for ex in examples]
            for fut in futures:  # collect in submission order so results are deterministic
                try:
                    done.append(fut.result())
                except BudgetExceeded as exc:
                    budget_hit = budget_hit or exc
                except Exception:  # noqa: BLE001 - provider hiccups cost the example, not the run
                    errors += 1
        if budget_hit is not None:
            raise budget_hit
        if not done or errors / len(examples) > MAX_ERROR_RATE:
            raise EvaluationFailed(f"{errors}/{len(examples)} examples failed for candidate {cand.id}")

        quality = sum(sc.quality for _, _, sc in done) / len(done)
        cost = cost_per_1k([o.usage for _, o, _ in done])
        judged = [sc for _, _, sc in done if sc.judged]
        escalated = sum(1 for sc in judged if sc.escalated)
        ev = Evaluated(
            candidate=cand,
            objectives=Objectives(quality=quality, cost_usd_per_1k=cost, length_tokens=template_length(cand.template)),
            n_examples=len(done),
            escalation_ratio=escalated / len(judged) if judged else 0.0,
        )
        traces = [
            Trace(example_id=ex.id, input=_clip(self.task.format_input(ex)), output=_clip(out.text or ""),
                  quality=sc.quality, judged=sc.judged, components=dict(sc.components))
            for ex, out, sc in done
        ]
        judge_usd = sum((sc.usage.usd for _, _, sc in done), 0.0)
        gen_usd = sum((o.usage.usd for _, o, _ in done), 0.0)
        return EvalResult(evaluated=ev, traces=traces, n_errors=errors, n_judged=len(judged),
                          n_escalated=escalated, judge_usd=judge_usd, generation_usd=gen_usd)


def worst_traces(traces: Sequence[Trace], k: int = 4) -> list[Trace]:
    """Lowest-scoring traces first; ties keep input order, so the choice is deterministic."""
    return sorted(traces, key=lambda t: t.quality)[:k]


__all__ = ["EvalResult", "EvaluationFailed", "Evaluator", "Trace", "worst_traces"]
