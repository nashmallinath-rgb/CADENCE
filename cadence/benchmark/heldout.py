"""Held-out evaluation of the final archive, with exact-match anchoring where the task has an automatic metric.

Every archive member is run on the SAME held-out examples. Rows keep the output and the rubric so a second judge can
re-score them later (dual_judge.py). Anchored rows (automatic_score is not None) never touch a judge.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cadence.benchmark.stats import bootstrap_ci, mean, rank_agreement
from cadence.cascade import Cascade
from cadence.cost import cost_per_1k, template_length
from cadence.llm.budget import BudgetExceeded
from cadence.schemas import Candidate, Evaluated, Example, Output, Score
from cadence.tasks.base import Task


class HeldoutRow(BaseModel):
    candidate_id: str
    example_id: str
    input: dict[str, Any]
    output_text: str | None = None
    output_path: str | None = None
    output_kind: str = "text"
    auto_score: float | None = None
    primary_score: float | None = None  # 0-1 from the cascade; None when the example errored
    judged: bool = False
    escalated: bool = False
    rubric: str | None = None
    gen_usd: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class MemberSummary(BaseModel):
    candidate_id: str
    template: str
    n: int
    n_errors: int
    heldout_quality: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    opt_quality: float | None  # from the archive (final re-evaluation on the optimisation split); None for a baseline
    gap: float | None = None  # opt_quality - heldout_quality; large positive = overfitting to the opt split
    is_baseline: bool = False  # the run's starting (seed) prompt, evaluated on the same held-out examples
    cost_usd_per_1k: float | None = None
    length_tokens: int
    anchored_fraction: float = 0.0
    escalation_ratio: float = 0.0


class HeldoutResult(BaseModel):
    task: str
    seed: int
    n_requested: int
    partial: bool = False
    stop_reason: str | None = None
    rows: list[HeldoutRow] = Field(default_factory=list)
    members: list[MemberSummary] = Field(default_factory=list)
    opt_vs_heldout_rank: dict[str, float | None] = Field(default_factory=dict)


def _row(task: Task, cascade: Cascade, template: str, cid: str, ex: Example, client: Any) -> HeldoutRow:
    out: Output = task.generate(client, task.build_request(template, ex))
    sc: Score = cascade.score(task, ex, out)
    auto = sc.components.get("auto") if not sc.judged else None
    return HeldoutRow(candidate_id=cid, example_id=ex.id, input=ex.input, output_text=out.text, output_path=out.path,
                      output_kind=out.kind, auto_score=auto, primary_score=sc.quality, judged=sc.judged,
                      escalated=sc.escalated, rubric=task.rubric(ex) if sc.judged else None, gen_usd=out.usage.usd,
                      components={k: v for k, v in sc.components.items() if k != "auto"})


BASELINE_ID = "baseline-seed"


def summarize(rows: Sequence[HeldoutRow], members: Sequence[Evaluated], seed: int,
              baseline: Candidate | None = None) -> list[MemberSummary]:
    """One summary per archive member, plus one for the baseline seed prompt when it is not already a member."""
    items: list[tuple[Candidate, float | None, bool]] = [
        (m.candidate, m.objectives.quality, baseline is not None and m.candidate.template == baseline.template)
        for m in members]
    if baseline is not None and not any(is_b for _, _, is_b in items):
        items.append((baseline, None, True))
    out: list[MemberSummary] = []
    for cand, opt_q, is_b in items:
        mine = [r for r in rows if r.candidate_id == cand.id]
        ok = [r for r in mine if r.primary_score is not None]
        q = [float(r.primary_score) for r in ok]  # type: ignore[arg-type]
        judged = [r for r in ok if r.judged]
        hq = mean(q) if q else None
        lo, hi = bootstrap_ci(q, seed=seed) if q else (None, None)
        out.append(MemberSummary(
            candidate_id=cand.id, template=cand.template, n=len(ok), n_errors=len(mine) - len(ok),
            heldout_quality=hq, ci_low=lo, ci_high=hi, opt_quality=opt_q, is_baseline=is_b,
            gap=(opt_q - hq) if (hq is not None and opt_q is not None) else None,
            cost_usd_per_1k=1000 * mean([r.gen_usd for r in ok]) if ok else None,
            length_tokens=template_length(cand.template),
            anchored_fraction=(sum(r.auto_score is not None for r in ok) / len(ok)) if ok else 0.0,
            escalation_ratio=(sum(r.escalated for r in judged) / len(judged)) if judged else 0.0))
    return out


def evaluate_heldout(task: Task, client: Any, cascade: Cascade, members: Sequence[Evaluated], n: int, seed: int = 0,
                     workers: int = 4, log: Callable[[str], None] = print,
                     baseline_template: str | None = None) -> HeldoutResult:
    """Run every member (and, optionally, the run's starting prompt as a baseline) on the same held-out examples."""
    examples = task.load("heldout", n=n, seed=seed)
    if not examples:
        raise ValueError("task has no held-out examples")
    res = HeldoutResult(task=task.name, seed=seed, n_requested=len(examples))
    baseline = Candidate(id=BASELINE_ID, template=baseline_template) if baseline_template else None
    contenders = [m.candidate for m in members]
    if baseline is not None and not any(c.template == baseline.template for c in contenders):
        contenders.append(baseline)  # a seed that is already on the front is not run twice
    jobs = [(c, ex) for c in contenders for ex in examples]
    budget_hit: BudgetExceeded | None = None

    def work(job: tuple[Candidate, Example]) -> HeldoutRow:
        c, ex = job
        try:
            return _row(task, cascade, c.template, c.id, ex, client)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad example costs one row, not the benchmark
            return HeldoutRow(candidate_id=c.id, example_id=ex.id, input=ex.input,
                              error=f"{type(exc).__name__}: {str(exc)[:120]}")

    extra = " + the seed prompt as a baseline" if len(contenders) > len(members) else ""
    log(f"held-out: {len(members)} member(s){extra} x {len(examples)} example(s)")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in futures:  # submission order keeps the output deterministic
            try:
                res.rows.append(fut.result())
            except BudgetExceeded as exc:
                budget_hit = budget_hit or exc
    if budget_hit is not None:
        res.partial, res.stop_reason = True, f"budget:{budget_hit.kind}"
        log(f"held-out stopped early: {budget_hit}")
    res.members = summarize(res.rows, members, seed, baseline)
    pairs = [(s.opt_quality, s.heldout_quality) for s in res.members
             if s.heldout_quality is not None and s.opt_quality is not None]
    res.opt_vs_heldout_rank = rank_agreement([p[0] for p in pairs], [p[1] for p in pairs])
    return res


def write_heldout(run_dir: Path, res: HeldoutResult) -> None:
    (run_dir / "heldout.jsonl").write_text("\n".join(r.model_dump_json() for r in res.rows) + "\n", encoding="utf-8")
    summary = res.model_dump(mode="json", exclude={"rows"})
    (run_dir / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_rows(run_dir: Path) -> list[HeldoutRow]:
    p = run_dir / "heldout.jsonl"
    if not p.exists():
        return []
    return [HeldoutRow.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
