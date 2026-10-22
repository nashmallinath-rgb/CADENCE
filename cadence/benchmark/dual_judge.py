"""Dual-judge cross-validation: a second model family re-scores the judged held-out outputs.

Reports inter-judge agreement (kappa, weighted kappa, ICC), member-ranking agreement, and a same-vendor
self-preference gap. Anchored (exact-match) rows are ignored: they have ground truth and no judge.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cadence.benchmark.heldout import HeldoutRow
from cadence.benchmark.stats import bootstrap_ci, mean, rank_agreement
from cadence.cascade.cascade import media_of
from cadence.judge import Judge
from cadence.llm.budget import BudgetExceeded
from cadence.schemas import Example, Output
from cadence.validator import binarize, cohen_kappa
from cadence.validator.agreement import icc_2_1

CAVEAT = ("The same-family judge is the same VENDOR as the generator, not the same model; a gap is suggestive, "
          "not proof of self-preference.")


class DualJudgeResult(BaseModel):
    skipped: bool = False
    reason: str = ""
    n: int = 0
    second_role: str = "second_family_judge"
    same_family_role: str | None = None
    kappa: float | None = None
    kappa_quadratic: float | None = None
    icc: float | None = None
    kappa_min: float = 0.60
    flagged: bool = False
    mean_primary: float | None = None
    mean_second: float | None = None
    member_means: dict[str, dict[str, float]] = Field(default_factory=dict)  # candidate -> {primary, second}
    rank_agreement: dict[str, float | None] = Field(default_factory=dict)
    self_preference: dict[str, Any] = Field(default_factory=dict)
    parse_failures: int = 0
    caveats: list[str] = Field(default_factory=list)


def _output(row: HeldoutRow) -> Output:
    return Output(kind=row.output_kind, text=row.output_text, path=row.output_path)  # type: ignore[arg-type]


def _score(judge: Judge, row: HeldoutRow) -> tuple[float | None, int]:
    out = _output(row)
    try:
        media = media_of(out)
    except OSError:  # image file missing: this row cannot be re-scored
        return None, 0
    text = row.output_text or ("(the response is the attached image)" if media else "")
    r = judge.score(Example(id=row.example_id, input=row.input), text, row.rubric or "", media=media) if media \
        else judge.score(Example(id=row.example_id, input=row.input), text, row.rubric or "")
    return (None if not r.trials else r.score_0_10), r.parse_failures


def run_dual_judge(client: Any, rows: Sequence[HeldoutRow], *, trials: int = 3, second_role: str = "second_family_judge",
                   same_family_role: str | None = "same_family_judge", kappa_min: float = 0.60, seed: int = 0,
                   log: Callable[[str], None] = print) -> DualJudgeResult:
    res = DualJudgeResult(second_role=second_role, same_family_role=same_family_role, kappa_min=kappa_min,
                          caveats=[CAVEAT])
    judged = [r for r in rows if r.judged and r.primary_score is not None and r.rubric]
    if not judged:
        res.skipped, res.reason = True, "no judged held-out rows (the task is scored by an automatic metric)"
        res.caveats = []
        return res
    second: dict[int, float] = {}
    same: dict[int, float] = {}
    try:
        j2 = Judge(client, second_role, trials)
        for i, r in enumerate(judged):
            s, pf = _score(j2, r)
            res.parse_failures += pf
            if s is not None:
                second[i] = s
        if same_family_role:
            try:
                j3 = Judge(client, same_family_role, trials)
                for i in second:
                    s, pf = _score(j3, judged[i])
                    res.parse_failures += pf
                    if s is not None:
                        same[i] = s
            except KeyError:
                res.same_family_role = None
                res.caveats.append(f"role {same_family_role!r} is not configured; self-preference gap not computed")
    except KeyError as exc:
        res.skipped, res.reason = True, f"judge role not configured: {exc}"
        return res
    except BudgetExceeded as exc:
        res.caveats.append(f"stopped early by the budget ({exc}); results use the rows scored so far")
    idx = sorted(second)
    if len(idx) < 2:
        res.skipped, res.reason = True, "fewer than 2 rows could be scored by the second judge"
        return res
    p = [judged[i].primary_score * 10 for i in idx]  # type: ignore[operator]
    q = [second[i] for i in idx]
    res.n = len(idx)
    res.kappa = cohen_kappa(binarize(p), binarize(q))
    res.kappa_quadratic = cohen_kappa([round(x) for x in p], [round(x) for x in q], weights="quadratic")
    try:
        v = icc_2_1([[a, b] for a, b in zip(p, q)])
        res.icc = None if v != v else v
    except ValueError:
        res.icc = None
    res.flagged = res.kappa < kappa_min
    res.mean_primary, res.mean_second = mean(p), mean(q)
    by: dict[str, list[tuple[float, float]]] = {}
    for i, a, b in zip(idx, p, q):
        by.setdefault(judged[i].candidate_id, []).append((a, b))
    res.member_means = {c: {"primary": mean([x for x, _ in v]), "second": mean([y for _, y in v])} for c, v in by.items()}
    cids = sorted(res.member_means)
    res.rank_agreement = rank_agreement([res.member_means[c]["primary"] for c in cids],
                                        [res.member_means[c]["second"] for c in cids])
    if same:
        common = [i for i in idx if i in same]
        gaps = [same[i] - (judged[i].primary_score * 10 + second[i]) / 2 for i in common]  # type: ignore[operator]
        lo, hi = bootstrap_ci(gaps, seed=seed)
        res.self_preference = {"n": len(gaps), "mean_gap": mean(gaps), "ci_low": lo, "ci_high": hi,
                               "significant": bool(lo > 0 or hi < 0)}
    log(f"dual judge: n={res.n} kappa={res.kappa:.2f}" + (" (FLAGGED)" if res.flagged else ""))
    return res


def write_dual_judge(run_dir: Path, res: DualJudgeResult) -> None:
    (run_dir / "dual_judge.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
