"""Live (or mock) smoke test of the model path: generation, scoring, and judge trustworthiness.

Stages per task: (1) the generator answers n calib-split examples with the task's seed prompt and is scored by exact
match; (2) a calibration set is built from those outputs plus deliberately corrupted copies, all labelled by exact
match; (3) every judge role rates the set reference-free and is compared with ground truth (Cohen's kappa, ICC);
(4) the cascade is run over the same set. Corrupted items are easy, so kappa here is optimistic; the genuine
generator mistakes are the real test.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.judge import Judge
from cadence.llm.budget import BudgetExceeded
from cadence.schemas import Example, Output, Usage
from cadence.tasks.base import Task
from cadence.validator import CalibItem, binarize, calibrate, cohen_kappa

JUDGE_ROLES = ("cheap_judge", "strong_judge", "second_family_judge")


class _ItemScorer:
    """Adapter so the cascade can judge a calibration item (no automatic score, task-specific rubric)."""

    def __init__(self, rubric: str) -> None:
        self._rubric = rubric

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return None

    def rubric(self, ex: Example) -> str | None:
        return self._rubric


def _stage(log: Callable[[str], None], msg: str) -> None:
    log(f"  {msg}")


def run_smoke(client: Any, tasks: Mapping[str, Task], *, n: int = 10, trials: int = 2,
              cascade_cfg: CascadeConfig | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = cascade_cfg or CascadeConfig()
    report: dict[str, Any] = {"n": n, "trials": trials, "tasks": {}, "errors": []}
    t_start = time.perf_counter()

    for name, task in tasks.items():
        log(f"[{name}] generating {n} answers with the seed prompt...")
        examples = task.load("calib", n=n)
        genuine: list[tuple[Example, Output]] = []
        gen_secs, usages = 0.0, Usage()
        for ex in examples:
            t0 = time.perf_counter()
            try:
                out = task.generate(client, task.build_request(task.seed_prompt(), ex))
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - a smoke test reports failures instead of dying
                report["errors"].append(f"{name} generate {ex.id}: {type(exc).__name__}")
                continue
            gen_secs += time.perf_counter() - t0
            usages = usages + out.usage
            genuine.append((ex, out))
        scores = [task.automatic_score(ex, out) or 0.0 for ex, out in genuine]
        k = max(1, len(genuine))
        trow: dict[str, Any] = {
            "generated": len(genuine),
            "accuracy": sum(scores) / k if genuine else None,
            "seconds_per_call": gen_secs / k,
            "avg_input_tokens": usages.input_tokens / k,
            "avg_output_tokens": usages.output_tokens / k,
            "cost_usd_per_1k": 1000 * usages.usd / k,
        }
        report["tasks"][name] = trow

        # calibration set: genuine outputs + corrupted copies, all labelled by exact match
        rubric_for = task.calibration_rubric
        items: list[CalibItem] = []
        for ex, out in genuine:
            rub = rubric_for(ex)
            if rub is None:
                continue
            items.append(CalibItem(example=ex, response_text=out.text or "", rubric=rub,
                                   correct=(task.automatic_score(ex, out) or 0.0) >= 0.5))
            bad = task.corrupt_output(ex, out.text or "")
            if bad is not None:
                bad_out = Output(kind="text", text=bad)
                items.append(CalibItem(example=ex, response_text=bad, rubric=rub,
                                       correct=(task.automatic_score(ex, bad_out) or 0.0) >= 0.5))
        n_correct = sum(i.correct for i in items)
        trow["calibration_items"] = len(items)
        trow["calibration_correct"] = n_correct
        if len(items) < 4 or n_correct in (0, len(items)):
            report["errors"].append(f"{name}: calibration set unusable ({len(items)} items, {n_correct} correct)")
            continue

        judges: dict[str, Any] = {}
        health_scores: dict[str, list[float]] = {}
        for role in JUDGE_ROLES:
            log(f"[{name}] calibrating {role} on {len(items)} items x {trials} trials...")
            t0 = time.perf_counter()
            try:
                h = calibrate(Judge(client, role, trials), items, kappa_min=cfg.kappa_min)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"{name} {role}: {type(exc).__name__}: {str(exc)[:120]}")
                judges[role] = {"error": type(exc).__name__}
                continue
            health_scores[role] = h.scores
            judges[role] = {"kappa": h.kappa, "icc": h.icc, "flagged": h.flagged, "n": h.n,
                            "seconds": time.perf_counter() - t0}
        trow["judges"] = judges

        if "cheap_judge" in health_scores and "second_family_judge" in health_scores:
            trow["inter_judge_kappa"] = cohen_kappa(binarize(health_scores["cheap_judge"]),
                                                    binarize(health_scores["second_family_judge"]))

        log(f"[{name}] running the cascade...")
        cas = Cascade(Judge(client, "cheap_judge", trials), Judge(client, "strong_judge", trials), cfg.tau, cfg.delta)
        try:
            preds = [int(cas.score(_ItemScorer(i.rubric), i.example, Output(kind="text", text=i.response_text)).quality >= 0.5)
                     for i in items]
            trow["cascade"] = {"escalation_ratio": cas.stats.escalation_ratio,
                               "kappa": cohen_kappa([int(i.correct) for i in items], preds)}
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{name} cascade: {type(exc).__name__}")

    stats = getattr(client, "stats", None)
    budget = getattr(client, "budget", None)
    report["spend"] = {
        "seconds": time.perf_counter() - t_start,
        "live_calls": getattr(stats, "live_calls", None), "cache_hits": getattr(stats, "cache_hits", None),
        "retries": getattr(stats, "retries", None),
        "budget_calls": getattr(budget, "calls", None), "budget_tokens": getattr(budget, "tokens", None),
        "budget_usd": getattr(budget, "spent_usd", None),
    }
    return report


def format_report(rep: dict[str, Any], kappa_min: float = 0.60) -> str:
    def f(x: Any, spec: str = ".2f") -> str:
        return "n/a" if x is None else format(x, spec)

    lines = [f"\nSMOKE TEST  n={rep['n']} examples/task, {rep['trials']} judge trials", ""]
    for name, t in rep["tasks"].items():
        lines.append(f"== {name} ==")
        lines.append(f"  generation: accuracy {f(t.get('accuracy'))} on {t['generated']} answers | "
                     f"{f(t['seconds_per_call'], '.1f')}s/call | avg {f(t['avg_input_tokens'], '.0f')} in / "
                     f"{f(t['avg_output_tokens'], '.0f')} out tokens | ${f(t['cost_usd_per_1k'], '.3f')} per 1k runs (list price)")
        if "calibration_items" in t:
            lines.append(f"  calibration set: {t['calibration_items']} items ({t['calibration_correct']} correct)")
        for role, j in t.get("judges", {}).items():
            if "error" in j:
                lines.append(f"  {role:20} ERROR {j['error']}")
            else:
                lines.append(f"  {role:20} kappa {j['kappa']:.2f}  icc {f(j['icc'])}  "
                             f"{'FLAGGED' if j['flagged'] else 'ok':8} ({j['seconds']:.0f}s)")
        if "inter_judge_kappa" in t:
            lines.append(f"  inter-judge kappa (cheap vs second-family): {t['inter_judge_kappa']:.2f}")
        if "cascade" in t:
            c = t["cascade"]
            lines.append(f"  cascade: escalation ratio {c['escalation_ratio']:.0%}, kappa vs truth {c['kappa']:.2f}")
        lines.append("")
    s = rep["spend"]
    na = lambda x: "n/a" if x is None else x  # noqa: E731 - the mock client keeps no stats
    lines.append(f"spend: {na(s['live_calls'])} live calls, {na(s['cache_hits'])} cache hits, {na(s['retries'])} retries, "
                 f"{na(s['budget_tokens'])} tokens, ${f(s['budget_usd'], '.4f')} (list price), {s['seconds']:.0f}s total")
    if rep["errors"]:
        lines.append("\nERRORS:")
        lines += [f"  - {e}" for e in rep["errors"][:20]]
    flagged = [(n, r) for n, t in rep["tasks"].items() for r, j in t.get("judges", {}).items() if j.get("flagged")]
    lines.append("\nVERDICT")
    lines.append(("  FAIL: judges below kappa %.2f: " % kappa_min + ", ".join(f"{n}/{r}" for n, r in flagged))
                 if flagged else ("  Judges all clear kappa %.2f." % kappa_min if rep["tasks"] else "  Nothing ran."))
    if rep["errors"]:
        lines.append("  WARN: errors above; fix before trusting the numbers.")
    return "\n".join(lines)
