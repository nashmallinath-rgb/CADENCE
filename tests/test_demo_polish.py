"""Dual-judge provider failures, the seed baseline in the held-out benchmark, and the 2D headline plot."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadence.benchmark import build_report, evaluate_heldout, run_dual_judge
from cadence.benchmark.dual_judge import MAX_CONSECUTIVE_ERRORS
from cadence.benchmark.heldout import BASELINE_ID
from cadence.benchmark.plots import plot_pareto_front
from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.llm import MockClient
from cadence.schemas import Candidate, Evaluated, GenRequest, Objectives
from test_benchmark import JudgedToy, _client, _judge, _members, _n, _rows  # noqa: F401
from cadence.tasks.toy import ToyTask


class Timeout(Exception):
    """Stands in for litellm.Timeout, which is not a KeyError or a BudgetExceeded."""


def _cascade(client):  # type: ignore[no-untyped-def]
    return Cascade.from_config(client, CascadeConfig())


# ------------------------------------------------------------------ dual judge survives provider errors
def test_dual_judge_timeouts_cost_rows_not_the_benchmark() -> None:
    rows = _rows()

    def flaky(req: GenRequest) -> str:
        if _n(req) % 5 == 0:  # some rows time out, the rest score normally
            raise Timeout("read operation timed out")
        return json.dumps({"score": _n(req) % 11, "confidence": 0.9, "rationale": "s"})

    c = _client()
    c.script["second_family_judge"] = flaky  # type: ignore[index]
    res = run_dual_judge(c, rows, trials=1, log=lambda m: None)
    assert not res.skipped and res.errors > 0 and 2 <= res.n < len(rows)
    assert any("failed" in x and "skipped" in x for x in res.caveats)


def test_dual_judge_gives_up_on_a_dead_judge_instead_of_waiting_on_every_row() -> None:
    rows = _rows()
    calls = {"n": 0}

    def dead(req: GenRequest) -> str:
        calls["n"] += 1
        raise Timeout("read operation timed out")

    c = _client()
    c.script["second_family_judge"] = dead  # type: ignore[index]
    res = run_dual_judge(c, rows, trials=1, log=lambda m: None)
    assert res.skipped and "provider error" in res.reason and res.errors == MAX_CONSECUTIVE_ERRORS
    assert calls["n"] == MAX_CONSECUTIVE_ERRORS < len(rows)  # circuit breaker: no one-timeout-per-row crawl
    assert any("in a row" in x for x in res.caveats)


def test_dual_judge_same_family_timeouts_keep_the_agreement_result() -> None:
    """The exact failure from ifb-live1: the Gemma (same-family) judge timed out and crashed the whole command."""
    def dead(req: GenRequest) -> str:
        raise Timeout("Connection timed out after None seconds")

    c = _client()
    c.script["same_family_judge"] = dead  # type: ignore[index]
    res = run_dual_judge(c, _rows(), trials=1, log=lambda m: None)
    assert not res.skipped and res.kappa is not None and not res.self_preference
    assert res.errors == MAX_CONSECUTIVE_ERRORS and any("self-preference" in x for x in res.caveats)


# ------------------------------------------------------------------ seed baseline in the held-out benchmark
def test_heldout_adds_the_seed_prompt_as_a_baseline() -> None:
    client = MockClient()
    res = evaluate_heldout(ToyTask(), client, _cascade(client), _members(), n=6, workers=1, log=lambda m: None,
                           baseline_template="Reply with a word.")
    base = [s for s in res.members if s.is_baseline]
    assert len(base) == 1 and base[0].candidate_id == BASELINE_ID and base[0].opt_quality is None
    assert base[0].gap is None and base[0].n == 6 and base[0].template == "Reply with a word."
    ids = {r.example_id for r in res.rows if r.candidate_id == BASELINE_ID}
    assert ids == {r.example_id for r in res.rows if r.candidate_id == "m1"}  # same held-out examples


def test_heldout_baseline_already_on_the_front_is_not_run_twice() -> None:
    client = MockClient()
    res = evaluate_heldout(ToyTask(), client, _cascade(client), _members(), n=4, workers=1, log=lambda m: None,
                           baseline_template="Answer the question.")  # identical to member "seed"
    assert len(res.members) == 3 and [s.candidate_id for s in res.members if s.is_baseline] == ["seed"]
    assert not any(r.candidate_id == BASELINE_ID for r in res.rows)


def test_heldout_without_baseline_is_unchanged() -> None:
    client = MockClient()
    res = evaluate_heldout(ToyTask(), client, _cascade(client), _members(), n=4, workers=1, log=lambda m: None)
    assert not any(s.is_baseline for s in res.members) and len(res.members) == 3


# ------------------------------------------------------------------ 2D headline plot
def _ev(i: int, q: float, c: float, n: int) -> Evaluated:
    return Evaluated(candidate=Candidate(id=f"c{i}", template=f"t{i}"),
                     objectives=Objectives(quality=q, cost_usd_per_1k=c, length_tokens=n), n_examples=4)


def _held(base_cost: float = 0.3) -> dict:
    return {"members": [
        {"candidate_id": "c1", "heldout_quality": 0.9, "ci_low": 0.8, "ci_high": 0.97, "cost_usd_per_1k": 0.014},
        {"candidate_id": "c2", "heldout_quality": 0.85, "ci_low": 0.75, "ci_high": 0.93, "cost_usd_per_1k": 0.01},
        {"candidate_id": BASELINE_ID, "is_baseline": True, "heldout_quality": 0.0, "ci_low": 0.0, "ci_high": 0.0,
         "cost_usd_per_1k": base_cost, "length_tokens": 9}]}


@pytest.mark.parametrize("name,members,held", [
    ("with_baseline", [_ev(1, 0.94, 0.014, 15), _ev(2, 0.88, 0.01, 16)], _held()),
    ("zero_cost_falls_back_to_linear", [_ev(1, 0.94, 0.014, 15)], _held(base_cost=0.0)),
    ("no_heldout", [_ev(1, 0.94, 0.014, 15)], None),
    ("single", [_ev(1, 1.0, 2.6, 29)], None),
    ("empty", [], None),
    ("heldout_with_null_numbers", [_ev(1, 0.9, 0.5, 5)], {"members": [{"candidate_id": "c1", "heldout_quality": None}]}),
    ("seed_is_a_member", [_ev(1, 0.9, 0.5, 5)], {"members": [
        {"candidate_id": "c1", "is_baseline": True, "heldout_quality": 0.8, "cost_usd_per_1k": 0.5}]}),
])
def test_front_plot_writes_a_png_for_every_input(tmp_path: Path, name: str, members, held) -> None:  # type: ignore[no-untyped-def]
    p = plot_pareto_front(members, tmp_path / "sub" / f"{name}.png", held)
    assert p.exists() and p.stat().st_size > 2000 and p.read_bytes()[:4] == b"\x89PNG"


# ------------------------------------------------------------------ report and CLI
def test_report_states_seed_versus_best(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    (d / "archive.json").write_text(json.dumps([_ev(1, 0.9, 0.014, 5).model_dump(mode="json")]), encoding="utf-8")
    (d / "heldout_summary.json").write_text(json.dumps({
        "task": "t", "seed": 0, "n_requested": 8, "partial": False, "members": [
            {"candidate_id": "c1", "template": "a", "n": 8, "n_errors": 0, "heldout_quality": 0.9, "ci_low": 0.8,
             "ci_high": 0.95, "opt_quality": 0.9, "gap": 0.0, "cost_usd_per_1k": 0.014, "length_tokens": 5,
             "anchored_fraction": 1.0, "escalation_ratio": 0.0},
            {"candidate_id": BASELINE_ID, "template": "b", "n": 8, "n_errors": 0, "heldout_quality": 0.0, "ci_low": 0.0,
             "ci_high": 0.0, "opt_quality": None, "gap": None, "is_baseline": True, "cost_usd_per_1k": 0.28,
             "length_tokens": 9, "anchored_fraction": 1.0, "escalation_ratio": 0.0}]}), encoding="utf-8")
    text = build_report(d)
    assert "(seed baseline)" in text and "Seed vs best optimised prompt" in text
    assert "0.000 to 0.900" in text and "-95%" in text


def test_cli_benchmark_includes_baseline_and_headline_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    import cadence.config as config
    from cadence.cli import main

    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert main(["run", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "2", "--minibatch", "8",
                 "--run-id", "r1", "--seed-prompt", "Do it."]) == 0
    assert main(["benchmark", "--run", "r1", "--mock", "--n", "6"]) == 0
    out = capsys.readouterr().out
    assert "seed prompt as a baseline" in out
    summary = json.loads((tmp_path / "runs/r1/heldout_summary.json").read_text())
    assert sum(m["is_baseline"] for m in summary["members"]) == 1
    assert (tmp_path / "runs/r1/pareto_front.png").exists()
    assert "pareto_front.png" in (tmp_path / "runs/r1/report.md").read_text(encoding="utf-8")
    assert main(["benchmark", "--run", "r1", "--mock", "--n", "6", "--no-baseline", "--no-plots"]) == 0
    summary = json.loads((tmp_path / "runs/r1/heldout_summary.json").read_text())
    assert not any(m["is_baseline"] for m in summary["members"])
