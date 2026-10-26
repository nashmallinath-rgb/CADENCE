"""Phase 5: generator, evaluator, loop, tracking. Everything offline (MockClient + ToyTask)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.llm import BudgetGuard, MockClient
from cadence.orchestrator import (EvaluationFailed, Evaluator, Generator, LoopSettings, extract_template, optimize,
                                  pareto_menu, template_hash)
from cadence.orchestrator.loop import minibatch_for, reference_point, select_parents
from cadence.orchestrator.mock_reflector import toy_reflector
from cadence.pareto import ParetoArchive
from cadence.schemas import Candidate, Evaluated, Objectives
from cadence.tasks.toy import ToyTask
from cadence.tracking import MlflowTracker, NullTracker


def _client(budget: BudgetGuard | None = None) -> MockClient:
    return MockClient(script={"reflector": toy_reflector}, budget=budget)


def _cascade(client: MockClient) -> Cascade:
    return Cascade.from_config(client, CascadeConfig())


def _ev(cid: str, q: float, c: float, n: int, tpl: str = "t") -> Evaluated:
    return Evaluated(candidate=Candidate(id=cid, template=tpl), objectives=Objectives(quality=q, cost_usd_per_1k=c,
                                                                                    length_tokens=n), n_examples=4)


# ---------------------------------------------------------------- generator
def test_extract_template_variants() -> None:
    assert extract_template("blah <template>  Do X.  </template> tail") == "Do X."
    assert extract_template("```text\nDo X.\n```") == "Do X."
    assert extract_template("  plain reply ") == "plain reply"
    assert extract_template("<TEMPLATE>a\nb</TEMPLATE>") == "a\nb"


def test_template_hash_ignores_case_and_whitespace() -> None:
    assert template_hash("Be  Concise.") == template_hash("be concise.\n")
    assert template_hash("a") != template_hash("b")


def test_schedule_skips_crossover_with_one_parent_and_is_deterministic() -> None:
    g = Generator(_client(), seed=3)
    assert "crossover" not in g.schedule(8, 1, gen=1)
    assert "crossover" in g.schedule(4, 2, gen=1)
    assert g.schedule(4, 2, gen=1) == g.schedule(4, 2, gen=1)


def test_propose_distinct_valid_children_and_updates_seen() -> None:
    client = _client()
    parents = [_ev("seed", 0.3, 0.1, 4, "Answer the question.")]
    seen = {template_hash("Answer the question.")}
    kids = Generator(client, seed=0).propose(parents, {}, 4, gen=1, seen=seen)
    assert kids and len({k.template for k in kids}) == len(kids)
    assert all(k.parent_ids == ["seed"] and k.generation == 1 for k in kids)
    assert all(template_hash(k.template) in seen for k in kids)
    assert all(k.template.strip() != "Answer the question." for k in kids)
    assert all(c.role == "reflector" for c in client.calls)


def test_propose_rejects_empty_duplicate_and_overlong() -> None:
    parents = [_ev("seed", 0.3, 0.1, 4, "Answer.")]
    replies = iter(["<template></template>", "<template>Answer.</template>", "<template>" + "word " * 500 + "</template>",
                    "<template>Fresh idea.</template>"])
    client = MockClient(script={"reflector": lambda req: next(replies)})
    seen = {template_hash("Answer.")}
    kids = Generator(client, seed=0).propose(parents, {}, 4, gen=1, seen=seen)
    assert [k.template for k in kids] == ["Fresh idea."]


def test_propose_prompt_carries_parent_and_worst_traces_but_no_reference() -> None:
    from cadence.orchestrator import Trace

    client = _client()
    parents = [_ev("seed", 0.3, 0.1, 4, "Answer the question.")]
    tr = {"seed": [Trace(example_id="x", input="Toy question 7", output="WRONG", quality=0.0)]}
    Generator(client, seed=0).propose(parents, tr, 4, gen=1, seen=set())
    texts = [c.messages[-1]["content"] for c in client.calls]
    assert any("Toy question 7" in t for t in texts)  # reflect/strengthen see failures
    assert all("Answer the question." in t for t in texts)
    assert not any("a7" in t.split("Toy question 7")[-1][:5] for t in texts)


def test_propose_requires_parents() -> None:
    with pytest.raises(ValueError):
        Generator(_client()).propose([], {}, 2, 1, set())


# ---------------------------------------------------------------- evaluator
def test_evaluator_objectives_and_deterministic_across_workers() -> None:
    task, exs = ToyTask(), ToyTask().load("opt", n=12)
    cand = Candidate(id="c", template="Answer the question. Step by step.")
    r1 = Evaluator(task, _client(), _cascade(_client()), workers=1).evaluate(cand, exs)
    r4 = Evaluator(task, _client(), _cascade(_client()), workers=4).evaluate(cand, exs)
    assert r1.evaluated.objectives == r4.evaluated.objectives
    o = r1.evaluated.objectives
    assert 0 <= o.quality <= 1 and o.cost_usd_per_1k > 0 and o.length_tokens > 0
    assert r1.evaluated.n_examples == 12 and len(r1.traces) == 12 and r1.n_judged == 0


def test_cost_objective_is_generation_cost_only() -> None:
    class Judged(ToyTask):
        def automatic_score(self, ex, out):  # force the judge path
            return None

        def rubric(self, ex):
            return "rate it"

    client = _client()
    res = Evaluator(Judged(), client, _cascade(client), workers=2).evaluate(
        Candidate(id="c", template="Answer."), ToyTask().load("opt", n=6))
    assert res.n_judged == 6 and res.judge_usd == 0.0  # mock judge is free
    base = Evaluator(ToyTask(), client, _cascade(client)).evaluate(Candidate(id="c", template="Answer."),
                                                                    ToyTask().load("opt", n=6))
    assert res.evaluated.objectives.cost_usd_per_1k == pytest.approx(base.evaluated.objectives.cost_usd_per_1k)


def test_evaluator_error_handling() -> None:
    class Flaky(ToyTask):
        def __init__(self, bad: int) -> None:
            super().__init__()
            self.bad = bad

        def generate(self, client, req):
            if int(req.messages[-1]["content"].split()[-1]) % self.bad == 0:
                raise RuntimeError("provider hiccup")
            return super().generate(client, req)

    exs = ToyTask().load("opt", n=20)
    c = Candidate(id="c", template="Answer.")
    client = _client()
    n_bad = sum(int(e.input["question"].split()[-1]) % 10 == 0 for e in exs)
    assert n_bad / 20 <= 0.25
    ok = Evaluator(Flaky(10), client, _cascade(client)).evaluate(c, exs)
    assert ok.n_errors == n_bad and ok.evaluated.n_examples == 20 - n_bad
    with pytest.raises(EvaluationFailed):
        Evaluator(Flaky(1), client, _cascade(client)).evaluate(c, exs)  # everything fails


def test_evaluator_propagates_budget_exceeded() -> None:
    from cadence.llm import BudgetExceeded

    class Pricey(ToyTask):
        def generate(self, client, req):
            client.budget.check(est_tokens=1)
            client.budget.record(__import__("cadence.schemas", fromlist=["Usage"]).Usage(input_tokens=1))
            return super().generate(client, req)

    client = _client(BudgetGuard(10, 3, 10_000))
    with pytest.raises(BudgetExceeded):
        Evaluator(Pricey(), client, _cascade(client), workers=2).evaluate(Candidate(id="c", template="x"),
                                                                           ToyTask().load("opt", n=8))


# ---------------------------------------------------------------- loop helpers
def test_minibatch_rotates_and_wraps() -> None:
    pool = ToyTask().load("opt", n=10)
    a, b = minibatch_for(pool, 0, 4), minibatch_for(pool, 1, 4)
    assert [e.id for e in a] != [e.id for e in b] and len(a) == len(b) == 4
    assert len(minibatch_for(pool, 3, 4)) == 4  # wraps past the end
    assert len(minibatch_for(pool, 0, 99)) == 10


def test_select_parents_distinct_and_deterministic() -> None:
    members = [_ev(f"m{i}", 0.1 * i, 1.0 - 0.1 * i, 10 + i) for i in range(1, 8)]
    p1 = select_parents(members, 3, random.Random(1))
    p2 = select_parents(members, 3, random.Random(1))
    assert [m.candidate.id for m in p1] == [m.candidate.id for m in p2]
    assert len({m.candidate.id for m in p1}) == 3
    assert len(select_parents(members[:2], 3, random.Random(1))) == 2


def test_reference_point_is_beyond_seed() -> None:
    ref = reference_point(_ev("seed", 0.3, 0.2, 10))
    assert ref.quality == 0 and ref.cost_usd_per_1k > 0.2 and ref.length_tokens > 10
    assert reference_point(_ev("seed", 0.3, 0.0, 10)).cost_usd_per_1k > 0  # free models still get a valid ref


# ---------------------------------------------------------------- e2e
SETTINGS = LoopSettings(seed=0, max_gen=6, children_per_gen=4, minibatch=16, final_sample=48, workers=3,
                        parents_per_gen=2)


def _run(tmp: Path, name: str, settings: LoopSettings = SETTINGS, budget: BudgetGuard | None = None, **kw):
    client = _client(budget)
    return optimize(ToyTask(), client, _cascade(client), settings, tmp / name, log=lambda m: None, **kw), client


@pytest.mark.e2e
def test_e2e_front_improves_and_run_dir_is_written(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, "r")
    assert res.stopped_reason == "max_gen" and res.generations_done == SETTINGS.max_gen
    assert res.hv_history == sorted(res.hv_history)  # archive hypervolume never decreases during the search
    assert res.hv_history[-1] > res.hv_history[0]
    assert len(res.archive) > 1  # a real trade-off front, not a single point
    d = tmp_path / "r"
    for name in ("config.json", "state.json", "archive.json", "archive_search.json", "trace.jsonl", "menu.md"):
        assert (d / name).exists(), name
    arch = ParetoArchive.from_json((d / "archive.json").read_text())
    assert {m.candidate.id for m in arch} == {m.candidate.id for m in res.archive}
    events = [json.loads(line) for line in (d / "trace.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "stop" and res.final_reevaluated
    assert sum(e["event"] == "generation" for e in events) == SETTINGS.max_gen + 1
    best = max(m.objectives.quality for m in res.archive)
    assert best > 0.40  # the toy seed scores ~0.3; reflection must find the helpful phrases
    assert "| quality |" in pareto_menu(res.archive)


@pytest.mark.e2e
def test_e2e_deterministic_for_a_seed(tmp_path: Path) -> None:
    a, _ = _run(tmp_path, "a")
    b, _ = _run(tmp_path, "b")
    key = lambda r: sorted((m.candidate.template, round(m.objectives.quality, 9)) for m in r.archive)  # noqa: E731
    assert key(a) == key(b) and a.hv_history == b.hv_history
    c, _ = _run(tmp_path, "c", SETTINGS.model_copy(update={"seed": 5}))
    assert key(c) != key(a) or c.hv_history != a.hv_history  # a different seed explores differently


@pytest.mark.e2e
def test_e2e_budget_stops_the_loop_and_keeps_results(tmp_path: Path) -> None:
    # reflector calls count against the budget; a small call cap must end the run early but cleanly
    res, client = _run(tmp_path, "b", budget=BudgetGuard(10.0, 12, 10**9))
    assert res.stopped_reason.startswith("budget")
    assert res.generations_done < SETTINGS.max_gen
    assert client.budget.calls <= 12  # never past the cap
    assert res.archive and (tmp_path / "b" / "archive.json").exists()
    events = [json.loads(line) for line in (tmp_path / "b" / "trace.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "stop" and events[-1]["reason"].startswith("budget")


@pytest.mark.e2e
def test_e2e_resume_continues_and_refuses_mismatches(tmp_path: Path) -> None:
    short = SETTINGS.model_copy(update={"max_gen": 2})
    r1, _ = _run(tmp_path, "r", short)
    assert r1.generations_done == 2
    with pytest.raises(FileExistsError):
        _run(tmp_path, "r", short)  # silently overwriting a run is not allowed
    with pytest.raises(ValueError):  # a different seed is a different search; only max_gen may change
        _run(tmp_path, "r", SETTINGS.model_copy(update={"max_gen": 4, "seed": 9}), resume=True)
    r2, _ = _run(tmp_path, "r", short.model_copy(update={"max_gen": 2}), resume=True)
    assert r2.resumed and r2.generations_done == 2 and len(r2.archive) >= 1


@pytest.mark.e2e
def test_e2e_resume_with_larger_max_gen_is_allowed(tmp_path: Path) -> None:
    short = SETTINGS.model_copy(update={"max_gen": 2})
    _run(tmp_path, "r", short)
    res, _ = _run(tmp_path, "r", SETTINGS.model_copy(update={"max_gen": 4}), resume=True)
    assert res.resumed and res.generations_done == 4 and len(res.hv_history) == 5


# ---------------------------------------------------------------- tracking + CLI
def test_mlflow_tracker_logs_params_metrics_artifacts(tmp_path: Path) -> None:
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'ml.db').as_posix()}"
    t = MlflowTracker(uri, experiment="cadence-test")
    art = tmp_path / "a.txt"
    art.write_text("hi")
    t.start("run-x", {"seed": 0, "task": "toy"})
    t.log_generation(0, {"front_size": 3, "kappa": None, "hypervolume": 1.5})
    t.log_artifact(art)
    t.end()
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["cadence-test"])
    assert len(runs) == 1 and runs.iloc[0]["metrics.front_size"] == 3 and runs.iloc[0]["params.task"] == "toy"
    assert "metrics.kappa" not in runs.columns or runs.iloc[0]["metrics.kappa"] != runs.iloc[0]["metrics.kappa"]


def test_null_tracker_is_inert() -> None:
    t = NullTracker()
    t.start("x", {})
    t.log_generation(0, {"a": 1.0})
    t.end()


def test_loop_marks_tracker_failed_on_error(tmp_path: Path) -> None:
    class Rec(NullTracker):
        status = ""

        def end(self, status: str = "FINISHED") -> None:
            Rec.status = status

    class Boom(ToyTask):
        def generate(self, client, req):
            raise RuntimeError("boom")

    client = _client()
    with pytest.raises(EvaluationFailed):
        optimize(Boom(), client, _cascade(client), SETTINGS, tmp_path / "x", tracker=Rec(), log=lambda m: None)
    assert Rec.status == "FAILED"


def test_cli_run_mock_prints_pareto_menu(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch) -> None:
    import cadence.config as config
    from cadence.cli import main

    monkeypatch.setattr(config, "ROOT", tmp_path)
    rid = "cli-test"
    rc = main(["run", "--stream", "text", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "3",
               "--minibatch", "8", "--run-id", rid])
    out = capsys.readouterr().out
    assert rc == 0 and "Pareto menu" in out and "| quality |" in out
    assert (tmp_path / "runs" / rid / "archive.json").exists()
    assert main(["run", "--task", "toy", "--mock", "--no-mlflow", "--run-id", rid]) == 1  # existing run, no --resume


def test_cli_rejects_stream_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    from cadence.cli import main

    assert main(["run", "--stream", "image", "--task", "toy", "--mock"]) == 1
    assert "stream" in capsys.readouterr().out


def test_cli_final_sample_flag_reaches_the_loop(tmp_path: Path, monkeypatch) -> None:
    import cadence.config as config
    from cadence.cli import main

    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert main(["run", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "1", "--minibatch", "4",
                 "--final-sample", "6", "--run-id", "fs", "--no-report"]) == 0
    cfg = json.loads((tmp_path / "runs/fs/config.json").read_text())
    assert cfg["settings"]["final_sample"] == 6


def test_weak_seed_prompt_is_used_and_hidden_hint_does_not_leak_default(tmp_path: Path) -> None:
    client = _client()
    res = optimize(ToyTask(), client, _cascade(client), SETTINGS.model_copy(update={"max_gen": 1}), tmp_path / "w",
                   seed_template="Weak start.", log=lambda m: None)
    assert res.archive
    cfg = json.loads((tmp_path / "w/config.json").read_text())
    assert cfg["seed_template"] == "Weak start."
    first = next(json.loads(line) for line in (tmp_path / "w/trace.jsonl").read_text().splitlines()
                 if json.loads(line).get("id") == "seed")
    assert first["template"] == "Weak start."
    assert all("Answer the question." not in c.messages[-1]["content"] for c in client.calls if c.role == "reflector")


def test_resume_requires_the_same_seed_template(tmp_path: Path) -> None:
    c = _client()
    short = SETTINGS.model_copy(update={"max_gen": 1})
    optimize(ToyTask(), c, _cascade(c), short, tmp_path / "r", seed_template="A.", log=lambda m: None)
    with pytest.raises(ValueError):
        optimize(ToyTask(), c, _cascade(c), short, tmp_path / "r", seed_template="B.", resume=True, log=lambda m: None)


def test_cli_seed_prompt_options(tmp_path: Path, monkeypatch, capsys) -> None:
    import cadence.config as config
    from cadence.cli import main

    monkeypatch.setattr(config, "ROOT", tmp_path)
    base = ["run", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "1", "--minibatch", "4", "--no-report"]
    assert main(base + ["--seed-prompt", "weak", "--run-id", "a"]) == 1  # ToyTask has no weak prompt
    assert "no weak seed" in capsys.readouterr().out
    assert main(base + ["--seed-prompt", "Custom start.", "--run-id", "b"]) == 0
    assert json.loads((tmp_path / "runs/b/config.json").read_text())["seed_template"] == "Custom start."
    from cadence.tasks import get_task

    for name in ("gsm8k", "sst2", "hotpotqa", "ifbench", "code_web", "audio"):
        assert get_task(name).weak_seed_prompt()


class _Flaky(ToyTask):
    """Fails for the given template (provider hiccup), works otherwise."""

    def __init__(self, bad_template: str) -> None:
        super().__init__()
        self.bad = bad_template

    def generate(self, client, req):  # type: ignore[no-untyped-def]
        if req.messages[0]["content"] == self.bad:
            raise RuntimeError("429 quota exceeded for tts")
        return super().generate(client, req)


def test_evaluation_failure_names_the_underlying_error() -> None:
    client = _client()
    with pytest.raises(EvaluationFailed, match="429 quota exceeded"):
        Evaluator(_Flaky("x"), client, _cascade(client)).evaluate(Candidate(id="c", template="x"),
                                                                  ToyTask().load("opt", n=4))
    ok = Evaluator(_Flaky("zzz"), client, _cascade(client)).evaluate(Candidate(id="c", template="x"),
                                                                    ToyTask().load("opt", n=4))
    assert ok.error_samples == []


def test_final_reevaluation_keeps_search_numbers_for_a_member_that_fails(tmp_path: Path) -> None:
    pool_ids = [e.id for e in ToyTask().load("opt")]
    first_four = set(pool_ids[:4])

    class SeedBreaksOnBigSample(ToyTask):
        """The seed prompt only works on the generation-0 minibatch (first 4 examples), so the 8-example final fails."""

        def generate(self, client, req):  # type: ignore[no-untyped-def]
            q = req.messages[-1]["content"]
            ex_id = "toy-" + q.rsplit(" ", 1)[-1]
            if req.messages[0]["content"] == "Answer the question." and ex_id not in first_four:
                raise RuntimeError("tts 429")
            return super().generate(client, req)

    client = _client()
    res = optimize(SeedBreaksOnBigSample(), client, _cascade(client),
                   SETTINGS.model_copy(update={"max_gen": 3, "minibatch": 4, "final_sample": 8}), tmp_path / "r",
                   log=lambda m: None)
    events = [json.loads(line) for line in (tmp_path / "r/trace.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["event"] == "final_reeval_member_failed"]
    assert failed and failed[0]["id"] == "seed" and "tts 429" in failed[0]["reason"]
    by_id = {m.candidate.id: m for m in res.archive}
    assert by_id["seed"].n_examples == 4  # kept its search-time numbers
    assert any(m.n_examples == 8 for m in res.archive if m.candidate.id != "seed")  # the rest were re-evaluated
    assert res.final_reevaluated
