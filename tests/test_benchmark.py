"""Phase 8: stats, held-out evaluation, dual judge, report, calibration command, CLI. Offline only."""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from cadence.benchmark import (DualJudgeResult, build_report, evaluate_heldout, read_rows, run_dual_judge,
                               write_dual_judge, write_heldout, write_report)
from cadence.benchmark.stats import bootstrap_ci, mean, rank_agreement
from cadence.calibration import run_calibration, supports_calibration
from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.llm import BudgetGuard, MockClient
from cadence.schemas import Candidate, Evaluated, GenRequest, Objectives
from cadence.tasks.text import Sst2Task
from cadence.tasks.toy import ToyTask

HEADINGS = ["## 1. Run", "## 2. Pareto menu", "## 3. Judge reliability", "## 4. Cascade", "## 5. Spend versus budget",
            "## 6. Held-out benchmark and anchoring", "## 7. Dual-judge cross-validation and self-preference",
            "## 8. Streams demonstrated and limitations"]


class JudgedToy(ToyTask):
    """ToyTask that needs a judge, so the dual-judge path is exercised."""

    name = "toy"

    def automatic_score(self, ex, out):  # type: ignore[no-untyped-def]
        return None

    def rubric(self, ex):  # type: ignore[no-untyped-def]
        return "Rate the answer from 0 to 10."


def _n(req: GenRequest) -> int:
    return int(re.search(r"Toy question (\d+)", req.messages[-1]["content"]).group(1))  # type: ignore[union-attr]


def _judge(fn):  # type: ignore[no-untyped-def]
    return lambda req: json.dumps({"score": fn(_n(req)), "confidence": 0.9, "rationale": "scripted"})


def _client(second=lambda n: n % 11, same=lambda n: n % 11, **kw):  # type: ignore[no-untyped-def]
    base = lambda n: n % 11  # noqa: E731
    script = {"cheap_judge": _judge(base), "strong_judge": _judge(base), "second_family_judge": _judge(second),
              "same_family_judge": _judge(same)}
    return MockClient(script=script, **kw)


def _members() -> list[Evaluated]:
    tpls = {"seed": "Answer the question.", "m1": "Answer. Step by step.", "m2": "Answer."}
    return [Evaluated(candidate=Candidate(id=k, template=v),
                      objectives=Objectives(quality=q, cost_usd_per_1k=c, length_tokens=n), n_examples=8)
            for (k, v), q, c, n in zip(tpls.items(), (0.30, 0.55, 0.20), (0.2, 0.9, 0.1), (4, 6, 2))]


def _cascade(client) -> Cascade:  # type: ignore[no-untyped-def]
    return Cascade.from_config(client, CascadeConfig())


# ------------------------------------------------------------------ stats
def test_bootstrap_ci_is_seeded_and_ordered() -> None:
    v = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    a, b = bootstrap_ci(v, seed=1), bootstrap_ci(v, seed=1)
    assert a == b and a[0] <= mean(v) <= a[1]
    assert bootstrap_ci([0.4]) == (0.4, 0.4)
    lo, hi = bootstrap_ci([])
    assert lo != lo and hi != hi  # NaN for no data


def test_rank_agreement_cases() -> None:
    assert rank_agreement([1, 2, 3, 4], [10, 20, 30, 40])["spearman"] == pytest.approx(1.0)
    assert rank_agreement([1, 2, 3, 4], [4, 3, 2, 1])["kendall"] == pytest.approx(-1.0)
    assert rank_agreement([1, 2], [1, 2]) == {"spearman": None, "kendall": None}  # too few
    assert rank_agreement([1, 1, 1], [1, 2, 3]) == {"spearman": None, "kendall": None}  # constant
    with pytest.raises(ValueError):
        rank_agreement([1, 2, 3], [1, 2])


# ------------------------------------------------------------------ held-out
def test_heldout_same_examples_anchored_and_deterministic(tmp_path: Path) -> None:
    task, client = ToyTask(), MockClient()
    r1 = evaluate_heldout(task, client, _cascade(client), _members(), n=10, seed=0, workers=1, log=lambda m: None)
    r4 = evaluate_heldout(task, client, _cascade(client), _members(), n=10, seed=0, workers=4, log=lambda m: None)
    assert [x.model_dump() for x in r1.rows] == [x.model_dump() for x in r4.rows]
    ids = {m: {r.example_id for r in r1.rows if r.candidate_id == m} for m in ("seed", "m1", "m2")}
    assert ids["seed"] == ids["m1"] == ids["m2"] and len(ids["seed"]) == 10
    assert all(r.auto_score is not None and not r.judged and r.rubric is None for r in r1.rows)  # anchored
    assert all(s.anchored_fraction == 1.0 and s.n == 10 for s in r1.members)
    seed_m = next(s for s in r1.members if s.candidate_id == "seed")
    q = [r.primary_score for r in r1.rows if r.candidate_id == "seed"]
    assert seed_m.heldout_quality == pytest.approx(sum(q) / 10)
    assert seed_m.gap == pytest.approx(0.30 - seed_m.heldout_quality)  # type: ignore[operator]
    assert seed_m.ci_low <= seed_m.heldout_quality <= seed_m.ci_high  # type: ignore[operator]
    write_heldout(tmp_path, r1)
    assert [x.model_dump() for x in read_rows(tmp_path)] == [x.model_dump() for x in r1.rows]
    assert "rows" not in json.loads((tmp_path / "heldout_summary.json").read_text())
    assert read_rows(tmp_path / "missing") == []


def test_heldout_judged_rows_keep_rubric_and_do_not_blend_auto() -> None:
    client = _client()
    res = evaluate_heldout(JudgedToy(), client, _cascade(client), _members(), n=6, workers=2, log=lambda m: None)
    assert all(r.judged and r.auto_score is None and r.rubric for r in res.rows)
    assert all(s.anchored_fraction == 0.0 for s in res.members)


def test_heldout_example_errors_cost_a_row_not_the_run() -> None:
    class Flaky(ToyTask):
        def generate(self, client, req):  # type: ignore[no-untyped-def]
            if req.messages[-1]["content"].endswith(("0", "5")):
                raise RuntimeError("provider hiccup")
            return super().generate(client, req)

    client = MockClient()
    res = evaluate_heldout(Flaky(), client, _cascade(client), _members()[:1], n=20, workers=2, log=lambda m: None)
    bad = [r for r in res.rows if r.error]
    assert bad and all(r.primary_score is None for r in bad)
    s = res.members[0]
    assert s.n_errors == len(bad) and s.n == 20 - len(bad) and not res.partial


def test_heldout_budget_stop_is_partial_not_a_crash() -> None:
    class Chatty(ToyTask):
        def generate(self, client, req):  # type: ignore[no-untyped-def]
            client.complete(GenRequest(role="generator", messages=req.messages))
            return super().generate(client, req)

    client = MockClient(budget=BudgetGuard(10.0, 5, 10**9))
    res = evaluate_heldout(Chatty(), client, _cascade(client), _members(), n=8, workers=1, log=lambda m: None)
    assert res.partial and res.stop_reason == "budget:calls"
    assert 0 < len(res.rows) < 24


def test_heldout_needs_examples() -> None:
    class Empty(ToyTask):
        def load(self, split, n=None, seed=0):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(ValueError):
        evaluate_heldout(Empty(), MockClient(), _cascade(MockClient()), _members(), n=5, log=lambda m: None)


# ------------------------------------------------------------------ dual judge
def _rows(client=None):  # type: ignore[no-untyped-def]
    c = client or _client()
    return evaluate_heldout(JudgedToy(), c, _cascade(c), _members(), n=22, workers=1, log=lambda m: None).rows


def test_dual_judge_agreeing_judges() -> None:
    rows = _rows()
    res = run_dual_judge(_client(), rows, trials=2, log=lambda m: None)
    assert not res.skipped and res.n == len(rows)
    assert res.kappa == pytest.approx(1.0) and not res.flagged
    assert res.kappa_quadratic == pytest.approx(1.0)
    assert res.rank_agreement["spearman"] in (None, pytest.approx(1.0))
    assert res.self_preference["mean_gap"] == pytest.approx(0.0) and not res.self_preference["significant"]
    assert res.caveats  # the same-vendor caveat is always stated


def test_dual_judge_disagreement_is_flagged() -> None:
    rows = _rows()
    res = run_dual_judge(_client(second=lambda n: 10 - n % 11), rows, trials=2, log=lambda m: None)
    assert res.kappa is not None and res.kappa < 0.6 and res.flagged


def test_dual_judge_detects_same_family_bias() -> None:
    rows = _rows()
    res = run_dual_judge(_client(same=lambda n: min(10, n % 11 + 2)), rows, trials=2, log=lambda m: None)
    sp = res.self_preference
    assert sp["mean_gap"] > 0.5 and sp["significant"] and sp["ci_low"] > 0


def test_dual_judge_skips_cleanly() -> None:
    client = MockClient()
    anchored = evaluate_heldout(ToyTask(), client, _cascade(client), _members(), n=5, log=lambda m: None).rows
    res = run_dual_judge(client, anchored, log=lambda m: None)
    assert res.skipped and "automatic metric" in res.reason and res.kappa is None
    assert run_dual_judge(client, [], log=lambda m: None).skipped

    def boom(req: GenRequest) -> str:
        raise KeyError("second_family_judge")  # what LLMClient.model_for raises for an unconfigured role

    res = run_dual_judge(MockClient(script={"second_family_judge": boom}), _rows(), log=lambda m: None)
    assert res.skipped and "not configured" in res.reason


def test_dual_judge_missing_same_family_role_still_reports_agreement() -> None:
    def boom(req: GenRequest) -> str:
        raise KeyError("same_family_judge")

    c = _client()
    c.script["same_family_judge"] = boom  # type: ignore[index]
    res = run_dual_judge(c, _rows(), trials=2, log=lambda m: None)
    assert not res.skipped and res.kappa is not None and res.same_family_role is None and not res.self_preference
    assert any("not configured" in x for x in res.caveats)


def test_dual_judge_budget_stop_keeps_partial_results() -> None:
    rows = _rows()
    c = _client(budget=BudgetGuard(10.0, 20, 10**9))
    res = run_dual_judge(c, rows, trials=2, log=lambda m: None)
    assert any("budget" in x for x in res.caveats) and res.n < len(rows)


def test_dual_judge_write(tmp_path: Path) -> None:
    write_dual_judge(tmp_path, DualJudgeResult(skipped=True, reason="x"))
    assert json.loads((tmp_path / "dual_judge.json").read_text())["skipped"] is True


# ------------------------------------------------------------------ report
def _run_dir(tmp_path: Path) -> Path:
    from cadence.orchestrator import LoopSettings, optimize

    client = MockClient(script={"reflector": __import__("cadence.orchestrator.mock_reflector", fromlist=["x"]).toy_reflector})
    d = tmp_path / "run"
    optimize(ToyTask(), client, _cascade(client), LoopSettings(max_gen=2, minibatch=8, final_sample=16, workers=2), d,
             log=lambda m: None)
    return d


def test_report_has_every_section_on_a_real_run(tmp_path: Path) -> None:
    d = _run_dir(tmp_path)
    text = build_report(d)
    for h in HEADINGS:
        assert h in text, h
    assert "No judge calls were needed" not in text and "automatic metric" in text
    arch = json.loads((d / "archive.json").read_text())
    best = max(m["objectives"]["quality"] for m in arch)
    assert f"{best:.3f}" in text  # numbers come from the files
    assert write_report(d, with_plots=False).read_text(encoding="utf-8") == build_report(d)
    assert build_report(d) == build_report(d)  # deterministic


def test_report_survives_an_empty_directory(tmp_path: Path) -> None:
    text = build_report(tmp_path)
    for h in HEADINGS:
        assert h in text
    assert text.count("_not available") >= 4


def test_report_tolerates_corrupt_files(tmp_path: Path) -> None:
    (tmp_path / "archive.json").write_text("{not json")
    (tmp_path / "trace.jsonl").write_text('{"event": "stop"}\nnot json\n')
    (tmp_path / "heldout_summary.json").write_text("")
    for h in HEADINGS:
        assert h in build_report(tmp_path)


def test_report_shows_heldout_gap_and_flagged_dual_judge(tmp_path: Path) -> None:
    d = _run_dir(tmp_path)
    client = _client(second=lambda n: 10 - n % 11)
    held = evaluate_heldout(JudgedToy(), client, _cascade(client), _members(), n=22, workers=1, log=lambda m: None)
    write_heldout(d, held)
    write_dual_judge(d, run_dual_judge(client, held.rows, trials=2, log=lambda m: None))
    text = build_report(d)
    assert "gap (opt - held-out)" in text and "FLAGGED" in text and "not trustworthy" in text
    assert "same-family self-preference gap" in text


def test_report_plots_are_optional_and_never_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cadence.benchmark.report import make_plots

    d = _run_dir(tmp_path)
    broken = types.ModuleType("cadence.benchmark.plots")
    broken.load_members = lambda p: (_ for _ in ()).throw(RuntimeError("bad"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cadence.benchmark.plots", broken)
    assert make_plots(d) == {}  # a failing plot module must not lose the report
    fake = types.ModuleType("cadence.benchmark.plots")
    fake.load_members = lambda p: ["m"]  # type: ignore[attr-defined]
    fake.plot_pareto_3d = lambda m, out: out  # type: ignore[attr-defined]
    fake.plot_pareto_2d = lambda m, prefix: [Path(f"{prefix}_{s}.png") for s in ("a", "b", "c")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cadence.benchmark.plots", fake)
    got = make_plots(d)
    assert got["3D Pareto"] == "pareto_3d.png" and len(got) == 4
    assert "pareto_3d.png" in build_report(d, got)


# ------------------------------------------------------------------ calibration
SST = [{"idx": i, "sentence": f"sentence number {i}", "label": i % 2} for i in range(80)]


def test_calibration_unsupported_for_tasks_without_ground_truth() -> None:
    assert not supports_calibration(ToyTask())
    rep = run_calibration(MockClient(), ToyTask(), log=lambda m: None)
    assert rep["supported"] is False and "ground truth" in rep["reason"] and rep["flagged"] == []


def test_calibration_runs_on_a_task_with_ground_truth() -> None:
    task = Sst2Task(records=SST)
    assert supports_calibration(task)
    truth = {r["sentence"]: ("positive" if r["label"] else "negative") for r in SST}

    def gen(req: GenRequest) -> str:
        return truth[req.messages[-1]["content"]]

    def judge(req: GenRequest) -> str:
        text = req.messages[-1]["content"]
        inp = text.split("TASK INPUT:\n", 1)[1].split("\n\nRUBRIC:", 1)[0]
        good = truth[json.loads(inp)["sentence"]] in text.split("RESPONSE:\n", 1)[1].lower()
        return json.dumps({"score": 9 if good else 1, "confidence": 0.95, "rationale": "s"})

    client = MockClient(script={"generator": gen, "cheap_judge": judge, "strong_judge": judge,
                                "second_family_judge": judge})
    rep = run_calibration(client, task, n=8, trials=2, log=lambda m: None)
    assert rep["supported"] and rep["flagged"] == [] and rep["kappa_cascade_min"] == pytest.approx(1.0)
    assert rep["limits"] and rep["errors"] == []


# ------------------------------------------------------------------ CLI
@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import cadence.config as config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def test_cli_run_benchmark_report_flow(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cadence.cli import main

    assert main(["run", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "2", "--minibatch", "8",
                 "--run-id", "r1"]) == 0
    assert (root / "runs/r1/report.md").exists()  # run writes a report by default
    assert main(["benchmark", "--run", "r1", "--mock", "--n", "6", "--no-plots"]) == 0
    out = capsys.readouterr().out
    assert "held-out rows" in out and (root / "runs/r1/heldout.jsonl").exists()
    assert "Held-out benchmark" in (root / "runs/r1/report.md").read_text(encoding="utf-8")
    (root / "runs/r1/report.md").unlink()
    assert main(["report", "--run", "r1", "--no-plots"]) == 0 and (root / "runs/r1/report.md").exists()


def test_cli_failures_exit_one(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cadence.cli import main

    assert main(["benchmark", "--run", "nope", "--mock"]) == 1
    assert main(["report", "--run", "nope"]) == 1
    (root / "runs/empty").mkdir(parents=True)
    (root / "runs/empty/config.json").write_text(json.dumps({"task": "toy", "settings": {"seed": 0}}))
    (root / "runs/empty/archive.json").write_text("[]")
    assert main(["benchmark", "--run", "empty", "--mock"]) == 1
    assert "empty" in capsys.readouterr().out


def test_cli_calibrate_toy_reports_unsupported(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cadence.cli import main

    assert main(["calibrate", "--task", "toy", "--mock"]) == 0
    assert "not possible" in capsys.readouterr().out
    saved = json.loads((root / "runs/calibration-toy/calibration.json").read_text())
    assert saved["supported"] is False


def test_cli_no_report_flag(root: Path) -> None:
    from cadence.cli import main

    assert main(["run", "--task", "toy", "--mock", "--no-mlflow", "--max-gen", "1", "--minibatch", "4",
                 "--run-id", "r2", "--no-report"]) == 0
    assert not (root / "runs/r2/report.md").exists()
