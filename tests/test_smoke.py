import json
import re

import pytest

from cadence import cli
from cadence.llm import BudgetExceeded, BudgetGuard, MockClient
from cadence.schemas import GenRequest
from cadence.smoke import format_report, run_smoke
from cadence.tasks.text import Gsm8kTask, Sst2Task, metrics
from tests.judge_helpers import noisy_judge, user_text

SST = [{"idx": i, "sentence": f"sentence number {i}", "label": i % 2} for i in range(80)]
GSM = [{"question": f"What is {i}+1?", "answer": f"{i}+1\n#### {i + 1}"} for i in range(80)]
TRUTH = {r["sentence"]: ("positive" if r["label"] else "negative") for r in SST}
REF = {r["question"]: r["answer"].split("####")[-1].strip() for r in GSM}


def tasks():  # type: ignore[no-untyped-def]
    return {"sst2": Sst2Task(records=SST), "gsm8k": Gsm8kTask(records=GSM)}


def generator(req: GenRequest) -> str:
    q = user_text(req)
    if q in TRUTH:  # SST-2: wrong on every 4th sentence so genuine mistakes exist
        n = int(q.split()[-1])
        label = TRUTH[q]
        return label if n % 4 else ("negative" if label == "positive" else "positive")
    ref = REF[q]
    n = int(ref)
    return f"#### {ref}" if n % 4 else f"#### {n + 100}"


def perfect_judge(req: GenRequest) -> str:
    text = user_text(req)
    inp = json.loads(text.split("TASK INPUT:\n", 1)[1].split("\n\nRUBRIC:", 1)[0])
    resp = text.split("RESPONSE:\n", 1)[1]
    if "sentence" in inp:
        good = metrics.sentiment_label(resp) == TRUTH[inp["sentence"]]
    else:
        good = metrics.numeric_equal(metrics.extract_number(resp), REF[inp["question"]])
    return json.dumps({"score": 9 if good else 1, "confidence": 0.95, "rationale": "scripted"})


def make_client(**overrides):  # type: ignore[no-untyped-def]
    script = {"generator": generator, "cheap_judge": perfect_judge, "strong_judge": perfect_judge,
              "second_family_judge": perfect_judge, **overrides}
    return MockClient(script=script)


def test_smoke_with_a_perfect_scripted_stack() -> None:
    rep = run_smoke(make_client(), tasks(), n=8, trials=2, log=lambda m: None)
    assert rep["errors"] == []
    for name in ("sst2", "gsm8k"):
        t = rep["tasks"][name]
        assert 0.4 < t["accuracy"] < 1.0  # generator is deliberately wrong on every 4th example
        assert t["calibration_items"] > 4 and 0 < t["calibration_correct"] < t["calibration_items"]
        for role in ("cheap_judge", "strong_judge", "second_family_judge"):
            assert t["judges"][role]["kappa"] == pytest.approx(1.0) and not t["judges"][role]["flagged"]
        assert t["inter_judge_kappa"] == pytest.approx(1.0)
        assert t["cascade"]["kappa"] == pytest.approx(1.0)
        assert t["cascade"]["escalation_ratio"] == 0.0  # confident, consistent cheap judge
    text = format_report(rep)
    assert "Judges all clear" in text and "FLAGGED" not in text and "sst2" in text and "gsm8k" in text


def test_a_coin_flip_judge_is_flagged_in_the_report() -> None:
    rep = run_smoke(make_client(cheap_judge=noisy_judge(0.5)), tasks(), n=8, trials=2, log=lambda m: None)
    assert rep["tasks"]["sst2"]["judges"]["cheap_judge"]["flagged"]
    text = format_report(rep)
    assert "FLAGGED" in text and "FAIL: judges below kappa" in text and "sst2/cheap_judge" in text


def test_generation_errors_are_reported_not_fatal() -> None:
    calls = {"n": 0}

    def flaky(req: GenRequest) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom")
        return generator(req)

    rep = run_smoke(make_client(generator=flaky), {"sst2": Sst2Task(records=SST)}, n=8, trials=2, log=lambda m: None)
    assert any("generate" in e and "ValueError" in e for e in rep["errors"])
    assert rep["tasks"]["sst2"]["generated"] == 7
    assert "WARN" in format_report(rep)


def test_judge_errors_are_reported_per_role() -> None:
    def broken(req: GenRequest) -> str:
        raise RuntimeError("judge down")

    rep = run_smoke(make_client(second_family_judge=broken), {"sst2": Sst2Task(records=SST)}, n=8, trials=2,
                    log=lambda m: None)
    assert rep["tasks"]["sst2"]["judges"]["second_family_judge"] == {"error": "RuntimeError"}
    assert "inter_judge_kappa" not in rep["tasks"]["sst2"]
    assert "ERROR RuntimeError" in format_report(rep)


def test_budget_exhaustion_stops_the_run() -> None:
    client = MockClient(script={"generator": generator}, budget=BudgetGuard(1.0, 3, 100_000))
    with pytest.raises(BudgetExceeded):
        run_smoke(client, {"sst2": Sst2Task(records=SST)}, n=8, trials=2, log=lambda m: None)


def test_unusable_calibration_set_is_reported() -> None:
    class NoCorrupt(Sst2Task):
        def corrupt_output(self, ex, text):  # type: ignore[no-untyped-def]
            return None

    # generator always right -> every item correct -> no negatives to calibrate against
    always_right = lambda req: TRUTH[user_text(req)]  # noqa: E731
    rep = run_smoke(make_client(generator=always_right), {"sst2": NoCorrupt(records=SST)}, n=8, trials=2,
                    log=lambda m: None)
    assert any("calibration set unusable" in e for e in rep["errors"])


def test_cli_smoke_mock_runs_and_writes_a_report(monkeypatch, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("cadence.tasks.get_task", lambda name: Sst2Task(records=SST))
    out = tmp_path / "r.json"
    code = cli.main(["smoke", "--tasks", "sst2", "--n", "6", "--trials", "2", "--mock", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "SMOKE TEST" in printed and "MOCK" in printed
    assert code in (0, 1)
    assert json.loads(out.read_text())["n"] == 6


def test_cli_smoke_reports_missing_data(monkeypatch, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("cadence.tasks.get_task", lambda name: Gsm8kTask(path=tmp_path / "missing.jsonl"))
    assert cli.main(["smoke", "--tasks", "gsm8k", "--mock", "--out", str(tmp_path / "r.json")]) == 1
    assert "fetch_data.py gsm8k" in capsys.readouterr().out


def test_flagged_judge_report_explains_itself() -> None:
    rep = run_smoke(make_client(cheap_judge=noisy_judge(0.5)), {"sst2": Sst2Task(records=SST)}, n=8, trials=2,
                    log=lambda m: None)
    j = rep["tasks"]["sst2"]["judges"]["cheap_judge"]
    assert j["flagged"] and 1 <= len(j["misses"]) <= 5 and {"id", "judge_score", "rationale", "response"} <= set(j["misses"][0])
    assert "miss sst2-" in format_report(rep)


def test_unparseable_judge_replies_are_counted_and_shown() -> None:
    rep = run_smoke(make_client(cheap_judge=lambda req: "I think it is fine."), {"sst2": Sst2Task(records=SST)}, n=8,
                    trials=2, log=lambda m: None)
    j = rep["tasks"]["sst2"]["judges"]["cheap_judge"]
    assert j["parse_failures"] == 2 * rep["tasks"]["sst2"]["calibration_items"]
    assert "unparseable replies" in format_report(rep)


def test_flagged_judge_report_lists_misjudged_items_and_explains_consistent_errors() -> None:
    # a judge that is always wrong but identical across trials: kappa -1, ICC 1
    def inverted(req: GenRequest) -> str:
        d = json.loads(perfect_judge(req))
        d["score"] = 10 - d["score"]
        return json.dumps(d)

    rep = run_smoke(make_client(cheap_judge=inverted), {"sst2": Sst2Task(records=SST)}, n=8, trials=2,
                    log=lambda m: None)
    j = rep["tasks"]["sst2"]["judges"]["cheap_judge"]
    assert j["flagged"] and j["kappa"] == pytest.approx(-1.0) and j["icc"] == pytest.approx(1.0)
    text = format_report(rep)
    assert "miss sst2-" in text and "response ends:" in text
    assert "consistent across trials but wrong" in text
    healthy = format_report(run_smoke(make_client(), {"sst2": Sst2Task(records=SST)}, n=8, trials=2, log=lambda m: None))
    assert "miss sst2-" not in healthy and "consistent across trials" not in healthy
