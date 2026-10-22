from pathlib import Path

import pytest

from cadence.cascade import Cascade
from cadence.judge import Judge
from cadence.llm import MockClient
from cadence.schemas import Example, Output
from cadence.tasks import TASKS, assign_split, get_task, select
from cadence.tasks.text import Gsm8kTask, HotpotQATask, IfbenchTask, Sst2Task, metrics
from cadence.tasks.text.tasks import read_jsonl

FIX = Path(__file__).parent / "fixtures"


def task_from_fixture(cls):  # type: ignore[no-untyped-def]
    return cls(records=read_jsonl(FIX / f"{cls.name}.jsonl"))


def all_examples(task):  # type: ignore[no-untyped-def]
    return [e for s in ("opt", "heldout", "calib") for e in task.load(s)]


@pytest.mark.parametrize("cls", [Gsm8kTask, HotpotQATask, Sst2Task, IfbenchTask])
def test_every_fixture_row_lands_in_exactly_one_split(cls) -> None:  # type: ignore[no-untyped-def]
    task = task_from_fixture(cls)
    n_rows = len(read_jsonl(FIX / f"{cls.name}.jsonl"))
    parts = {s: {e.id for e in task.load(s)} for s in ("opt", "heldout", "calib")}
    assert sum(len(p) for p in parts.values()) == n_rows
    assert parts["opt"].isdisjoint(parts["heldout"]) and parts["opt"].isdisjoint(parts["calib"])
    assert parts["heldout"].isdisjoint(parts["calib"])


def test_splits_are_deterministic_seed_dependent_and_respect_n() -> None:
    exs = [Example(id=f"x{i}", input={}) for i in range(1000)]
    a = [e.id for e in select(exs, "opt", seed=0)]
    assert a == [e.id for e in select(exs, "opt", seed=0)]
    assert a != [e.id for e in select(exs, "opt", seed=1)]
    assert len(select(exs, "opt", n=10, seed=0)) == 10
    sizes = {s: len(select(exs, s)) for s in ("opt", "heldout", "calib")}
    assert 540 < sizes["opt"] < 660 and 200 < sizes["heldout"] < 300 and 100 < sizes["calib"] < 200
    assert assign_split("x1", 0) in sizes
    with pytest.raises(ValueError):
        select(exs, "train")


def test_gsm8k_loader_and_reference() -> None:
    task = task_from_fixture(Gsm8kTask)
    refs = {e.id: e.reference for e in all_examples(task)}
    assert refs["gsm8k-1"] == "2400"  # '#### 2,400' -> commas stripped
    assert refs["gsm8k-4"] == "-6" and refs["gsm8k-3"] == "4.5"


@pytest.mark.parametrize("text,expected", [
    ("so 3+4=7\n#### 7", "7"),
    ("The answer is 2,400.", "2400"),
    ("Final: \\boxed{15}", "15"),
    ("it costs $15 in total", "15"),
    ("first 3 then 8 and finally 12 apples", "12"),
    ("**Answer: -6**", "-6"),
    ("no digits here", None),
])
def test_extract_number(text: str, expected: str | None) -> None:
    assert metrics.extract_number(text) == expected


def test_numeric_equal_edge_cases() -> None:
    assert metrics.numeric_equal("7", "7.0") and metrics.numeric_equal("4.50", "4.5")
    assert not metrics.numeric_equal("7", "8") and not metrics.numeric_equal(None, "7")


def test_gsm8k_scoring_end_to_end() -> None:
    task = task_from_fixture(Gsm8kTask)
    ex = next(e for e in all_examples(task) if e.id == "gsm8k-1")
    assert task.automatic_score(ex, Output(kind="text", text="2 boxes... #### 2400")) == 1.0
    assert task.automatic_score(ex, Output(kind="text", text="#### 2000")) == 0.0
    assert task.automatic_score(ex, Output(kind="text", text="")) == 0.0


def test_hotpot_loader_formats_context_and_scores_em() -> None:
    task = task_from_fixture(HotpotQATask)
    ex = next(e for e in all_examples(task) if e.id == "h1")
    assert "[T1a] Paris is in France. It is big." in ex.input["context"]
    assert "Question: Which country is Paris in?" in task.format_input(ex)
    assert task.automatic_score(ex, Output(kind="text", text="Answer: France.\nBecause...")) == 1.0
    assert task.automatic_score(ex, Output(kind="text", text="Germany")) == 0.0


def test_normalize_answer_and_exact_match() -> None:
    assert metrics.exact_match("The Beatles!", "beatles")
    assert metrics.exact_match("  A  cat ", "cat")
    assert not metrics.exact_match("cats", "cat")
    assert metrics.first_answer_line("\n\n**Answer:** Paris\nmore") == "Paris"
    assert metrics.first_answer_line("   ") == ""
    assert metrics.first_answer_line("Answer: **Paris**") == "Paris"
    assert metrics.first_answer_line("answer - Paris") == "Paris"
    assert metrics.first_answer_line("Paris") == "Paris"


def test_sst2_loader_labels_and_scoring() -> None:
    task = task_from_fixture(Sst2Task)
    exs = {e.id: e for e in all_examples(task)}
    assert exs["sst2-0"].reference == "positive" and exs["sst2-1"].reference == "negative"
    assert task.automatic_score(exs["sst2-0"], Output(kind="text", text="Positive")) == 1.0
    assert task.automatic_score(exs["sst2-0"], Output(kind="text", text="negative")) == 0.0
    assert task.automatic_score(exs["sst2-0"], Output(kind="text", text="unsure")) == 0.0
    assert metrics.sentiment_label("negative, definitely not positive") == "negative"
    with pytest.raises(ValueError):
        Sst2Task(records=[{"idx": 1, "sentence": "s", "label": -1}]).load("opt")


def test_ifbench_is_judged_and_rubric_never_leaks_reference() -> None:
    task = task_from_fixture(IfbenchTask)
    ex = all_examples(task)[0]
    assert task.automatic_score(ex, Output(kind="text", text="anything")) is None
    assert "constraint" in (task.rubric(ex) or "")
    ex2 = ex.model_copy(update={"reference": "SECRET-GOLD"})
    assert "SECRET-GOLD" not in (task.rubric(ex2) or "")


def test_build_request_puts_template_in_system_and_example_in_user() -> None:
    task = task_from_fixture(Sst2Task)
    ex = all_examples(task)[0]
    req = task.build_request("MY TEMPLATE", ex)
    assert req.role == "generator"
    assert req.messages[0] == {"role": "system", "content": "MY TEMPLATE"}
    assert req.messages[1]["role"] == "user" and ex.input["sentence"] in req.messages[1]["content"]
    assert ex.reference not in json_blob(req)


def json_blob(req) -> str:  # type: ignore[no-untyped-def]
    return str(req.messages)


def test_default_generate_uses_the_client_and_keeps_usage() -> None:
    task = task_from_fixture(Sst2Task)
    ex = all_examples(task)[0]
    client = MockClient(script={"generator": "positive"}, usd_per_input_token=0.001)
    out = task.generate(client, task.build_request(task.seed_prompt(), ex))
    assert out.text == "positive" and out.usage.usd > 0 and client.call_count == 1


def test_cascade_works_with_real_tasks() -> None:
    gsm = task_from_fixture(Gsm8kTask)
    ifb = task_from_fixture(IfbenchTask)
    client = MockClient(script={"cheap_judge": '{"score": 8, "confidence": 0.9}'})
    cas = Cascade(Judge(client, "cheap_judge", 3), Judge(client, "strong_judge", 3))
    ex_g = all_examples(gsm)[0]
    s = cas.score(gsm, ex_g, Output(kind="text", text="#### 1"))
    assert not s.judged and client.call_count == 0  # auto-scored: no judge
    s = cas.score(ifb, all_examples(ifb)[0], Output(kind="text", text="a reply"))
    assert s.judged and s.quality == pytest.approx(0.8)


def test_missing_data_file_gives_a_helpful_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError, match="fetch_data.py gsm8k"):
        Gsm8kTask(path=tmp_path / "nope.jsonl").load("opt")


def test_registry() -> None:
    # superset: other streams (image, audio, code_web) register themselves too
    assert set(TASKS) >= {"toy", "gsm8k", "hotpotqa", "sst2", "ifbench"}
    assert get_task("toy").name == "toy"
    with pytest.raises(KeyError, match="available"):
        get_task("nope")


def test_sst2_pooled_splits_with_colliding_idx_still_load() -> None:
    # HF train and validation both number rows from 0; the fetch script pools them
    rows = [{"idx": i % 40, "sentence": f"s{i}", "label": i % 2} for i in range(80)]
    task = Sst2Task(records=rows)
    ids = [e.id for s in ("opt", "heldout", "calib") for e in task.load(s)]
    assert len(ids) == 80 and len(set(ids)) == 80


def test_duplicate_ids_error_names_the_offenders() -> None:
    rows = [{"id": "dup", "question": "q", "answer": "a", "context": {"title": [], "sentences": []}}] * 2
    with pytest.raises(ValueError, match="dup"):
        HotpotQATask(records=rows).load("opt")
