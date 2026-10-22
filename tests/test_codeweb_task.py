import json
from pathlib import Path

import pytest

from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.llm import MockClient
from cadence.orchestrator import LoopSettings, optimize
from cadence.schemas import Output
from cadence.tasks import get_task
from cadence.tasks.code_web import CodeWebTask

PAGE = ('<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width"><title>t</title></head>'
        '<body><h1>Hi</h1><button id="go">Go</button><script>var a = 1;</script></body></html>')


def rec(i: int) -> dict:
    return {"id": f"cw-{i:03d}", "category": "tool", "brief": f"Make page {i} with a button with id `go` labelled \"Go\".",
            "requires": {"selectors": ["h1", "button#go"], "text": ["Go"], "scripts": True, "responsive": True},
            "forbid_external": True, "behaviours": ["a", "b"]}


RECORDS = [rec(i) for i in range(1, 41)]


def test_load_splits_are_deterministic_and_disjoint() -> None:
    t = CodeWebTask(records=RECORDS)
    parts = {s: t.load(s) for s in ("opt", "heldout", "calib")}
    ids = [e.id for p in parts.values() for e in p]
    assert sorted(ids) == sorted(r["id"] for r in RECORDS) and len(set(ids)) == 40
    assert [e.id for e in t.load("opt", seed=3)] == [e.id for e in CodeWebTask(records=RECORDS).load("opt", seed=3)]
    assert len(t.load("opt", n=5)) == 5


def test_example_shape_never_leaks_a_reference() -> None:
    ex = CodeWebTask(records=RECORDS).load("opt", n=1)[0]
    assert ex.reference is None and "brief" in ex.input and ex.meta["spec"]["requires"]["scripts"] is True
    assert CodeWebTask(records=RECORDS).format_input(ex) == ex.input["brief"]


def test_build_request_template_is_system_brief_is_user() -> None:
    t = CodeWebTask(records=RECORDS)
    ex = t.load("opt", n=1)[0]
    req = t.build_request("TEMPLATE", ex)
    assert req.role == "generator" and req.messages[0] == {"role": "system", "content": "TEMPLATE"}
    assert req.messages[1]["role"] == "user" and req.messages[1]["content"] == ex.input["brief"]
    assert req.params["max_tokens"] == 4096
    assert "<!DOCTYPE html>" in t.seed_prompt()


def test_automatic_score_uses_the_validator_and_extracts_fences() -> None:
    t = CodeWebTask(records=RECORDS)
    ex = t.load("opt", n=1)[0]
    assert t.automatic_score(ex, Output(kind="text", text=PAGE)) == 1.0
    assert t.automatic_score(ex, Output(kind="text", text=f"Sure!\n```html\n{PAGE}\n```")) == 1.0
    assert t.automatic_score(ex, Output(kind="text", text=PAGE.replace('id="go"', ""))) < 1.0
    assert t.automatic_score(ex, Output(kind="text", text="I cannot do that.")) < 0.4
    assert t.automatic_score(ex, Output(kind="text", text=None)) < 0.4
    assert t.rubric(ex) is None  # structural only: the cascade never calls a judge


def test_missing_file_and_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CodeWebTask(path=tmp_path / "nope.jsonl").load("opt")
    with pytest.raises(ValueError):
        CodeWebTask(records=[rec(1), rec(1)]).load("opt")
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in RECORDS[:5]) + "\n\n", encoding="utf-8")
    assert len(CodeWebTask(path=p)._all()) == 5  # blank lines are skipped


def test_registry_and_shipped_data() -> None:
    t = get_task("code_web")
    assert isinstance(t, CodeWebTask) and t.stream == "code_web"
    assert len(t.load("opt")) + len(t.load("heldout")) + len(t.load("calib")) == 30


@pytest.mark.e2e
def test_loop_runs_on_code_web_with_a_scripted_generator(tmp_path: Path) -> None:
    def gen(req):  # type: ignore[no-untyped-def]
        # a longer system prompt yields a "better" page, so the loop sees a quality/length trade-off
        good = len(req.messages[0]["content"]) > 120
        return PAGE if good else PAGE.replace('<button id="go">Go</button>', "")

    def reflect(req):  # type: ignore[no-untyped-def]
        return "<template>Produce one HTML file implementing the brief. Include every element and every id named in the brief, inline CSS and JS only, no external resources, nothing outside the document.</template>"

    client = MockClient(script={"generator": gen, "reflector": reflect})
    task = CodeWebTask(records=RECORDS)
    res = optimize(task, client, Cascade.from_config(client, CascadeConfig()),
                   LoopSettings(max_gen=2, children_per_gen=2, minibatch=4, final_sample=8, workers=2),
                   tmp_path / "r", log=lambda m: None)
    assert res.stopped_reason == "max_gen" and res.final_reevaluated
    best = max(m.objectives.quality for m in res.archive)
    assert best == 1.0 and res.hv_history[-1] >= res.hv_history[0]
    assert all(c.role != "cheap_judge" for c in client.calls)  # automatic score: no judge calls
