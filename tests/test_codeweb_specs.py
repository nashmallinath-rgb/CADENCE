"""The code/web dataset: schema, brief rules, and satisfiability by the validator."""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from cadence.config import ROOT
from cadence.tasks.code_web.validators import validate_page

PATH = ROOT / "data" / "tasks" / "code_web.jsonl"
SPECS = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
CATEGORIES = {"form", "tool", "dashboard", "game", "content", "widget"}
SELECTOR = re.compile(r"^[a-z][a-z0-9]*(#[A-Za-z0-9_-]+)?(\.[A-Za-z0-9_-]+)?(\[[A-Za-z0-9_-]+(=[A-Za-z0-9_-]+)?\])?$|^[#.][A-Za-z0-9_-]+$")
PARSE = re.compile(r"^(?P<tag>[a-z][a-z0-9]*)?(?:#(?P<id>[\w-]+))?(?:\.(?P<cls>[\w-]+))?(?:\[(?P<attr>[\w-]+)(?:=(?P<val>[\w-]+))?\])?$")


def test_file_is_clean_jsonl() -> None:
    raw = PATH.read_text(encoding="utf-8")
    assert "\r" not in raw and raw.endswith("\n") and "\n\n" not in raw
    assert len(SPECS) == 30


def test_schema_and_counts() -> None:
    assert [s["id"] for s in SPECS] == [f"cw-{i:03d}" for i in range(1, 31)]
    assert Counter(s["category"] for s in SPECS) == {c: 5 for c in CATEGORIES}
    for s in SPECS:
        assert set(s) == {"id", "category", "brief", "requires", "forbid_external", "behaviours"}, s["id"]
        r = s["requires"]
        assert set(r) == {"selectors", "text", "scripts", "responsive"}
        assert 3 <= len(r["selectors"]) <= 8 and 1 <= len(r["text"]) <= 4 and 2 <= len(s["behaviours"]) <= 4
        assert isinstance(r["scripts"], bool) and isinstance(r["responsive"], bool) and s["forbid_external"] is True
        assert all(isinstance(x, str) and x for x in r["selectors"] + r["text"] + s["behaviours"])
        assert 30 <= len(s["brief"].split()) <= 90, (s["id"], len(s["brief"].split()))
    assert sum(s["requires"]["scripts"] for s in SPECS) >= 20
    assert sum(s["requires"]["responsive"] for s in SPECS) >= 10
    assert len({s["brief"] for s in SPECS}) == 30


def test_selectors_follow_the_grammar() -> None:
    for s in SPECS:
        for sel in s["requires"]["selectors"]:
            assert SELECTOR.match(sel), (s["id"], sel)


def test_brief_states_everything_the_validator_checks() -> None:
    """The model never sees `requires`, so the brief must name every id, class, attribute value and text."""
    for s in SPECS:
        brief = s["brief"]
        for sel in s["requires"]["selectors"]:
            m = PARSE.match(sel)
            assert m, sel
            for key in ("id", "cls", "val"):
                if m.group(key):
                    assert m.group(key) in brief, (s["id"], sel, m.group(key))
            if m.group("tag") and not (m.group("id") or m.group("cls") or m.group("attr")):
                assert f"`{m.group('tag')}`" in brief, (s["id"], sel)
        for t in s["requires"]["text"]:
            assert f'"{t}"' in brief, (s["id"], t)
        if s["requires"]["responsive"]:
            assert "360px" in brief, s["id"]
        if not s["requires"]["scripts"]:
            assert "no JavaScript" in brief, s["id"]


def test_briefs_never_mention_external_resources() -> None:
    for s in SPECS:
        low = s["brief"].lower()
        assert not any(w in low for w in ("http", "cdn", "fetch(", "import ", "library", "framework")), s["id"]


def test_ids_and_classes_do_not_clash_within_a_spec() -> None:
    for s in SPECS:
        ids = [PARSE.match(x).group("id") for x in s["requires"]["selectors"] if PARSE.match(x).group("id")]  # type: ignore[union-attr]
        assert len(ids) == len(set(ids)), s["id"]


# ---------------------------------------------------------------- satisfiability
def synthesize_page(spec: dict) -> str:
    """Smallest page that meets a spec, built from its selectors; proves specs and validator agree."""
    els = []
    for sel in spec["requires"]["selectors"]:
        m = PARSE.match(sel)
        assert m
        tag = m.group("tag") or "div"
        attrs = ""
        if m.group("id"):
            attrs += f' id="{m.group("id")}"'
        if m.group("cls"):
            attrs += f' class="{m.group("cls")}"'
        if m.group("attr"):
            attrs += f' {m.group("attr")}="{m.group("val")}"' if m.group("val") else f" {m.group('attr')}"
        els.append(f"<{tag}{attrs}></{tag}>")
    texts = "".join(f"<p>{t}</p>" for t in spec["requires"]["text"])
    script = "<script>var a = 1;</script>" if spec["requires"]["scripts"] else ""
    return ('<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>t</title></head><body>{''.join(els)}{texts}{script}</body></html>")


@pytest.mark.parametrize("spec", SPECS, ids=[s["id"] for s in SPECS])
def test_every_spec_is_satisfiable(spec: dict) -> None:
    res = validate_page(synthesize_page(spec), spec, use_node=False)
    assert res.score == 1.0, [c for c in res.checks if not c.passed]


def test_an_empty_page_fails_every_spec() -> None:
    for s in SPECS:
        assert validate_page("<!DOCTYPE html><html><body><p>hi</p></body></html>", s, use_node=False).score < 0.5, s["id"]
