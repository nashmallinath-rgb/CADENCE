import shutil

import pytest

from cadence.tasks.code_web import validators
from cadence.tasks.code_web.validators import extract_html, validate_page

SPEC = {"id": "cw-001",
        "requires": {"selectors": ["form#signup-form", "input#email", "input#password", "button#submit-btn", "#error"],
                     "text": ["Create account"], "scripts": True, "responsive": True},
        "forbid_external": True}

GOOD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign up</title><style>body{font-family:sans-serif}</style></head>
<body><form id="signup-form"><input id="email" type="email"><input id="password" type="password">
<button id="submit-btn" type="submit">Create account</button><p id="error"></p></form>
<script>document.getElementById('signup-form').addEventListener('submit', function (e) { e.preventDefault(); });</script>
</body></html>"""


def v(html: str, spec: dict = SPEC, node: bool = False):  # type: ignore[no-untyped-def]
    return validate_page(html, spec, use_node=node)


def failed(res) -> set[str]:  # type: ignore[no-untyped-def]
    return {c.name for c in res.checks if not c.passed}


# 1 ---------------------------------------------------------------- extract_html
def test_extract_html_variants() -> None:
    doc = "<!DOCTYPE html><html><body>x</body></html>"
    assert extract_html(f"Here you go:\n```html\n{doc}\n```\nEnjoy") == doc
    assert extract_html(f"```HTML\n{doc}\n```") == doc  # upper-case info string
    assert extract_html(f"```\n{doc}\n```") == doc  # generic fence containing html
    assert extract_html(f"Sure! {doc} Hope it helps.") == doc  # prose around a raw document
    assert extract_html("  just words, no markup  ") == "just words, no markup"
    assert extract_html("```python\nprint(1)\n```\n<html><body>b</body></html>").startswith("<html>")
    assert extract_html("x <!doctype html><html><body>cut off").endswith("cut off")  # no closing tag: to the end
    assert extract_html("") == ""


# 2, 3 ------------------------------------------------------------- scoring
def test_perfect_page_scores_one() -> None:
    res = v(GOOD)
    assert res.score == 1.0 and all(c.passed for c in res.checks) and not res.node_checked


def test_worked_example_missing_doctype() -> None:
    res = v(GOOD.replace("<!DOCTYPE html>\n", ""))
    assert failed(res) == {"doctype"}
    assert res.score == pytest.approx(10 / 10.5)
    assert sum(c.weight for c in res.checks) == pytest.approx(10.5)


# 4 ---------------------------------------------------------------- selectors
def test_missing_and_invalid_selectors() -> None:
    res = v(GOOD.replace('id="password"', 'id="pw"'))
    assert failed(res) == {"selector:input#password"}
    bad_spec = {**SPEC, "requires": {**SPEC["requires"], "selectors": ["form >"]}}
    res = v(GOOD, bad_spec)
    c = next(c for c in res.checks if c.name == "selector:form >")
    assert not c.passed and c.detail


# 5 ---------------------------------------------------------------- visible text
def test_required_text_must_be_visible() -> None:
    spec = {"requires": {"text": ["Secret"]}}
    for hidden in ("<script>var a='Secret';</script>", "<style>/* Secret */</style>"):
        assert failed(v(f"<!DOCTYPE html><html><body><p>x</p>{hidden}</body></html>", spec)) == {"text:Secret"}
    assert failed(v("<!DOCTYPE html><html><head><title>Secret</title></head><body><p>x</p></body></html>", spec)) == {"text:Secret"}
    for shown, needle in (('<input placeholder="Secret">', "Secret"), ('<input value="secret">', "Secret"),
                          ('<img alt="SECRET">', "Secret"), ("<p>  Se\n cret </p>", "Se cret")):
        res = v(f"<!DOCTYPE html><html><body>{shown}</body></html>", {"requires": {"text": [needle]}})
        assert f"text:{needle}" not in failed(res), shown


# 6 ---------------------------------------------------------------- external resources
@pytest.mark.parametrize("snippet", [
    '<script src="https://cdn.example.com/x.js"></script>',
    '<link rel="stylesheet" href="//fonts.example.com/f.css">',
    "<style>body{background:url(https://example.com/b.png)}</style>",
    '<div style="background:url(\'//example.com/b.png\')">x</div>',
    "<style>@import 'https://example.com/a.css';</style>",
    '<img src="http://example.com/i.png">',
])
def test_external_resources_fail(snippet: str) -> None:
    assert failed(v(GOOD.replace("</body>", snippet + "</body>"))) == {"no_external"}


def test_plain_links_and_local_urls_are_allowed() -> None:
    page = GOOD.replace("</body>", '<a href="https://example.com">docs</a><div style="background:url(data:image/png;base64,AAAA)">x</div></body>')
    assert v(page).score == 1.0
    assert "no_external" not in {c.name for c in v(GOOD, {**SPEC, "forbid_external": False}).checks}


# 7 ---------------------------------------------------------------- scripts
def test_scripts_requirement() -> None:
    no_script = GOOD[: GOOD.index("<script>")] + "</body></html>"
    assert failed(v(no_script)) == {"scripts"}
    empty = GOOD[: GOOD.index("<script>")] + "<script>  </script></body></html>"
    assert failed(v(empty)) == {"scripts"}
    external_only = GOOD[: GOOD.index("<script>")] + '<script src="app.js"></script></body></html>'
    assert failed(v(external_only)) == {"scripts"}  # src scripts do not count as inline code
    page = GOOD[: GOOD.index("<script>")] + '<script type="text/plain">not code</script></body></html>'
    assert failed(v(page)) == {"scripts"}
    off = {**SPEC, "requires": {**SPEC["requires"], "scripts": False}}
    assert "scripts" not in {c.name for c in v(no_script, off).checks}


# 8, 9, 10 --------------------------------------------------------------- parse, size, viewport
def test_truncated_page_fails_parses() -> None:
    assert "parses" in failed(v(GOOD[: GOOD.index("</form>")]))
    assert "parses" in failed(v("<html></html>"))  # no body content


def test_size_limit() -> None:
    assert "size" in failed(v(GOOD.replace("</body>", "<!--" + "x" * 60_000 + "--></body>")))
    assert "size" not in failed(v(GOOD))


def test_viewport_only_checked_for_responsive_specs() -> None:
    no_vp = GOOD.replace('<meta name="viewport" content="width=device-width, initial-scale=1">', "")
    assert failed(v(no_vp)) == {"viewport"}
    relaxed = {**SPEC, "requires": {**SPEC["requires"], "responsive": False}}
    assert v(no_vp, relaxed).score == 1.0


# 11, 12 ----------------------------------------------------------------- garbage, determinism
def test_garbage_never_raises_and_scores_low() -> None:
    assert v("").score < 0.25
    assert v("\x00\x01garbage").score < 0.35
    assert v("<" * 5000).score < 0.5
    assert v(None).score < 0.25  # type: ignore[arg-type]
    assert 0.0 <= v("<html><body><p>" + "a" * 10**6 + "</p></body></html>").score <= 1.0


def test_deterministic() -> None:
    assert v(GOOD).model_dump() == v(GOOD).model_dump()
    assert v(GOOD, node=True).model_dump() == v(GOOD, node=True).model_dump()


def test_empty_spec_tolerated() -> None:
    res = v(GOOD, {})
    assert {c.name for c in res.checks} == {"parses", "doctype", "size"} and res.score == 1.0


# 13, 14 ----------------------------------------------------------------- node
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_syntax_with_node() -> None:
    ok = v(GOOD, node=True)
    assert ok.node_checked and ok.score == 1.0 and "js_syntax" in {c.name for c in ok.checks}
    bad = v(GOOD.replace("e.preventDefault();", "var x = ;"), node=True)
    assert bad.node_checked and failed(bad) == {"js_syntax"} and bad.score < 1.0
    assert any(c.detail for c in bad.checks if c.name == "js_syntax")
    mod = GOOD.replace("<script>", '<script type="module">import {x} from "./a.js"; export const y = 1;')
    assert v(mod, node=True).node_checked  # module syntax parses as .mjs


def test_node_missing_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validators.shutil, "which", lambda name: None)
    res = v(GOOD, node=True)
    assert not res.node_checked and "js_syntax" not in {c.name for c in res.checks} and res.score == 1.0
    assert not v(GOOD, node=False).node_checked


def test_node_timeout_is_a_failed_check_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr(validators.shutil, "which", lambda name: "node")

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired("node", 10)

    monkeypatch.setattr(validators.subprocess, "run", boom)
    res = v(GOOD, node=True)
    c = next(c for c in res.checks if c.name == "js_syntax")
    assert not c.passed and "Timeout" in c.detail and res.node_checked
