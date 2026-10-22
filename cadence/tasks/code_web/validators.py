"""Structural validation of model-written HTML pages: a weighted checklist, deterministic, never raises.

Checks (weights): parses 1, doctype 0.5, one per required selector 1, one per required text 1, scripts 1,
js_syntax 2 (only when `node` exists), no_external 1, size 0.5, viewport 0.5 (responsive specs only).
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from bs4 import BeautifulSoup
from pydantic import BaseModel

MAX_CHARS = 60_000
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}
_EXTERNAL_ATTRS = (("script", "src"), ("link", "href"), ("img", "src"), ("iframe", "src"), ("source", "src"),
                   ("video", "src"), ("audio", "src"))
_VISIBLE_ATTRS = ("placeholder", "value", "alt", "aria-label", "title")
_CSS_EXTERNAL = re.compile(r"url\(\s*['\"]?\s*(?:https?:)?//|@import\s+(?:url\(\s*)?['\"]?\s*(?:https?:)?//", re.I)
_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)


class Check(BaseModel):
    name: str
    weight: float
    passed: bool
    detail: str = ""


class ValidationResult(BaseModel):
    score: float
    checks: list[Check]
    node_checked: bool = False


def extract_html(text: str) -> str:
    """Pull the HTML document out of a model reply (markdown fence, prose around it, or raw)."""
    fences = _FENCE.findall(text)
    for info, body in fences:
        if info.strip().lower() == "html":
            return body.strip()
    for _info, body in fences:
        low = body.lower()
        if "<html" in low or "<!doctype" in low:
            return body.strip()
    low = text.lower()
    starts = [i for i in (low.find("<!doctype"), low.find("<html")) if i != -1]
    if starts:
        start = min(starts)
        end = low.rfind("</html>")
        return text[start: end + len("</html>")].strip() if end != -1 else text[start:].strip()
    return text.strip()


def _raw(el: Any) -> str:
    """Contents of <script>/<style>: bs4's get_text() deliberately skips them."""
    return "".join(str(c) for c in el.contents)


def _collapse(s: str) -> str:
    return " ".join(s.split()).lower()


def _visible_text(page: BeautifulSoup) -> str:
    soup = copy.copy(page)  # decomposing tags must not change the page the other checks look at
    for tag in soup.find_all(["script", "style", "template", "noscript", "head"]):
        tag.decompose()
    parts = [soup.get_text(" ")]
    for el in soup.find_all(True):
        for attr in _VISIBLE_ATTRS:
            v = el.get(attr)
            if isinstance(v, str):
                parts.append(v)
    return _collapse(" ".join(parts))


def _is_external(url: str) -> bool:
    u = url.strip().lower()
    return u.startswith(("http://", "https://", "//"))


def _external_resources(soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    for tag, attr in _EXTERNAL_ATTRS:
        for el in soup.find_all(tag):
            v = el.get(attr)
            if isinstance(v, str) and _is_external(v):
                found.append(f"{tag}[{attr}]={v[:60]}")
    for st in soup.find_all("style"):
        if _CSS_EXTERNAL.search(_raw(st)):
            found.append("<style> loads an external URL")
    for el in soup.find_all(style=True):
        if isinstance(el.get("style"), str) and _CSS_EXTERNAL.search(el["style"]):
            found.append("style attribute loads an external URL")
    return found


def _inline_js(soup: BeautifulSoup) -> list[tuple[str, bool]]:
    """(source, is_module) of every inline JavaScript script with a non-empty body."""
    out: list[tuple[str, bool]] = []
    for s in soup.find_all("script"):
        if s.get("src"):
            continue
        t = (s.get("type") or "").strip().lower()
        body = _raw(s)
        if t in _JS_TYPES and body.strip():
            out.append((body, t == "module"))
    return out


def _node_check(scripts: list[tuple[str, bool]], node: str) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory() as d:
            for i, (src, module) in enumerate(scripts):
                path = os.path.join(d, f"s{i}.{'mjs' if module else 'js'}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(src)
                p = subprocess.run([node, "--check", path], capture_output=True, timeout=10)
                if p.returncode != 0:
                    err = p.stderr.decode("utf-8", "replace").strip().splitlines()
                    return False, f"script {i + 1}: " + (next((ln for ln in err if "Error" in ln), err[0] if err else "syntax error"))[:160]
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"[:160]
    return True, ""


def validate_page(html: str, spec: dict[str, Any], *, use_node: bool = True) -> ValidationResult:
    req = spec.get("requires") or {}
    selectors = list(req.get("selectors") or [])
    texts = list(req.get("texts") or req.get("text") or [])
    wants_scripts, responsive = bool(req.get("scripts", False)), bool(req.get("responsive", False))
    forbid_external = bool(spec.get("forbid_external", False))
    html = html if isinstance(html, str) else ""
    checks: list[Check] = []

    def add(name: str, weight: float, passed: bool, detail: str = "") -> None:
        checks.append(Check(name=name, weight=weight, passed=bool(passed), detail=detail))

    low = html.lower()
    try:
        soup = BeautifulSoup(html, "html5lib")
        parse_error = ""
    except Exception as exc:  # noqa: BLE001 - never raise on bad input
        soup, parse_error = BeautifulSoup("", "html5lib"), f"{type(exc).__name__}"
    body = soup.body
    has_content = bool(body is not None and (body.get_text(strip=True) or body.find(True)))
    closed = "<html" in low and "</html>" in low
    add("parses", 1.0, closed and has_content and not parse_error,
        parse_error or ("" if closed else "missing <html> or </html> (truncated output?)") or ("" if has_content else "empty body"))
    add("doctype", 0.5, html.lstrip().lower().startswith("<!doctype html"))
    for sel in selectors:
        try:
            ok, detail = bool(soup.select(sel)), ""
        except Exception as exc:  # noqa: BLE001 - invalid selector text
            ok, detail = False, f"invalid selector: {type(exc).__name__}"
        add(f"selector:{sel}", 1.0, ok, detail)
    js = _inline_js(soup)  # before _visible_text, which strips scripts
    external = _external_resources(soup) if forbid_external else []
    visible = _visible_text(soup)
    for t in texts:
        add(f"text:{t}", 1.0, _collapse(t) in visible)
    if wants_scripts:
        add("scripts", 1.0, bool(js), "" if js else "no inline <script> with code")
    node = shutil.which("node") if use_node else None
    node_checked = False
    if js and node:
        ok, detail = _node_check(js, node)
        node_checked = True
        add("js_syntax", 2.0, ok, detail)
    if forbid_external:
        add("no_external", 1.0, not external, "; ".join(external[:3]))
    add("size", 0.5, len(html) <= MAX_CHARS, f"{len(html)} characters")
    if responsive:
        has_vp = any((m.get("name") or "").lower() == "viewport" for m in soup.find_all("meta"))
        add("viewport", 0.5, has_vp)
    total = sum(c.weight for c in checks)
    score = sum(c.weight for c in checks if c.passed) / total if total else 0.0
    return ValidationResult(score=score, checks=checks, node_checked=node_checked)
