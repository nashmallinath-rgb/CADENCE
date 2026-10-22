"""Code / web UI stream: the template is the system prompt, a brief is the user turn, the output is one HTML page.

Quality is the weighted structural checklist from `validators.validate_page` (automatic, so no judge calls).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cadence.config import ROOT
from cadence.schemas import Example, GenRequest, Output
from cadence.tasks.base import Split, Task
from cadence.tasks.code_web.validators import extract_html, validate_page
from cadence.tasks.splits import select

DATA = ROOT / "data" / "tasks" / "code_web.jsonl"
MAX_TOKENS = 4096


class CodeWebTask(Task):
    name = "code_web"
    stream = "code_web"

    def __init__(self, records: Sequence[dict[str, Any]] | None = None, path: Path | None = None) -> None:
        self._records = list(records) if records is not None else None
        self._path = path or DATA
        self._examples: list[Example] | None = None

    def _all(self) -> list[Example]:
        if self._examples is None:
            if self._records is None:
                if not self._path.exists():
                    raise FileNotFoundError(f"{self._path} not found (data/tasks/code_web.jsonl ships with the repo)")
                self._records = [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._examples = [Example(id=r["id"], input={"brief": r["brief"]}, reference=None,
                                      meta={"spec": r, "category": r.get("category")}) for r in self._records]
            ids = [e.id for e in self._examples]
            if len(set(ids)) != len(ids):
                dups = sorted({i for i in ids if ids.count(i) > 1})[:5]
                raise ValueError(f"duplicate example ids in code_web specs, e.g. {dups}")
        return self._examples

    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        return select(self._all(), split, n, seed)

    def format_input(self, ex: Example) -> str:
        return str(ex.input["brief"])

    def seed_prompt(self) -> str:
        return ("You are a front-end engineer. Implement the brief as ONE self-contained HTML file with inline CSS and "
                "JavaScript and no external resources. Reply with only the complete HTML document, starting with "
                "<!DOCTYPE html>.")

    def build_request(self, template: str, ex: Example) -> GenRequest:
        req = super().build_request(template, ex)
        req.params = {**req.params, "max_tokens": MAX_TOKENS}
        return req

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return validate_page(extract_html(out.text or ""), ex.meta["spec"]).score

    def rubric(self, ex: Example) -> str | None:
        return None  # structural checklist only in v1; a design-quality judge is future work
