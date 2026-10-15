"""Offline ToyTask: no LLM, deterministic. Used by the e2e loop test and `--mock` demos.

Quality rises with helpful instruction phrases in the template; output length, cost and template length also rise
with them, so a genuine quality / cost / length trade-off front exists.
"""
from __future__ import annotations

import hashlib
from typing import Any

from cadence.cost import count_tokens
from cadence.schemas import Example, GenRequest, Output, Usage
from cadence.tasks.base import Split, Task
from cadence.tasks.splits import select

# phrase -> (quality boost, extra output tokens it provokes)
PHRASES: dict[str, tuple[float, int]] = {
    "step by step": (0.20, 120),
    "double-check": (0.15, 60),
    "show your work": (0.10, 100),
    "be concise": (0.05, -30),
    "cite the context": (0.10, 40),
}
BASE_P, BASE_OUT = 0.25, 40
USD_IN, USD_OUT = 1e-6, 4e-6  # toy prices per token


def keywords_present(template: str) -> list[str]:
    t = template.lower()
    return [p for p in PHRASES if p in t]


def p_correct(template: str) -> float:
    return min(0.95, BASE_P + sum(PHRASES[p][0] for p in keywords_present(template)))


def _u(*parts: str) -> float:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16) / 16**12


class ToyTask(Task):
    name = "toy"
    stream = "text"

    def __init__(self, size: int = 240) -> None:
        self._examples = [Example(id=f"toy-{i}", input={"question": f"Toy question {i}"}, reference=f"a{i}") for i in range(size)]

    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        return select(self._examples, split, n, seed)

    def seed_prompt(self) -> str:
        return "Answer the question."

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return 1.0 if (out.text or "").startswith("CORRECT") else 0.0

    def generate(self, client: Any, req: GenRequest) -> Output:  # client unused: fully offline
        template = req.messages[0]["content"]
        ex_text = req.messages[-1]["content"]
        correct = _u(template, ex_text) < p_correct(template)
        out_tokens = max(5, BASE_OUT + sum(PHRASES[p][1] for p in keywords_present(template)))
        in_tokens = count_tokens(template) + count_tokens(ex_text)
        usage = Usage(input_tokens=in_tokens, output_tokens=out_tokens, usd=in_tokens * USD_IN + out_tokens * USD_OUT)
        return Output(kind="text", text=("CORRECT" if correct else "WRONG") + " " + "x " * (out_tokens // 2), usage=usage)
