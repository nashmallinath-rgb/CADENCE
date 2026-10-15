"""Pairwise judge, used only for position-bias checks."""
from __future__ import annotations

import json
import re

from cadence.judge.judge import SupportsComplete, render_input
from cadence.schemas import Example, GenRequest, Usage

SYSTEM = (
    "You compare two responses to the same task using the rubric. "
    'Reply with ONLY JSON: {"winner": "A" | "B" | "tie"}.'
)


def parse_winner(text: str) -> str | None:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            w = str(json.loads(m.group(0)).get("winner", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError):
            w = ""
        if w in ("A", "B"):
            return w
        if w == "TIE":
            return "tie"
    return None


class PairwiseJudge:
    def __init__(self, client: SupportsComplete, role: str) -> None:
        self.client, self.role = client, role
        self.usage = Usage()

    def compare(self, ex: Example, a: str, b: str, rubric: str) -> str:
        """Returns 'A', 'B' or 'tie' (unparseable counts as 'tie')."""
        user = f"TASK INPUT:\n{render_input(ex)}\n\nRUBRIC:\n{rubric}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}"
        req = GenRequest(
            role=self.role,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            params={"temperature": 0.0},
        )
        text, u = self.client.complete(req)
        self.usage = self.usage + u
        return parse_winner(text) or "tie"
