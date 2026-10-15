"""Scripted judges for tests: quality is encoded in the response text ('GOOD' / 'BAD')."""
from __future__ import annotations

import hashlib
import json
import random

from cadence.schemas import Example, GenRequest, Output


def user_text(req: GenRequest) -> str:
    return req.messages[-1]["content"]


def noisy_judge(noise: float, conf: float = 0.9, seed: int = 0):  # type: ignore[no-untyped-def]
    """Judge that rates GOOD ~9 / BAD ~1 but flips its verdict with probability `noise`, independently per pass."""

    def script(req: GenRequest) -> str:
        text = user_text(req)
        rng = random.Random(int(hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()[:12], 16))
        good = "GOOD" in text.split("RESPONSE:", 1)[-1]
        if rng.random() < noise:
            good = not good
        return json.dumps({"score": 9 if good else 1, "confidence": conf, "rationale": "scripted"})

    return script


def ex(i: int, reference: object = None) -> Example:
    return Example(id=f"e{i}", input={"q": f"question {i}"}, reference=reference)


class FakeTask:
    """Auto-scores examples whose id starts with 'auto'; everything else needs a judge."""

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return 1.0 if ex.id.startswith("auto") else None

    def rubric(self, ex: Example) -> str | None:
        return "Is the response correct and clear?"
