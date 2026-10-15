"""Reference-free rubric judge with multiple trials."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from cadence.schemas import Example, GenRequest, Usage

SYSTEM = (
    "You are a strict, impartial evaluator. Rate the RESPONSE to the TASK INPUT using the RUBRIC. "
    "You are not given a reference answer; judge on the merits. "
    'Reply with ONLY a JSON object: {"score": <0-10 number>, "confidence": <0-1 number>, "rationale": "<one sentence>"}.'
)


class SupportsComplete(Protocol):
    def complete(self, req: GenRequest) -> tuple[str, Usage]: ...


class JudgeTrial(BaseModel):
    score: float  # 0-10
    confidence: float  # 0-1
    rationale: str = ""


class JudgeResult(BaseModel):
    score_0_10: float
    confidence_0_1: float
    spread: float = 0.0  # max - min of trial scores
    trials: list[JudgeTrial] = Field(default_factory=list)
    parse_failures: int = 0
    usage: Usage = Field(default_factory=Usage)


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_trial(text: str) -> JudgeTrial | None:
    """Lenient parser: JSON object anywhere in the text, else 'score: x' style fallbacks."""
    data: dict[str, Any] | None = None
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            cand = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "score" in cand:
            data = cand
            break
    if data is None:
        s = re.search(r"score\W{0,4}" + _NUM, text, flags=re.I)
        if not s:
            return None
        c = re.search(r"confidence\W{0,4}" + _NUM, text, flags=re.I)
        data = {"score": s.group(1), "confidence": c.group(1) if c else 0.5}
    try:
        score = float(data["score"])
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    if score != score or conf != conf:  # NaN
        return None
    return JudgeTrial(
        score=min(10.0, max(0.0, score)),
        confidence=min(1.0, max(0.0, conf)),
        rationale=str(data.get("rationale", ""))[:300],
    )


def render_input(ex: Example) -> str:
    return json.dumps(ex.input, sort_keys=True, ensure_ascii=False, default=str)


class Judge:
    """Scores one response with `trials` independent passes (different pass marker => distinct cache keys)."""

    def __init__(self, client: SupportsComplete, role: str, trials: int = 3, temperature: float = 0.7) -> None:
        if trials < 1:
            raise ValueError("trials must be >= 1")
        self.client, self.role, self.trials, self.temperature = client, role, trials, temperature

    def with_trials(self, trials: int) -> "Judge":
        return Judge(self.client, self.role, trials, self.temperature)

    def build_request(self, ex: Example, response_text: str, rubric: str, i: int) -> GenRequest:
        user = (
            f"[evaluation pass {i + 1}/{self.trials}]\n"
            f"TASK INPUT:\n{render_input(ex)}\n\nRUBRIC:\n{rubric}\n\nRESPONSE:\n{response_text}"
        )
        return GenRequest(
            role=self.role,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            params={"temperature": self.temperature},
        )

    def score(self, ex: Example, response_text: str, rubric: str) -> JudgeResult:
        """Reference-free: `ex.reference` is never put in the prompt."""
        trials: list[JudgeTrial] = []
        failures = 0
        usage = Usage()
        for i in range(self.trials):
            text, u = self.client.complete(self.build_request(ex, response_text, rubric, i))
            usage = usage + u
            t = parse_trial(text)
            if t is None:
                failures += 1
            else:
                trials.append(t)
        if not trials:  # nothing parseable: zero confidence forces escalation downstream
            return JudgeResult(score_0_10=0.0, confidence_0_1=0.0, parse_failures=failures, usage=usage)
        scores = [t.score for t in trials]
        return JudgeResult(
            score_0_10=math.fsum(scores) / len(scores),
            confidence_0_1=math.fsum(t.confidence for t in trials) / len(trials),
            spread=max(scores) - min(scores),
            trials=trials,
            parse_failures=failures,
            usage=usage,
        )
