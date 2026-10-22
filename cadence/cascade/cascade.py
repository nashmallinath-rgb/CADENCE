"""Cost-aware two-tier evaluation cascade."""
from __future__ import annotations

import threading
from typing import Any, Protocol

from pydantic import BaseModel, Field

from cadence.config import CascadeConfig
from cadence.judge import Judge, JudgeResult
from cadence.schemas import Example, Output, Score, Usage


EPS = 1e-9


class Scorable(Protocol):
    """The slice of the Task contract the cascade needs."""

    def automatic_score(self, ex: Example, out: Output) -> float | None: ...
    def rubric(self, ex: Example) -> str | None: ...


def media_of(out: Output) -> tuple[bytes, str] | None:
    """(bytes, mime) of an image output so a vision judge can see it; None for text/audio outputs."""
    if out.kind != "image" or not out.path:
        return None
    from pathlib import Path

    from cadence.llm.media import image_mime

    data = Path(out.path).read_bytes()
    return data, image_mime(data)


class CascadeStats(BaseModel):
    n_scored: int = 0
    n_auto: int = 0
    n_judged: int = 0
    n_escalated: int = 0
    judge_usage: Usage = Field(default_factory=Usage)

    @property
    def escalation_ratio(self) -> float:
        """Escalated / judged (auto-scored items never need a judge, so they are excluded)."""
        return self.n_escalated / self.n_judged if self.n_judged else 0.0


class Cascade:
    def __init__(self, cheap: Judge, strong: Judge, tau: float = 0.70, delta: float = 2.0) -> None:
        self.cheap, self.strong, self.tau, self.delta = cheap, strong, tau, delta
        self.stats = CascadeStats()
        self._lock = threading.Lock()  # the evaluator scores examples from several threads

    @classmethod
    def from_config(cls, client: Any, cfg: CascadeConfig, cheap_role: str = "cheap_judge",
                    strong_role: str = "strong_judge") -> "Cascade":
        return cls(Judge(client, cheap_role, cfg.cheap_trials), Judge(client, strong_role, cfg.strong_trials),
                   cfg.tau, cfg.delta)

    def needs_escalation(self, r: JudgeResult) -> bool:
        # EPS keeps boundary cases (confidence == tau, spread == delta) from flipping on float noise
        return r.confidence_0_1 < self.tau - EPS or r.spread > self.delta + EPS

    def score(self, task: Scorable, ex: Example, out: Output) -> Score:
        with self._lock:
            self.stats.n_scored += 1
        auto = task.automatic_score(ex, out)
        if auto is not None:  # exact-match style tasks never touch a judge
            with self._lock:
                self.stats.n_auto += 1
            comps_auto = {"auto": float(auto)}
            extra = getattr(task, "auto_components", None)  # optional hook: e.g. WER and speaking rate for audio
            if callable(extra):
                comps_auto.update({k: float(v) for k, v in extra(ex, out).items()})
            return Score(quality=min(1.0, max(0.0, float(auto))), components=comps_auto)
        rubric = task.rubric(ex)
        if rubric is None:
            raise ValueError("task returned neither an automatic score nor a rubric for example " + ex.id)
        media = media_of(out)
        text = out.text or ("(the response is the attached image)" if media else "")
        kw = {"media": media} if media is not None else {}  # text judges keep the 3-argument call
        with self._lock:
            self.stats.n_judged += 1
        cheap = self.cheap.score(ex, text, rubric, **kw)
        usage = cheap.usage
        comps = {"cheap": cheap.score_0_10 / 10, "cheap_confidence": cheap.confidence_0_1, "cheap_spread": cheap.spread}
        escalated = self.needs_escalation(cheap)
        final = cheap.score_0_10
        if escalated:
            with self._lock:
                self.stats.n_escalated += 1
            strong = self.strong.score(ex, text, rubric, **kw)
            usage = usage + strong.usage
            final = strong.score_0_10  # strong verdict replaces the cheap one
            comps["strong"] = strong.score_0_10 / 10
            comps["strong_confidence"] = strong.confidence_0_1
        with self._lock:
            self.stats.judge_usage = self.stats.judge_usage + usage
        return Score(quality=final / 10, components=comps, judged=True, escalated=escalated, usage=usage)
