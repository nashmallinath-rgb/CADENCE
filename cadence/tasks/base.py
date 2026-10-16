"""Task contract (docs/ARCHITECTURE.md section 2): what the optimizer and cascade need from a task."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Literal

from cadence.schemas import Example, GenRequest, Output

Split = Literal["opt", "heldout", "calib"]
Stream = Literal["text", "code_web", "image", "audio"]


class Task(ABC):
    """One evaluation task. A candidate prompt `template` is the system instruction; the example is the user turn."""

    name: str
    stream: Stream

    @abstractmethod
    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        """Examples for one split. Splits are disjoint and deterministic given `seed`."""

    @abstractmethod
    def seed_prompt(self) -> str:
        """Starting template for the optimizer."""

    @abstractmethod
    def automatic_score(self, ex: Example, out: Output) -> float | None:
        """Score in [0, 1] from a verifiable metric (exact match etc.); None means 'needs a judge'."""

    def rubric(self, ex: Example) -> str | None:
        """Judge rubric for tasks that cannot be scored automatically. Must never leak `ex.reference`."""
        return None

    def calibration_rubric(self, ex: Example) -> str | None:
        """Reference-free rubric used to check a judge against this task's exact-match ground truth (None = unsupported)."""
        return None

    def corrupt_output(self, ex: Example, text: str) -> str | None:
        """A deliberately altered response, used to guarantee wrong answers in judge calibration sets (None = unsupported)."""
        return None

    def format_input(self, ex: Example) -> str:
        return json.dumps(ex.input, ensure_ascii=False, sort_keys=True) if len(ex.input) != 1 else str(next(iter(ex.input.values())))

    def build_request(self, template: str, ex: Example) -> GenRequest:
        return GenRequest(
            role="generator",
            messages=[{"role": "system", "content": template}, {"role": "user", "content": self.format_input(ex)}],
        )

    def generate(self, client: Any, req: GenRequest) -> Output:
        """Default: text chat through the LLM client. Image/audio/toy tasks override."""
        text, usage = client.complete(req)
        return Output(kind="text", text=text, usage=usage)
