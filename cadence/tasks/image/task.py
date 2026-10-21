"""Image stream: a prompt-prefix `template` plus a visual concept -> one generated image, scored by a vision judge."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cadence.config import ROOT
from cadence.schemas import Example, GenRequest, Output
from cadence.tasks.base import Split, Task
from cadence.tasks.splits import select

DATA = ROOT / "data" / "tasks" / "image_prompts.jsonl"
DEFAULT_MEDIA = ROOT / "runs" / "_media"


def template_dir(template: str) -> str:
    return hashlib.sha256(" ".join(template.split()).encode()).hexdigest()[:10]


class ImageTask(Task):
    name = "image"
    stream = "image"

    def __init__(self, records: Sequence[dict[str, Any]] | None = None, path: Path | None = None,
                 media_dir: Path | None = None) -> None:
        self._records = list(records) if records is not None else None
        self._path = path or DATA
        self.media_dir = Path(media_dir) if media_dir else DEFAULT_MEDIA
        self._examples: list[Example] | None = None

    def _all(self) -> list[Example]:
        if self._examples is None:
            if self._records is None:
                if not self._path.exists():
                    raise FileNotFoundError(f"{self._path} not found (data/tasks/image_prompts.jsonl ships with the repo)")
                self._records = [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._examples = [Example(id=r["id"], input={"concept": r["concept"]}, reference=None,
                                      meta={"category": r.get("category"), "must_include": list(r.get("must_include", []))})
                              for r in self._records]
            ids = [e.id for e in self._examples]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate example ids in image prompts")
        return self._examples

    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        return select(self._all(), split, n, seed)

    def seed_prompt(self) -> str:
        return "Create a clear, well-composed, detailed image of the following concept."

    def format_input(self, ex: Example) -> str:
        return str(ex.input["concept"])

    def build_request(self, template: str, ex: Example) -> GenRequest:
        return GenRequest(role="image_gen", messages=[{"role": "user", "content": f"{template}\n\n{ex.input['concept']}"}])

    def generate(self, client: Any, req: GenRequest) -> Output:
        data, usage = client.image(req)
        template = req.messages[-1]["content"].split("\n\n", 1)[0]
        concept_key = hashlib.sha256(req.messages[-1]["content"].encode()).hexdigest()[:8]
        folder = self.media_dir / template_dir(template)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{concept_key}.png"
        path.write_bytes(data)
        return Output(kind="image", path=str(path), usage=usage)

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return None  # always needs the vision judge

    def rubric(self, ex: Example) -> str | None:
        must = ", ".join(ex.meta.get("must_include", [])) or "the described subject"
        return (
            "The response is an image generated from the concept in TASK INPUT. Score 0-10 on three things: "
            f"(1) prompt alignment: are these elements clearly visible: {must}; (2) aesthetic quality: composition, "
            "lighting, colour, clarity; (3) absence of artifacts: no garbled text, broken anatomy or visual noise. "
            "Weigh alignment most. A blank, unrelated or unreadable image scores 0-1."
        )

    # judge calibration hooks (reference-free: there is no ground truth for aesthetics, so only gross failures)
    def degraded_prompt(self) -> str | None:
        return "Ignore the concept. Produce an almost blank, featureless image."
