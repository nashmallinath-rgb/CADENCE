"""Audio stream: a delivery-directive `template` + spoken text -> TTS, then ASR round trip scored by WER."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cadence.config import ROOT
from cadence.schemas import Example, GenRequest, Output
from cadence.tasks.audio import metrics
from cadence.tasks.base import Split, Task
from cadence.tasks.splits import select

DATA = ROOT / "data" / "tasks" / "audio_prompts.jsonl"
DEFAULT_MEDIA = ROOT / "runs" / "_media"


class AudioTask(Task):
    name = "audio"
    stream = "audio"

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
                    raise FileNotFoundError(f"{self._path} not found (data/tasks/audio_prompts.jsonl ships with the repo)")
                self._records = [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
            # the text to be spoken is the task input AND the WER reference (it is what a perfect TTS+ASR returns)
            self._examples = [Example(id=r["id"], input={"text": r["text"]}, reference=r["text"]) for r in self._records]
            ids = [e.id for e in self._examples]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate example ids in audio prompts")
        return self._examples

    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        return select(self._all(), split, n, seed)

    def seed_prompt(self) -> str:
        return "Speak clearly at a natural, steady pace"

    def format_input(self, ex: Example) -> str:
        return str(ex.input["text"])

    def build_request(self, template: str, ex: Example) -> GenRequest:
        return GenRequest(role="tts", messages=[{"role": "user", "content": f"{template}: {ex.input['text']}"}])

    def generate(self, client: Any, req: GenRequest) -> Output:
        wav, tts_usage = client.speech(req)  # deployment cost = this stage only
        key = hashlib.sha256(req.messages[-1]["content"].encode()).hexdigest()
        folder = self.media_dir / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key[:16]}.wav"
        path.write_bytes(wav)
        transcript, _asr_usage = client.transcribe(wav, GenRequest(role="asr", messages=[]))  # charged to the budget, not to cost
        return Output(kind="audio", text=transcript, path=str(path), usage=tts_usage)

    def _parts(self, ex: Example, out: Output) -> tuple[float, float, float]:
        w = metrics.wer(str(ex.input["text"]), out.text or "")
        cps = metrics.chars_per_second(str(ex.input["text"]), float(out.usage.units.get("audio_seconds", 0.0)))
        return w, cps, metrics.rate_score(cps)

    def weak_seed_prompt(self) -> str | None:
        return "Say it as fast as you possibly can"

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        w, _cps, rate = self._parts(ex, out)
        return metrics.audio_quality(w, rate)

    def auto_components(self, ex: Example, out: Output) -> dict[str, float]:
        w, cps, rate = self._parts(ex, out)
        return {"wer": w, "chars_per_second": cps, "rate_score": rate}
