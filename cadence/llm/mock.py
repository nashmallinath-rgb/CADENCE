"""Deterministic, scriptable, offline stand-in for LLMClient. Used by every test."""

from __future__ import annotations

import hashlib
import json
import struct
import wave
import zlib
import io
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cadence.cost.tracker import count_tokens
from cadence.llm.budget import BudgetGuard
from cadence.schemas import GenRequest, Usage

JUDGE_ROLES = frozenset({"cheap_judge", "strong_judge", "second_family_judge", "vision_judge", "vision_judge_strong",
                         "same_family_judge"})

ScriptEntry = str | Sequence[str] | Callable[[GenRequest], str]
Script = Callable[[GenRequest], str] | Mapping[str, ScriptEntry]


def _prompt_text(req: GenRequest) -> str:
    parts: list[str] = []
    for m in req.messages:
        c = m.get("content", "")
        if isinstance(c, list):  # multimodal: count text parts only (images are not tokens)
            parts.extend(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
        else:
            parts.append(str(c))
    return "\n".join(parts)


def tiny_png(seed: bytes, size: int = 8) -> bytes:
    """A valid size x size RGB PNG whose colour is derived from `seed` (stdlib only)."""
    r, g, b = seed.ljust(3, b"\0")[:3]
    row = b"\x00" + bytes((r, g, b)) * size
    raw = row * size

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


_TEXT_CHUNK = b"cdnc"


def mock_wav(text: str, seconds: float, rate: int = 4000) -> bytes:
    """8-bit silent WAV of `seconds`, with the spoken text hidden in an extra chunk so mock ASR can 'hear' it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(rate)
        w.writeframes(b"\x80" * max(1, int(seconds * rate)))
    body = text.encode("utf-8")
    return buf.getvalue() + _TEXT_CHUNK + struct.pack("<I", len(body)) + body


def mock_wav_text(data: bytes) -> str:
    i = data.rfind(_TEXT_CHUNK)
    if i < 0:
        return ""
    (n,) = struct.unpack("<I", data[i + 4:i + 8])
    return data[i + 8:i + 8 + n].decode("utf-8", errors="replace")


class MockClient:
    """Same interface as LLMClient: complete(req) -> (text, Usage).

    - Deterministic: the same request always yields the same text.
    - Role-aware: judge roles return JSON {"score": 0-10, "confidence": 0.5-0.99, "rationale": ...}.
    - Scriptable: script=callable(req)->str, or {role: str | [str, ...] (cycled) | callable}.
    - Free: usd is 0.0 unless you pass usd_per_input_token / usd_per_output_token.
    - Optional budget: pass a BudgetGuard to test budget-stopping loops offline.
    """

    def __init__(
        self,
        script: Script | None = None,
        *,
        budget: BudgetGuard | None = None,
        usd_per_input_token: float = 0.0,
        usd_per_output_token: float = 0.0,
        unit_prices: Mapping[str, float] | None = None,
        chars_per_second: float = 15.0,
        reads_directive_aloud: bool = False,
    ) -> None:
        self.script = script
        self.budget = budget
        self.usd_per_input_token = usd_per_input_token
        self.usd_per_output_token = usd_per_output_token
        self.unit_prices = dict(unit_prices or {})
        self.chars_per_second = chars_per_second
        self.reads_directive_aloud = reads_directive_aloud  # a TTS that speaks the 'directive: ' prefix too
        self.calls: list[GenRequest] = []
        self._cursor: dict[str, int] = {}

    # -- scripting -------------------------------------------------------------------
    def _scripted(self, req: GenRequest) -> str | None:
        if self.script is None:
            return None
        if callable(self.script):
            return self.script(req)
        entry = self.script.get(req.role)
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        if callable(entry):
            return entry(req)
        i = self._cursor.get(req.role, 0)
        self._cursor[req.role] = i + 1
        return entry[i % len(entry)]

    # -- defaults --------------------------------------------------------------------
    @staticmethod
    def _digest(req: GenRequest) -> bytes:
        blob = json.dumps(
            {"role": req.role, "messages": req.messages, "params": req.params}, sort_keys=True, default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).digest()

    def _default_text(self, req: GenRequest) -> str:
        d = self._digest(req)
        if req.role in JUDGE_ROLES:
            return json.dumps(
                {"score": d[0] % 11, "confidence": round(0.5 + (d[1] % 50) / 100, 2), "rationale": "mock"}
            )
        return f"mock:{req.role}:{d.hex()[:8]}"

    # -- public API ------------------------------------------------------------------
    def complete(self, req: GenRequest) -> tuple[str, Usage]:
        self.calls.append(req)
        text = self._scripted(req)
        if text is None:
            text = self._default_text(req)
        in_tok = count_tokens(_prompt_text(req))
        out_tok = count_tokens(text)
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd=in_tok * self.usd_per_input_token + out_tok * self.usd_per_output_token,
        )
        if self.budget is not None:
            self.budget.check(est_tokens=usage.total_tokens, est_usd=usage.usd)
            self.budget.record(usage)
        return text, usage

    # -- media (same signatures as LLMClient) ------------------------------------------------------------------------
    def _charge(self, usage: Usage) -> Usage:
        usage.usd = sum(n * self.unit_prices.get(k, 0.0) for k, n in usage.units.items())
        if self.budget is not None:
            self.budget.check(est_tokens=usage.total_tokens, est_usd=usage.usd)
            self.budget.record(usage)
        return usage

    def image(self, req: GenRequest) -> tuple[bytes, Usage]:
        self.calls.append(req)
        scripted = self._scripted(req)
        digest = self._digest(req)
        data = scripted.encode("latin-1") if isinstance(scripted, str) else tiny_png(digest)
        return data, self._charge(Usage(input_tokens=count_tokens(_prompt_text(req)), units={"images": 1.0}))

    def speech(self, req: GenRequest) -> tuple[bytes, Usage]:
        self.calls.append(req)
        text = req.messages[-1].get("content", "") if req.messages else ""
        spoken = str(text) if self.reads_directive_aloud else str(text).rsplit(": ", 1)[-1]
        seconds = max(0.5, len(spoken) / self.chars_per_second)
        return mock_wav(spoken, seconds), self._charge(
            Usage(input_tokens=count_tokens(spoken), units={"audio_seconds": seconds}))

    def transcribe(self, audio: bytes, req: GenRequest) -> tuple[str, Usage]:
        heard = mock_wav_text(audio)
        probe = GenRequest(role=req.role, messages=[{"role": "user", "content": heard}], params=req.params)
        self.calls.append(probe)
        scripted = self._scripted(probe)
        text = scripted if scripted is not None else heard
        try:
            with wave.open(io.BytesIO(audio), "rb") as w:
                seconds = w.getnframes() / float(w.getframerate() or 1)
        except (wave.Error, EOFError):
            seconds = 0.0
        return text, self._charge(Usage(units={"audio_seconds": seconds}))

    @property
    def call_count(self) -> int:
        return len(self.calls)
