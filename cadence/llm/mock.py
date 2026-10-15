"""Deterministic, scriptable, offline stand-in for LLMClient. Used by every test."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cadence.cost.tracker import count_tokens
from cadence.llm.budget import BudgetGuard
from cadence.schemas import GenRequest, Usage

JUDGE_ROLES = frozenset({"cheap_judge", "strong_judge", "second_family_judge", "vision_judge"})

ScriptEntry = str | Sequence[str] | Callable[[GenRequest], str]
Script = Callable[[GenRequest], str] | Mapping[str, ScriptEntry]


def _prompt_text(req: GenRequest) -> str:
    return "\n".join(str(m.get("content", "")) for m in req.messages)


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
    ) -> None:
        self.script = script
        self.budget = budget
        self.usd_per_input_token = usd_per_input_token
        self.usd_per_output_token = usd_per_output_token
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

    @property
    def call_count(self) -> int:
        return len(self.calls)
