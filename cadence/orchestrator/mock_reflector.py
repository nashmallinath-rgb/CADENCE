"""Offline reflector for ToyTask demos and the e2e test: edits templates by adding/removing the toy phrases."""
from __future__ import annotations

import hashlib
import re

from cadence.schemas import GenRequest
from cadence.tasks.toy import PHRASES

_PARENT = re.compile(r"<<<\n(.*?)\n>>>", re.S)
_OP = re.compile(r"OPERATOR: (\w+)")
_PRIORITY = ["step by step", "double-check", "show your work", "cite the context", "be concise"]


def _pick(template: str, options: list[str], salt: str) -> str:
    h = int(hashlib.sha256((template + salt).encode()).hexdigest()[:8], 16)
    return options[h % len(options)]


def toy_reflector(req: GenRequest) -> str:
    """Script for MockClient(script={'reflector': toy_reflector}). Deterministic in the request."""
    user = req.messages[-1]["content"]
    op = (_OP.search(user) or [None, "reflect"])[1]
    parents = _PARENT.findall(user)
    base = parents[0] if parents else "Answer the question."
    if op == "crossover" and len(parents) > 1:
        phrases = [p for p in PHRASES if p in (parents[0] + " " + parents[1]).lower()]
        return "<template>Answer the question. " + " ".join(p.capitalize() + "." for p in phrases) + "</template>"
    present = [p for p in PHRASES if p in base.lower()]
    if op == "shorten":
        if present:
            drop = _pick(base, present, "drop")
            kept = [p for p in present if p != drop]
            return "<template>Answer the question. " + " ".join(p.capitalize() + "." for p in kept) + "</template>"
        return "<template>Answer.</template>"
    missing = [p for p in _PRIORITY if p not in present]
    if not missing:
        return "<template>" + base + "</template>"
    add = _pick(base, missing, op)
    return "<template>" + base.rstrip() + " " + add.capitalize() + ".</template>"
