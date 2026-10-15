"""Token counting and the deployment-cost formula from docs/SPEC.md section 2."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache

from cadence.schemas import Usage

_ENCODING_NAME = "cl100k_base"
_FALLBACK_RE = re.compile(r"\w+|[^\w\s]")


@lru_cache(maxsize=1)
def _encoder():  # type: ignore[no-untyped-def]
    try:
        import tiktoken

        return tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:  # offline machine, missing vocab download, etc.
        return None


def tokenizer_backend() -> str:
    """Which tokenizer is active. 'fallback' only happens when tiktoken cannot load its vocab."""
    return f"tiktoken:{_ENCODING_NAME}" if _encoder() is not None else "fallback:regex"


def count_tokens(text: str) -> int:
    """Token count via tiktoken cl100k_base (a documented proxy for Gemini/Grok tokenizers)."""
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return len(_FALLBACK_RE.findall(text))


def template_length(template: str) -> int:
    """The `length` objective: token count of the prompt template."""
    return count_tokens(template)


def cost_per_1k_from_tokens(
    avg_input_tokens: float, avg_output_tokens: float, price_in_per_token: float, price_out_per_token: float
) -> float:
    """Literal SPEC formula: 1000 * (avg_in * P_in + avg_out * P_out)."""
    return 1000.0 * (avg_input_tokens * price_in_per_token + avg_output_tokens * price_out_per_token)


def cost_per_1k(usages: Sequence[Usage]) -> float:
    """USD per 1,000 executions, from priced usage records (Usage.usd includes per-unit costs)."""
    if not usages:
        raise ValueError("cost_per_1k needs at least one Usage")
    return 1000.0 * sum(u.usd for u in usages) / len(usages)
