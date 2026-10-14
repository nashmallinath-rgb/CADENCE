"""LLM client stubs. All real calls must go through this package later."""

from __future__ import annotations

from typing import Any


class MockClient:
    """Deterministic stand-in so tests never hit the network."""

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Return a canned reply. Not a real completion."""
        return "mock-ok"


def cached_complete(*args: Any, **kwargs: Any) -> None:
    """Placeholder for disk-cached LiteLLM completion."""
    raise NotImplementedError("cadence.llm cache/complete is a stub")
