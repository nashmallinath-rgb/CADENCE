"""All model calls go through this package: cached, throttled, retried, budget-checked."""

from cadence.llm.budget import BudgetExceeded, BudgetGuard
from cadence.llm.cache import DiskCache, MemoryCache, NullCache, make_cache_key
from cadence.llm.client import (
    BackendResponse,
    LLMClient,
    TransientLLMError,
    cached_complete,
    get_client,
    reset_default_client,
)
from cadence.llm.mock import MockClient

__all__ = [
    "BackendResponse",
    "BudgetExceeded",
    "BudgetGuard",
    "DiskCache",
    "LLMClient",
    "MemoryCache",
    "MockClient",
    "NullCache",
    "TransientLLMError",
    "cached_complete",
    "get_client",
    "make_cache_key",
    "reset_default_client",
]
