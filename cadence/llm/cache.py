"""Disk cache for LLM calls. A cache hit costs nothing and never touches the budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


def make_cache_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any], seed: int | None) -> str:
    """Stable hash of everything that can change a response."""
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NullCache:
    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        return None


class MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value


class DiskCache:
    """Persistent cache backed by diskcache (SQLite). Safe to delete the directory."""

    def __init__(self, directory: Path | str) -> None:
        import diskcache

        self._cache = diskcache.Cache(str(directory))

    def get(self, key: str) -> dict[str, Any] | None:
        return self._cache.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._cache.set(key, value)

    def close(self) -> None:
        self._cache.close()
