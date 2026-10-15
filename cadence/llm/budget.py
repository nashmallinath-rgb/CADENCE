"""Hard spend caps for a run. Checked before every live call, recorded after it."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from cadence.schemas import Usage

if TYPE_CHECKING:
    from cadence.config import BudgetConfig


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a call that would push the run past a cap."""

    def __init__(self, kind: str, limit: float, would_be: float) -> None:
        super().__init__(f"budget exceeded: {kind} cap {limit:g}, would reach {would_be:g}")
        self.kind = kind
        self.limit = limit
        self.would_be = would_be


class BudgetGuard:
    """Tracks calls, tokens and USD for one run. Thread-safe. Cache hits are never recorded."""

    def __init__(self, max_usd: float, max_calls: int, max_tokens: int) -> None:
        self.max_usd = max_usd
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self._usd = 0.0
        self._calls = 0
        self._tokens = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: BudgetConfig) -> BudgetGuard:
        return cls(cfg.max_usd_per_run, cfg.max_calls_per_run, cfg.max_tokens_per_run)

    @property
    def spent_usd(self) -> float:
        return self._usd

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def tokens(self) -> int:
        return self._tokens

    def check(self, est_tokens: int = 0, est_usd: float = 0.0) -> None:
        """Raise BudgetExceeded if one more call (with these estimates) would break a cap.

        Estimates are best effort. Pass max_tokens on calls to keep the USD cap strict.
        """
        with self._lock:
            if self._calls + 1 > self.max_calls:
                raise BudgetExceeded("calls", self.max_calls, self._calls + 1)
            if self._tokens >= self.max_tokens or self._tokens + est_tokens > self.max_tokens:
                raise BudgetExceeded("tokens", self.max_tokens, self._tokens + max(est_tokens, 1))
            if self._usd >= self.max_usd or self._usd + est_usd > self.max_usd:
                raise BudgetExceeded("usd", self.max_usd, self._usd + est_usd)

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._calls += 1
            self._tokens += usage.total_tokens
            self._usd += usage.usd

    def ok(self) -> bool:
        """True while at least one more call could still start."""
        with self._lock:
            return self._calls < self.max_calls and self._tokens < self.max_tokens and self._usd < self.max_usd

    def remaining(self) -> dict[str, float]:
        with self._lock:
            return {
                "usd": max(self.max_usd - self._usd, 0.0),
                "calls": max(self.max_calls - self._calls, 0),
                "tokens": max(self.max_tokens - self._tokens, 0),
            }
