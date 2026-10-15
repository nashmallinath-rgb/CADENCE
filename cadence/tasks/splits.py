"""Deterministic, disjoint opt / heldout / calib splits by hashing example ids."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cadence.schemas import Example

# fractions of the pool: optimisation, held-out benchmarking, judge calibration
FRACTIONS: tuple[tuple[str, float], ...] = (("opt", 0.60), ("heldout", 0.25), ("calib", 0.15))


def _h(seed: int, key: str, salt: str = "") -> float:
    d = hashlib.sha256(f"{salt}{seed}:{key}".encode()).hexdigest()
    return int(d[:12], 16) / 16**12  # uniform in [0, 1)


def assign_split(example_id: str, seed: int = 0) -> str:
    u, acc = _h(seed, example_id), 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return FRACTIONS[-1][0]


def select(examples: Sequence[Example], split: str, n: int | None = None, seed: int = 0) -> list[Example]:
    """Examples of `split`, in a seeded pseudo-random order, truncated to `n`."""
    if split not in {s for s, _ in FRACTIONS}:
        raise ValueError(f"unknown split {split!r}")
    mine = [e for e in examples if assign_split(e.id, seed) == split]
    mine.sort(key=lambda e: _h(seed, e.id, "order"))
    return mine if n is None else mine[:n]
