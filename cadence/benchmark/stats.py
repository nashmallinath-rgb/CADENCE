"""Small statistics helpers for the benchmark (seeded, no hidden randomness)."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from scipy.stats import kendalltau, spearmanr


def mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
                 stat: Callable[[Sequence[float]], float] = mean) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat`. A single value (or none) gives a degenerate interval."""
    vals = list(values)
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo, hi = stats[int((alpha / 2) * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def rank_agreement(a: Sequence[float], b: Sequence[float]) -> dict[str, float | None]:
    """Spearman rho and Kendall tau between two score lists over the same items; None when undefined (<3 items, constant)."""
    if len(a) != len(b):
        raise ValueError("lists must be the same length")
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return {"spearman": None, "kendall": None}
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    return {"spearman": float(rho), "kendall": float(tau)}
