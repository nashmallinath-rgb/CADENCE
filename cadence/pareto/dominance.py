"""Pareto dominance utilities. All vectors are minimisation vectors."""
from __future__ import annotations

from collections.abc import Sequence

from cadence.schemas import Objectives

Vector = tuple[float, ...]


def to_min_vector(o: Objectives) -> Vector:
    """(quality is maximised, so negate it) -> (-quality, cost, length)."""
    return (-o.quality, o.cost_usd_per_1k, float(o.length_tokens))


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a dominates b: no worse in every objective and strictly better in one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_indices(vs: Sequence[Sequence[float]]) -> list[int]:
    return [i for i, v in enumerate(vs) if not any(dominates(w, v) for j, w in enumerate(vs) if j != i)]


def non_dominated_sort(vs: Sequence[Sequence[float]]) -> list[list[int]]:
    """Fronts of indices; front 0 is the Pareto set."""
    remaining = list(range(len(vs)))
    fronts: list[list[int]] = []
    while remaining:
        sub = [vs[i] for i in remaining]
        keep = set(non_dominated_indices(sub))
        fronts.append([remaining[k] for k in sorted(keep)])
        remaining = [r for k, r in enumerate(remaining) if k not in keep]
    return fronts


def crowding_distance(vs: Sequence[Sequence[float]]) -> list[float]:
    """NSGA-II crowding distance; boundary points get infinity."""
    n = len(vs)
    if n == 0:
        return []
    dist = [0.0] * n
    for m in range(len(vs[0])):
        order = sorted(range(n), key=lambda i: vs[i][m])
        lo, hi = vs[order[0]][m], vs[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = float("inf")
        if hi == lo:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (vs[order[k + 1]][m] - vs[order[k - 1]][m]) / (hi - lo)
    return dist
