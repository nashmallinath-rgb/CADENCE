"""Exact hypervolume (minimisation) for 1-3 objectives by slicing."""
from __future__ import annotations

from collections.abc import Sequence


def _hv2(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    area, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            nxt_x = ref[0]
            area += (nxt_x - x) * (best_y - y)
            best_y = y
    return area


def hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    pts = [tuple(p) for p in points if all(a < r for a, r in zip(p, ref))]
    dim = len(ref)
    if not pts:
        return 0.0
    if dim == 1:
        return ref[0] - min(p[0] for p in pts)
    if dim == 2:
        return _hv2(pts, (ref[0], ref[1]))  # type: ignore[arg-type]
    # slice along the last axis
    zs = sorted({p[-1] for p in pts})
    total = 0.0
    for k, z in enumerate(zs):
        upper = zs[k + 1] if k + 1 < len(zs) else ref[-1]
        active = [p[:-1] for p in pts if p[-1] <= z]
        total += hypervolume(active, ref[:-1]) * (upper - z)
    return total
