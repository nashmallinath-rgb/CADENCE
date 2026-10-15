from __future__ import annotations

import json

from cadence.pareto.dominance import crowding_distance, dominates, to_min_vector
from cadence.pareto.hypervolume import hypervolume as _hv
from cadence.schemas import Evaluated, Objectives


class ParetoArchive:
    """Bounded set of mutually non-dominated Evaluated prompts."""

    def __init__(self, max_size: int = 20) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._items: list[Evaluated] = []

    @property
    def members(self) -> list[Evaluated]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(list(self._items))

    def add(self, ev: Evaluated) -> bool:
        """Insert; returns True if the candidate is in the archive afterwards."""
        # same candidate id: replace the old evaluation
        self._items = [m for m in self._items if m.candidate.id != ev.candidate.id]
        v = to_min_vector(ev.objectives)
        vecs = [to_min_vector(m.objectives) for m in self._items]
        if any(w == v or dominates(w, v) for w in vecs):
            return False
        self._items = [m for m, w in zip(self._items, vecs) if not dominates(v, w)]
        self._items.append(ev)
        self._prune()
        return any(m is ev for m in self._items)

    def _prune(self) -> None:
        while len(self._items) > self.max_size:
            d = crowding_distance([to_min_vector(m.objectives) for m in self._items])
            # smallest distance goes; ties drop the most recently added
            victim = min(range(len(d)), key=lambda i: (d[i], -i))
            self._items.pop(victim)

    def hypervolume(self, ref: Objectives) -> float:
        return _hv([to_min_vector(m.objectives) for m in self._items], to_min_vector(ref))

    def to_json(self) -> str:
        return json.dumps([m.model_dump(mode="json") for m in self._items])

    @classmethod
    def from_json(cls, s: str, max_size: int = 20) -> "ParetoArchive":
        a = cls(max_size)
        for d in json.loads(s):
            a.add(Evaluated.model_validate(d))
        return a
