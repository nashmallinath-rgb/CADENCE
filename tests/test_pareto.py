import itertools
import random

import pytest

from cadence.pareto import (
    ParetoArchive,
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_indices,
    non_dominated_sort,
)
from cadence.schemas import Candidate, Evaluated, Objectives


def ev(cid: str, q: float, c: float, l: int) -> Evaluated:
    return Evaluated(
        candidate=Candidate(id=cid, template=cid),
        objectives=Objectives(quality=q, cost_usd_per_1k=c, length_tokens=l),
        n_examples=1,
    )


def test_dominance_basics() -> None:
    assert dominates((1, 1, 1), (2, 2, 2))
    assert dominates((1, 1, 1), (1, 1, 2))
    assert not dominates((1, 1, 1), (1, 1, 1))  # equal vectors don't dominate
    assert not dominates((1, 2, 1), (2, 1, 1))


def test_non_dominated_sort_hand_fronts() -> None:
    vs = [(1, 4, 1), (4, 1, 1), (2, 2, 1), (3, 3, 1), (5, 5, 5)]
    assert non_dominated_indices(vs) == [0, 1, 2]
    assert non_dominated_sort(vs) == [[0, 1, 2], [3], [4]]


def test_matches_pymoo() -> None:
    pymoo = pytest.importorskip("pymoo.util.nds.non_dominated_sorting")
    import numpy as np

    rng = random.Random(1)
    vs = [(rng.random(), rng.random(), rng.random()) for _ in range(40)]
    ours = non_dominated_sort(vs)
    theirs = pymoo.NonDominatedSorting().do(np.array(vs))
    assert [sorted(f) for f in ours] == [sorted(int(i) for i in f) for f in theirs]


def test_crowding_distance() -> None:
    d = crowding_distance([(0, 3), (1, 2), (2, 1), (3, 0)])
    assert d[0] == d[3] == float("inf")
    assert d[1] == pytest.approx(4 / 3) and d[2] == pytest.approx(4 / 3)  # 2/3 per axis, 2 axes
    assert crowding_distance([]) == []


def test_hypervolume_known_values() -> None:
    assert hypervolume([(1, 1, 1)], (2, 2, 2)) == pytest.approx(1.0)
    assert hypervolume([(1, 0, 0), (0, 1, 0)], (2, 2, 2)) == pytest.approx(6.0)
    assert hypervolume([(3, 3, 3)], (2, 2, 2)) == 0.0
    assert hypervolume([(0.5, 0.5)], (1, 1)) == pytest.approx(0.25)


def brute_hv(points, ref):  # type: ignore[no-untyped-def]
    axes = [sorted({p[k] for p in points} | {ref[k]}) for k in range(3)]
    total = 0.0
    for i, j, k in itertools.product(*[range(len(a) - 1) for a in axes]):
        corner = (axes[0][i], axes[1][j], axes[2][k])
        if any(all(p[m] <= corner[m] for m in range(3)) for p in points):
            total += (axes[0][i + 1] - corner[0]) * (axes[1][j + 1] - corner[1]) * (axes[2][k + 1] - corner[2])
    return total


def test_hypervolume_matches_bruteforce_and_pymoo() -> None:
    rng = random.Random(7)
    for _ in range(10):
        pts = [(rng.random(), rng.random(), rng.random()) for _ in range(8)]
        assert hypervolume(pts, (1, 1, 1)) == pytest.approx(brute_hv(pts, (1, 1, 1)))
    ind = pytest.importorskip("pymoo.indicators.hv")
    import numpy as np

    pts = [(rng.random(), rng.random(), rng.random()) for _ in range(12)]
    assert hypervolume(pts, (1, 1, 1)) == pytest.approx(float(ind.HV(ref_point=np.array([1, 1, 1]))(np.array(pts))))


def test_archive_evicts_dominated_and_rejects_dominated_and_duplicates() -> None:
    a = ParetoArchive()
    assert a.add(ev("a", 0.5, 2.0, 100))
    assert a.add(ev("b", 0.9, 1.0, 50))  # dominates a
    assert [m.candidate.id for m in a] == ["b"]
    assert not a.add(ev("c", 0.4, 3.0, 200))  # dominated
    assert not a.add(ev("d", 0.9, 1.0, 50))  # exact duplicate objectives
    assert a.add(ev("e", 0.95, 5.0, 50))  # trade-off
    assert len(a) == 2


def test_same_id_replaces() -> None:
    a = ParetoArchive()
    a.add(ev("x", 0.5, 1.0, 10))
    a.add(ev("x", 0.7, 1.0, 10))
    assert len(a) == 1 and a.members[0].objectives.quality == 0.7


def test_prune_keeps_extremes_and_respects_size() -> None:
    a = ParetoArchive(max_size=3)
    for i in range(6):  # a perfect trade-off front
        a.add(ev(f"p{i}", i / 5, float(i), 10 + i))
    assert len(a) == 3
    ids = {m.candidate.id for m in a}
    assert {"p0", "p5"} <= ids


def test_serialize_round_trip_and_hv_monotone() -> None:
    ref = Objectives(quality=0.0, cost_usd_per_1k=10.0, length_tokens=1000)
    a = ParetoArchive()
    hvs = []
    for e in [ev("a", 0.5, 5.0, 500), ev("b", 0.8, 8.0, 300), ev("c", 0.9, 9.0, 100), ev("d", 0.6, 1.0, 900)]:
        a.add(e)
        hvs.append(a.hypervolume(ref))
    assert hvs == sorted(hvs)
    b = ParetoArchive.from_json(a.to_json())
    assert {m.candidate.id for m in b} == {m.candidate.id for m in a}
    assert b.hypervolume(ref) == pytest.approx(a.hypervolume(ref))
