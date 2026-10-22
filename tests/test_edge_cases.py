"""Edge cases for archive, hypervolume, loop helpers, template extraction and resume."""
import random
from pathlib import Path

import pytest

from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.llm import MockClient
from cadence.orchestrator import LoopSettings, extract_template, optimize
from cadence.orchestrator.loop import minibatch_for, select_parents
from cadence.pareto import ParetoArchive, hypervolume
from cadence.schemas import Candidate, Evaluated, Objectives
from cadence.tasks.toy import ToyTask


def ev(cid: str, q: float, c: float, n: int) -> Evaluated:
    return Evaluated(candidate=Candidate(id=cid, template=cid),
                     objectives=Objectives(quality=q, cost_usd_per_1k=c, length_tokens=n), n_examples=1)


REF = Objectives(quality=0.0, cost_usd_per_1k=10.0, length_tokens=100)


def test_archive_edges() -> None:
    a = ParetoArchive(5)
    assert a.hypervolume(REF) == 0.0 and len(a) == 0
    assert a.add(ev("a", 0.5, 1.0, 10))
    assert not a.add(ev("b", 0.5, 1.0, 10))  # identical objectives: nothing gained
    assert a.add(ev("a", 0.6, 1.0, 10)) and len(a) == 1  # same id replaces the old evaluation
    assert a.members[0].objectives.quality == 0.6
    one = ParetoArchive(1)
    for i in range(4):
        one.add(ev(f"m{i}", 0.2 * (i + 1), 1.0 - 0.1 * i, 10 + i))
    assert len(one) == 1
    full = ParetoArchive(10)
    for m in (ev("x", 0.9, 5.0, 50), ev("y", 0.5, 1.0, 10)):
        full.add(m)
    back = ParetoArchive.from_json(full.to_json(), 10)
    assert [m.candidate.id for m in back.members] == [m.candidate.id for m in full.members]
    with pytest.raises(ValueError):
        ParetoArchive(0)


def test_hypervolume_edges() -> None:
    ref = (0.0, 10.0, 100.0)
    assert hypervolume([(-0.5, 5.0, 50.0)], ref) == pytest.approx(0.5 * 5.0 * 50.0)
    assert hypervolume([(0.1, 5.0, 50.0)], ref) == 0.0  # worse than the reference quality: no volume
    assert hypervolume([(-0.5, 20.0, 50.0)], ref) == 0.0  # outside the reference box
    p = (-0.5, 5.0, 50.0)
    assert hypervolume([p, p, p], ref) == hypervolume([p], ref)  # duplicates add nothing
    assert hypervolume([], ref) == 0.0


def test_select_parents_and_minibatch_edges() -> None:
    members = [ev(f"m{i}", 0.1 * i, 1.0 - 0.05 * i, 10 + i) for i in range(1, 4)]
    assert len(select_parents(members, 10, random.Random(0))) == 3
    assert select_parents([], 2, random.Random(0)) == []
    pool = ToyTask().load("opt", n=5)
    assert len(minibatch_for(pool, 7, 99)) == 5
    with pytest.raises(ValueError):
        minibatch_for([], 0, 4)


def test_extract_template_edges() -> None:
    assert extract_template("") == ""
    assert extract_template("<template></template>") == ""
    assert extract_template("<template>no close") == "no close"  # truncated reply: keep the text, drop the tag
    assert extract_template("<TEMPLATE> A </TEMPLATE> trailing") == "A"
    assert extract_template("<template>a</template><template>b</template>") == "a"  # first one wins


def test_resume_with_missing_archive_is_a_clear_error(tmp_path: Path) -> None:
    c = MockClient()
    s = LoopSettings(max_gen=1, minibatch=4, final_sample=8)
    d = tmp_path / "r"
    optimize(ToyTask(), c, Cascade.from_config(c, CascadeConfig()), s, d, log=lambda m: None)
    (d / "archive.json").unlink()
    with pytest.raises(ValueError, match="archive.json"):
        optimize(ToyTask(), c, Cascade.from_config(c, CascadeConfig()), s, d, resume=True, log=lambda m: None)
