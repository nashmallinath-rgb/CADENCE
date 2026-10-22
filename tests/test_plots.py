from pathlib import Path

import pytest

from cadence.benchmark.plots import load_members, plot_pareto_2d, plot_pareto_3d
from cadence.pareto import ParetoArchive
from cadence.schemas import Candidate, Evaluated, Objectives

PNG = b"\x89PNG"


def ev(i: int, q: float, c: float, n: int) -> Evaluated:
    return Evaluated(candidate=Candidate(id=f"c{i}", template=f"t{i}"),
                     objectives=Objectives(quality=q, cost_usd_per_1k=c, length_tokens=n), n_examples=4)


MEMBERS = [ev(1, 0.9, 1.0, 50), ev(2, 0.7, 0.4, 20), ev(3, 0.4, 0.1, 5)]
CASES = {"many": MEMBERS, "empty": [], "single": MEMBERS[:1], "duplicates": [ev(1, 0.5, 0.2, 10)] * 3,
         "zero_cost": [ev(i, 0.2 * i, 0.0, 10 * i) for i in range(1, 4)]}


def check_png(p: Path) -> None:
    assert p.exists() and p.stat().st_size > 500 and p.read_bytes()[:4] == PNG


@pytest.mark.parametrize("name", list(CASES))
def test_plots_write_pngs_for_every_input(tmp_path: Path, name: str) -> None:
    check_png(plot_pareto_3d(CASES[name], tmp_path / "deep" / "dir" / "p3d.png"))  # creates parent directories
    paths = plot_pareto_2d(CASES[name], tmp_path / "pareto")
    assert [p.name for p in paths] == ["pareto_quality_cost.png", "pareto_quality_length.png", "pareto_cost_length.png"]
    for p in paths:
        check_png(p)


def test_load_members_round_trips_an_archive(tmp_path: Path) -> None:
    a = ParetoArchive(10)
    for m in MEMBERS:
        a.add(m)
    p = tmp_path / "archive.json"
    p.write_text(a.to_json(), encoding="utf-8")
    got = load_members(p)
    assert sorted(m.candidate.id for m in got) == ["c1", "c2", "c3"]
    assert got[0].objectives.quality == a.members[0].objectives.quality
    assert load_members(tmp_path / "archive.json") and not load_members(_empty(tmp_path))


def _empty(tmp_path: Path) -> Path:
    p = tmp_path / "empty.json"
    p.write_text("[]")
    return p


def test_load_members_failures(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_members(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        load_members(bad)


def test_report_picks_up_the_real_plots_module(tmp_path: Path) -> None:
    from cadence.benchmark.report import make_plots

    a = ParetoArchive(10)
    for m in MEMBERS:
        a.add(m)
    (tmp_path / "archive.json").write_text(a.to_json(), encoding="utf-8")
    got = make_plots(tmp_path)
    assert got["3D Pareto"] == "pareto_3d.png" and len(got) == 4
    assert (tmp_path / "pareto_3d.png").exists()
