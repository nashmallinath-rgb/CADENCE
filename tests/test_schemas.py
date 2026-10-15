import pytest
from pydantic import ValidationError

from cadence.schemas import Candidate, Evaluated, Objectives, Score, Usage


def test_usage_add_sums_tokens_usd_and_units() -> None:
    a = Usage(input_tokens=10, output_tokens=5, usd=0.5, units={"images": 1})
    b = Usage(input_tokens=1, output_tokens=2, usd=0.25, units={"images": 2, "audio_seconds": 3.0})
    c = a + b
    assert (c.input_tokens, c.output_tokens, c.total_tokens) == (11, 7, 18)
    assert c.usd == pytest.approx(0.75)
    assert c.units == {"images": 3, "audio_seconds": 3.0}
    assert a.units == {"images": 1}  # operands untouched


def test_sum_of_usages_works_from_empty_start() -> None:
    total = sum([Usage(input_tokens=1), Usage(input_tokens=2)], Usage())
    assert total.input_tokens == 3


def test_quality_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Score(quality=1.5)
    with pytest.raises(ValidationError):
        Objectives(quality=-0.1, cost_usd_per_1k=0.0, length_tokens=1)


def test_evaluated_round_trips_through_json() -> None:
    ev = Evaluated(
        candidate=Candidate(id="c1", template="Answer: {q}", parent_ids=["c0"], generation=2),
        objectives=Objectives(quality=0.8, cost_usd_per_1k=1.2, length_tokens=40),
        n_examples=16,
        escalation_ratio=0.25,
    )
    assert Evaluated.model_validate_json(ev.model_dump_json()) == ev
