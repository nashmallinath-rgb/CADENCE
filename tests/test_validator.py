import math
import random

import pytest

from cadence.judge import Judge, PairwiseJudge
from cadence.llm import MockClient
from cadence.schemas import GenRequest
from cadence.validator import (
    CalibItem,
    binarize,
    calibrate,
    calibrate_with_recheck,
    cohen_kappa,
    icc_2_1,
    position_bias,
    verbosity_bias,
)
from tests.judge_helpers import ex, noisy_judge, user_text


def test_kappa_hand_values() -> None:
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)  # po .75, pe .5
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # degenerate but identical
    with pytest.raises(ValueError):
        cohen_kappa([], [])


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_kappa_matches_sklearn(weights) -> None:  # type: ignore[no-untyped-def]
    sk = pytest.importorskip("sklearn.metrics")
    rng = random.Random(3)
    a = [rng.randint(0, 4) for _ in range(80)]
    b = [min(4, max(0, x + rng.choice([-1, 0, 0, 1]))) for x in a]
    assert cohen_kappa(a, b, weights) == pytest.approx(sk.cohen_kappa_score(a, b, weights=weights))


def test_icc_matches_shrout_fleiss_1979_example() -> None:
    data = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert icc_2_1(data) == pytest.approx(0.29, abs=0.005)


def test_icc_extremes() -> None:
    assert icc_2_1([[1, 1, 1], [5, 5, 5], [9, 9, 9]]) == pytest.approx(1.0)
    assert math.isnan(icc_2_1([[4, 4], [4, 4], [4, 4]]))
    with pytest.raises(ValueError):
        icc_2_1([[1, 2]])


def test_binarize() -> None:
    assert binarize([0, 4.9, 5, 10]) == [0, 0, 1, 1]


def items(n: int = 40) -> list[CalibItem]:
    return [CalibItem(example=ex(i), response_text=("GOOD answer" if i % 2 == 0 else "BAD answer") + f" #{i}",
                      correct=i % 2 == 0, rubric="rate it") for i in range(n)]


def judge_with(noise: float, trials: int = 3, seed: int = 0) -> Judge:
    return Judge(MockClient(script={"cheap_judge": noisy_judge(noise, seed=seed)}), "cheap_judge", trials)


def test_accurate_judge_is_healthy() -> None:
    h = calibrate(judge_with(0.0), items())
    assert h.kappa == pytest.approx(1.0) and not h.flagged
    assert h.icc is not None and h.icc > 0.9 and h.n == 40


def test_slightly_noisy_judge_still_passes() -> None:
    h = calibrate(judge_with(0.1), items(60))
    assert h.kappa >= 0.60 and not h.flagged


def test_coin_flip_judge_is_flagged() -> None:
    h = calibrate(judge_with(0.5), items(60))
    assert h.kappa < 0.60 and h.flagged


def test_anti_correlated_judge_is_flagged_with_negative_kappa() -> None:
    h = calibrate(judge_with(1.0), items())
    assert h.kappa == pytest.approx(-1.0) and h.flagged


def test_calibration_input_validation() -> None:
    with pytest.raises(ValueError):
        calibrate(judge_with(0), items()[:2])
    one_class = [i.model_copy(update={"correct": True}) for i in items(8)]
    with pytest.raises(ValueError):
        calibrate(judge_with(0), one_class)


def test_recheck_doubles_trials_once_when_flagged() -> None:
    def script(req: GenRequest) -> str:
        # accurate only when the judge runs 6 passes; with 3 passes it is anti-correlated
        good = "GOOD" in user_text(req).split("RESPONSE:", 1)[-1]
        accurate = "/6]" in user_text(req)
        return '{"score": %d, "confidence": 0.9}' % (9 if good == accurate else 1)

    client = MockClient(script={"cheap_judge": script})
    health, used = calibrate_with_recheck(Judge(client, "cheap_judge", 3), items())
    assert health.rechecked and not health.flagged and used.trials == 6
    first_pass_calls = 40 * 3
    assert client.call_count == first_pass_calls + 40 * 6
    assert health.usage.total_tokens > 0


def test_recheck_skipped_when_healthy() -> None:
    health, used = calibrate_with_recheck(judge_with(0.0), items())
    assert not health.rechecked and used.trials == 3


def pairs(n: int = 10):  # type: ignore[no-untyped-def]
    return [(ex(i), f"GOOD {i}", f"BAD {i}") for i in range(n)]


def test_unbiased_pairwise_judge_has_zero_flip_rate() -> None:
    def script(req: GenRequest) -> str:
        t = user_text(req)
        a = t.split("RESPONSE A:", 1)[1].split("RESPONSE B:", 1)[0]
        return '{"winner": "%s"}' % ("A" if "GOOD" in a else "B")

    pb = position_bias(PairwiseJudge(MockClient(script={"cheap_judge": script}), "cheap_judge"), pairs(), "r")
    assert pb.flip_rate == 0.0 and pb.first_slot_rate == pytest.approx(0.5) and pb.n == 10


def test_always_pick_first_judge_is_fully_position_biased() -> None:
    pb = position_bias(PairwiseJudge(MockClient(script={"cheap_judge": '{"winner": "A"}'}), "cheap_judge"), pairs(), "r")
    assert pb.flip_rate == 1.0 and pb.first_slot_rate == 1.0


def test_position_bias_requires_pairs() -> None:
    with pytest.raises(ValueError):
        position_bias(PairwiseJudge(MockClient(), "cheap_judge"), [], "r")


def test_verbosity_bias() -> None:
    v = verbosity_bias([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert v.rho == pytest.approx(1.0)
    rng = random.Random(0)
    sc = [rng.random() for _ in range(200)]
    ln = [rng.random() for _ in range(200)]
    assert abs(verbosity_bias(sc, ln).rho) < 0.2
    assert verbosity_bias([5, 5, 5, 5], [1, 2, 3, 4]).rho == 0.0
    with pytest.raises(ValueError):
        verbosity_bias([1, 2], [1, 2])
