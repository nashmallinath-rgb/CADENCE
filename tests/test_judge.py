import json

import pytest
from tenacity import wait_none

from cadence.cost import ModelPrice, Pricing
from cadence.judge import Judge, PairwiseJudge, parse_trial, parse_winner
from cadence.llm import BackendResponse, BudgetGuard, LLMClient, MemoryCache, MockClient
from tests.judge_helpers import ex, noisy_judge


def test_parse_plain_fenced_and_chatty_json() -> None:
    assert parse_trial('{"score": 7, "confidence": 0.8, "rationale": "ok"}').score == 7  # type: ignore[union-attr]
    t = parse_trial('Sure!\n```json\n{"score": 6.5, "confidence": 0.9}\n```\nHope that helps')
    assert t is not None and t.score == 6.5 and t.confidence == 0.9


def test_parse_fallback_and_clamping_and_garbage() -> None:
    t = parse_trial("Score: 8, confidence: 0.7")
    assert t is not None and (t.score, t.confidence) == (8.0, 0.7)
    t = parse_trial('{"score": 42, "confidence": -3}')
    assert t is not None and (t.score, t.confidence) == (10.0, 0.0)
    assert parse_trial("no idea") is None
    assert parse_trial('{"score": "high"}') is None
    assert parse_trial('{"score": NaN}') is None


def test_aggregation_mean_confidence_and_spread() -> None:
    client = MockClient(script={"cheap_judge": [
        '{"score": 2, "confidence": 0.6}', '{"score": 8, "confidence": 0.8}', '{"score": 5, "confidence": 1.0}']})
    r = Judge(client, "cheap_judge", trials=3).score(ex(1), "resp", "rubric")
    assert r.score_0_10 == pytest.approx(5.0)
    assert r.confidence_0_1 == pytest.approx(0.8)
    assert r.spread == pytest.approx(6.0)
    assert len(r.trials) == 3 and r.parse_failures == 0


def test_unparseable_trials_are_counted_and_all_failing_gives_zero_confidence() -> None:
    mixed = MockClient(script={"cheap_judge": ["garbage", '{"score": 6, "confidence": 0.9}']})
    r = Judge(mixed, "cheap_judge", trials=2).score(ex(1), "r", "rub")
    assert r.parse_failures == 1 and r.score_0_10 == 6
    dead = MockClient(script={"cheap_judge": "garbage"})
    r = Judge(dead, "cheap_judge", trials=3).score(ex(1), "r", "rub")
    assert (r.score_0_10, r.confidence_0_1, r.parse_failures) == (0.0, 0.0, 3)


def test_judge_never_sees_the_reference_answer() -> None:
    client = MockClient()
    Judge(client, "cheap_judge", trials=2).score(ex(1, reference="SECRET42"), "my response", "rubric text")
    assert client.call_count == 2
    for req in client.calls:
        blob = json.dumps(req.messages)
        assert "SECRET42" not in blob
        assert "my response" in blob and "rubric text" in blob


def test_trials_use_distinct_cache_keys_with_the_real_client() -> None:
    n = {"calls": 0}

    def backend(model, messages, params):  # type: ignore[no-untyped-def]
        n["calls"] += 1
        return BackendResponse(text='{"score": 5, "confidence": 0.9}', input_tokens=10, output_tokens=5)

    client = LLMClient({"cheap_judge": "gemini/x"}, budget=BudgetGuard(10, 100, 100_000),
                       pricing=Pricing(default=ModelPrice(input_per_mtok=1, output_per_mtok=1)),
                       cache=MemoryCache(), backend=backend, retry_wait=wait_none(), sleeper=lambda s: None)
    judge = Judge(client, "cheap_judge", trials=3)
    judge.score(ex(1), "r", "rub")
    assert n["calls"] == 3  # trials are not collapsed by the cache
    judge.score(ex(1), "r", "rub")
    assert n["calls"] == 3  # but rerunning the same item is fully cached


def test_with_trials_returns_new_judge() -> None:
    j = Judge(MockClient(), "cheap_judge", trials=3)
    assert j.with_trials(6).trials == 6 and j.trials == 3


def test_pairwise_parse_and_compare() -> None:
    assert parse_winner('{"winner": "a"}') == "A"
    assert parse_winner('{"winner": "tie"}') == "tie"
    assert parse_winner("meh") is None
    pj = PairwiseJudge(MockClient(script={"cheap_judge": '{"winner": "B"}'}), "cheap_judge")
    assert pj.compare(ex(1), "x", "y", "rub") == "B"
    assert PairwiseJudge(MockClient(script={"cheap_judge": "??"}), "cheap_judge").compare(ex(1), "x", "y", "r") == "tie"


def test_noisy_judge_script_is_deterministic() -> None:
    a, b = MockClient(script={"cheap_judge": noisy_judge(0.3)}), MockClient(script={"cheap_judge": noisy_judge(0.3)})
    ra = Judge(a, "cheap_judge", 3).score(ex(1), "GOOD answer", "r")
    rb = Judge(b, "cheap_judge", 3).score(ex(1), "GOOD answer", "r")
    assert ra.score_0_10 == rb.score_0_10
