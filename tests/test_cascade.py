import json

import pytest

from cadence.cascade import Cascade
from cadence.config import CascadeConfig
from cadence.judge import Judge
from cadence.llm import MockClient
from cadence.schemas import Output
from tests.judge_helpers import FakeTask, ex


def j(score: float, conf: float) -> str:
    return json.dumps({"score": score, "confidence": conf})


def make(cheap_script, strong_script=None, tau=0.70, delta=2.0, client_kw=None):  # type: ignore[no-untyped-def]
    client = MockClient(script={"cheap_judge": cheap_script, "strong_judge": strong_script or j(9, 0.95)},
                        **(client_kw or {}))
    return client, Cascade(Judge(client, "cheap_judge", 3), Judge(client, "strong_judge", 3), tau, delta)


OUT = Output(kind="text", text="some response")


def roles(client: MockClient) -> list[str]:
    return [c.role for c in client.calls]


def test_auto_scored_examples_never_call_a_judge() -> None:
    client, cas = make(j(5, 0.9))
    s = cas.score(FakeTask(), ex(0).model_copy(update={"id": "auto-1"}), OUT)
    assert s.quality == 1.0 and not s.judged and not s.escalated
    assert client.call_count == 0
    assert cas.stats.n_auto == 1 and cas.stats.n_judged == 0 and cas.stats.escalation_ratio == 0.0


def test_confident_and_consistent_cheap_judge_does_not_escalate() -> None:
    client, cas = make(j(7, 0.9))
    s = cas.score(FakeTask(), ex(1), OUT)
    assert s.judged and not s.escalated and s.quality == pytest.approx(0.7)
    assert roles(client) == ["cheap_judge"] * 3


def test_low_confidence_escalates_and_strong_verdict_replaces_cheap() -> None:
    client, cas = make(j(3, 0.5), j(9, 0.95))
    s = cas.score(FakeTask(), ex(1), OUT)
    assert s.escalated and s.quality == pytest.approx(0.9)
    assert roles(client) == ["cheap_judge"] * 3 + ["strong_judge"] * 3
    assert s.components["cheap"] == pytest.approx(0.3) and s.components["strong"] == pytest.approx(0.9)


def test_high_spread_escalates_even_when_confident() -> None:
    cheap = [j(2, 0.95), j(8, 0.95), j(5, 0.95)]  # spread 6 > delta 2
    client, cas = make(cheap)
    assert cas.score(FakeTask(), ex(1), OUT).escalated


def test_thresholds_are_strict_at_the_boundary() -> None:
    _, cas = make(j(5, 0.70))  # confidence == tau
    assert not cas.score(FakeTask(), ex(1), OUT).escalated
    _, cas = make([j(4, 0.9), j(6, 0.9), j(5, 0.9)])  # spread == delta
    assert not cas.score(FakeTask(), ex(1), OUT).escalated
    _, cas = make(j(5, 0.69))
    assert cas.score(FakeTask(), ex(1), OUT).escalated


def test_unparseable_cheap_judge_always_escalates() -> None:
    _, cas = make("garbage")
    assert cas.score(FakeTask(), ex(1), OUT).escalated


def test_escalation_ratio_counts_only_judged_items() -> None:
    # judged item 1: confident; item 2: unsure -> 1 of 2 judged escalated; auto item excluded
    seq = [j(7, 0.9)] * 3 + [j(7, 0.4)] * 3
    _, cas = make(seq)
    auto = ex(0).model_copy(update={"id": "auto-0"})
    cas.score(FakeTask(), auto, OUT)
    cas.score(FakeTask(), ex(1), OUT)
    cas.score(FakeTask(), ex(2), OUT)
    st = cas.stats
    assert (st.n_scored, st.n_auto, st.n_judged, st.n_escalated) == (3, 1, 2, 1)
    assert st.escalation_ratio == pytest.approx(0.5)


def test_spend_accounting_matches_per_score_usage() -> None:
    client, cas = make(j(3, 0.4), j(9, 0.9), client_kw={"usd_per_input_token": 0.001, "usd_per_output_token": 0.002})
    s1 = cas.score(FakeTask(), ex(1), OUT)  # escalates
    _, cas_ok = make(j(7, 0.9), client_kw={"usd_per_input_token": 0.001, "usd_per_output_token": 0.002})
    s2 = cas_ok.score(FakeTask(), ex(2), OUT)  # does not
    assert s1.usage.usd > s2.usage.usd > 0  # escalation costs strictly more
    assert cas.stats.judge_usage.usd == pytest.approx(s1.usage.usd)
    assert cas.stats.judge_usage.total_tokens == s1.usage.total_tokens


def test_task_without_score_or_rubric_is_an_error() -> None:
    class Broken:
        def automatic_score(self, ex, out):  # type: ignore[no-untyped-def]
            return None

        def rubric(self, ex):  # type: ignore[no-untyped-def]
            return None

    _, cas = make(j(5, 0.9))
    with pytest.raises(ValueError):
        cas.score(Broken(), ex(1), OUT)


def test_from_config_uses_thresholds_and_trials() -> None:
    cas = Cascade.from_config(MockClient(), CascadeConfig(tau=0.5, delta=1.0, cheap_trials=2, strong_trials=4))
    assert (cas.tau, cas.delta, cas.cheap.trials, cas.strong.trials) == (0.5, 1.0, 2, 4)
