import json

import pytest

from cadence.llm import BudgetExceeded, BudgetGuard, MockClient, cached_complete, get_client, reset_default_client
from cadence.schemas import GenRequest


def req(role: str = "generator", text: str = "hi", **params) -> GenRequest:  # type: ignore[no-untyped-def]
    return GenRequest(role=role, messages=[{"role": "user", "content": text}], params=params)


def test_deterministic_for_same_request_and_different_across_requests() -> None:
    a, b = MockClient(), MockClient()
    assert a.complete(req())[0] == b.complete(req())[0]
    assert a.complete(req(text="one"))[0] != a.complete(req(text="two"))[0]


def test_judge_roles_return_parseable_json_in_range() -> None:
    client = MockClient()
    for i in range(30):
        text, _ = client.complete(req(role="cheap_judge", text=f"item {i}"))
        data = json.loads(text)
        assert 0 <= data["score"] <= 10
        assert 0.5 <= data["confidence"] <= 0.99


def test_script_by_role_cycles_and_supports_callables() -> None:
    client = MockClient(script={"generator": ["a", "b"], "reflector": "fixed", "cheap_judge": lambda r: "cb"})
    assert [client.complete(req())[0] for _ in range(3)] == ["a", "b", "a"]
    assert client.complete(req(role="reflector"))[0] == "fixed"
    assert client.complete(req(role="cheap_judge"))[0] == "cb"
    assert client.complete(req(role="strong_judge"))[0].startswith("{")  # unscripted role falls back


def test_global_callable_script_and_call_recording() -> None:
    client = MockClient(script=lambda r: r.messages[-1]["content"].upper())
    assert client.complete(req(text="yo"))[0] == "YO"
    assert client.call_count == 1 and client.calls[0].role == "generator"


def test_free_by_default_and_priced_when_asked() -> None:
    assert MockClient().complete(req())[1].usd == 0.0
    _, usage = MockClient(usd_per_input_token=0.5).complete(req(text="a b c"))
    assert usage.usd == pytest.approx(0.5 * usage.input_tokens)


def test_optional_budget_stops_the_mock() -> None:
    client = MockClient(budget=BudgetGuard(1.0, 2, 10_000))
    client.complete(req(text="a"))
    client.complete(req(text="b"))
    with pytest.raises(BudgetExceeded):
        client.complete(req(text="c"))


def test_default_client_is_mock_when_env_set(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CADENCE_MOCK", "1")
    reset_default_client()
    try:
        assert isinstance(get_client(), MockClient)
        text, _ = cached_complete("generator", [{"role": "user", "content": "x"}])
        assert text.startswith("mock:generator:")
    finally:
        reset_default_client()
