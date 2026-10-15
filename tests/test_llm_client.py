import json

import pytest
from tenacity import wait_none

from cadence.cost import ModelPrice, Pricing
from cadence.llm import (
    BackendResponse,
    BudgetExceeded,
    BudgetGuard,
    DiskCache,
    LLMClient,
    MemoryCache,
    TransientLLMError,
    make_cache_key,
)
from cadence.schemas import GenRequest

ROLES = {"generator": "gemini/test-model", "cheap_judge": "xai/test-judge"}
PRICING = Pricing(default=ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0))


class FakeBackend:
    """Counts calls; can fail first with a scripted list of exceptions."""

    def __init__(self, failures: list[BaseException] | None = None, in_tok: int = 10, out_tok: int = 5) -> None:
        self.calls = 0
        self.failures = list(failures or [])
        self.in_tok, self.out_tok = in_tok, out_tok

    def __call__(self, model, messages, params):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return BackendResponse(text=f"reply-{self.calls}", input_tokens=self.in_tok, output_tokens=self.out_tok)


def make_client(backend, **kw):  # type: ignore[no-untyped-def]
    kw.setdefault("budget", BudgetGuard(10.0, 100, 1_000_000))
    kw.setdefault("pricing", PRICING)
    kw.setdefault("cache", MemoryCache())
    kw.setdefault("retry_wait", wait_none())
    kw.setdefault("sleeper", lambda s: None)
    return LLMClient(ROLES, backend=backend, **kw)


def req(text: str = "hello world", role: str = "generator", **params) -> GenRequest:  # type: ignore[no-untyped-def]
    return GenRequest(role=role, messages=[{"role": "user", "content": text}], params=params)


def test_cache_hit_skips_backend_and_budget() -> None:
    backend = FakeBackend()
    client = make_client(backend)
    t1, u1 = client.complete(req())
    t2, u2 = client.complete(req())
    assert backend.calls == 1
    assert (t1, u1) == (t2, u2)  # hit returns the original usage, so cost objectives stay stable
    assert client.budget.calls == 1
    assert client.stats.cache_hits == 1 and client.stats.live_calls == 1


def test_different_params_or_seed_miss_the_cache() -> None:
    backend = FakeBackend()
    client = make_client(backend)
    client.complete(req(temperature=0.0))
    client.complete(req(temperature=0.7))
    assert backend.calls == 2
    base = dict(model="m", messages=[{"role": "user", "content": "x"}], params={})
    assert make_cache_key(**base, seed=0) == make_cache_key(**base, seed=0)
    assert make_cache_key(**base, seed=0) != make_cache_key(**base, seed=1)


def test_usage_is_priced_from_the_pricing_table() -> None:
    client = make_client(FakeBackend(in_tok=1_000_000, out_tok=500_000))
    _, usage = client.complete(req(max_tokens=10))
    assert usage.usd == pytest.approx(1.5)  # $1/M in + $1/M out


def test_calls_cap_raises_before_backend() -> None:
    backend = FakeBackend()
    client = make_client(backend, budget=BudgetGuard(10.0, 2, 1_000_000))
    client.complete(req("a"))
    client.complete(req("b"))
    with pytest.raises(BudgetExceeded):
        client.complete(req("c"))
    assert backend.calls == 2


def test_usd_cap_is_checked_with_an_estimate_before_the_call() -> None:
    pricey = Pricing(default=ModelPrice(input_per_mtok=100_000.0, output_per_mtok=100_000.0))  # $0.10 per token
    backend = FakeBackend(in_tok=10, out_tok=5)  # a call really costs $1.50
    client = make_client(backend, pricing=pricey, budget=BudgetGuard(2.0, 100, 1_000_000))
    client.complete(req("hi", max_tokens=5))
    with pytest.raises(BudgetExceeded) as info:
        client.complete(req("other", max_tokens=5))  # 1.5 spent + >=0.5 estimated > 2.0
    assert info.value.kind == "usd"
    assert backend.calls == 1


def test_retries_transient_errors_then_succeeds() -> None:
    backend = FakeBackend(failures=[TransientLLMError("429"), TransientLLMError("503")])
    client = make_client(backend)
    text, _ = client.complete(req())
    assert text == "reply-3" and backend.calls == 3
    assert client.stats.retries == 2


def test_status_code_429_counts_as_transient() -> None:
    class Boom(Exception):
        status_code = 429

    backend = FakeBackend(failures=[Boom("slow down")])
    assert make_client(backend).complete(req())[0] == "reply-2"


def test_non_transient_errors_are_not_retried() -> None:
    backend = FakeBackend(failures=[ValueError("bad request")])
    with pytest.raises(ValueError):
        make_client(backend).complete(req())
    assert backend.calls == 1


def test_retries_are_bounded() -> None:
    backend = FakeBackend(failures=[TransientLLMError("x")] * 10)
    with pytest.raises(TransientLLMError):
        make_client(backend, max_retries=2).complete(req())
    assert backend.calls == 3  # 1 try + 2 retries


def test_throttle_spaces_calls_per_provider() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleeper(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    client = make_client(FakeBackend(), rpm={"gemini": 60}, clock=lambda: now[0], sleeper=sleeper)
    client.complete(req("a"))  # first call goes straight through
    client.complete(req("b"))  # second must wait one second (60 rpm)
    assert sleeps == [pytest.approx(1.0)]
    client.complete(req("c", role="cheap_judge"))  # xai has no limit configured
    assert sleeps == [pytest.approx(1.0)]


def test_unknown_role_lists_known_roles() -> None:
    with pytest.raises(KeyError, match="generator"):
        make_client(FakeBackend()).complete(req(role="nonsense"))


def test_disk_cache_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = DiskCache(tmp_path / "c")
    backend = FakeBackend()
    make_client(backend, cache=cache).complete(req())
    cache.close()
    cache2 = DiskCache(tmp_path / "c")  # simulates a fresh process
    make_client(backend, cache=cache2).complete(req())
    cache2.close()
    assert backend.calls == 1


def test_cache_value_is_json_serializable() -> None:
    cache = MemoryCache()
    make_client(FakeBackend(), cache=cache).complete(req())
    (value,) = cache._data.values()
    json.dumps(value)
