import pytest

from cadence.llm import BudgetExceeded, BudgetGuard
from cadence.schemas import Usage


def test_calls_cap_raises_before_the_extra_call() -> None:
    g = BudgetGuard(max_usd=10, max_calls=2, max_tokens=10_000)
    for _ in range(2):
        g.check()
        g.record(Usage(input_tokens=1, output_tokens=1, usd=0.01))
    with pytest.raises(BudgetExceeded) as info:
        g.check()
    assert info.value.kind == "calls"


def test_usd_estimate_blocks_overshoot() -> None:
    g = BudgetGuard(max_usd=1.0, max_calls=100, max_tokens=10_000)
    g.record(Usage(usd=0.8))
    g.check(est_usd=0.1)  # fits
    with pytest.raises(BudgetExceeded) as info:
        g.check(est_usd=0.3)  # 0.8 + 0.3 > 1.0
    assert info.value.kind == "usd"


def test_token_cap() -> None:
    g = BudgetGuard(max_usd=10, max_calls=100, max_tokens=100)
    g.record(Usage(input_tokens=60, output_tokens=30))
    with pytest.raises(BudgetExceeded) as info:
        g.check(est_tokens=20)
    assert info.value.kind == "tokens"


def test_ok_and_remaining() -> None:
    g = BudgetGuard(max_usd=1.0, max_calls=2, max_tokens=100)
    assert g.ok()
    g.record(Usage(input_tokens=10, usd=0.4))
    assert g.remaining() == {"usd": pytest.approx(0.6), "calls": 1, "tokens": 90}
    g.record(Usage(input_tokens=10, usd=0.4))
    assert not g.ok()  # calls exhausted
