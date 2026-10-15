import pytest

from cadence.cost import (
    ModelPrice,
    Pricing,
    cost_per_1k,
    cost_per_1k_from_tokens,
    count_tokens,
    load_pricing,
    template_length,
    tokenizer_backend,
)
from cadence.schemas import Usage

PRICING = Pricing(
    default=ModelPrice(input_per_mtok=5.0, output_per_mtok=15.0),
    models={
        "p/cheap": ModelPrice(input_per_mtok=2.0, output_per_mtok=8.0, verify=False),
        "p/img": ModelPrice(units={"images": 0.03}),
    },
)


def test_price_tokens_by_hand() -> None:
    # 1000 in * $2/M + 200 out * $8/M = 0.002 + 0.0016
    assert PRICING.price_tokens("p/cheap", 1000, 200) == pytest.approx(0.0036)


def test_unknown_model_uses_pessimistic_default() -> None:
    assert PRICING.price_tokens("nope/x", 1_000_000, 0) == pytest.approx(5.0)


def test_unit_pricing_for_media() -> None:
    usage = Usage(units={"images": 2})
    assert PRICING.price(usage, "p/img") == pytest.approx(0.06)


def test_cost_per_1k_matches_spec_formula_by_hand() -> None:
    raw = [(1000, 200), (3000, 400)]  # avg_in = 2000, avg_out = 300
    usages = [Usage(input_tokens=i, output_tokens=o, usd=PRICING.price_tokens("p/cheap", i, o)) for i, o in raw]
    # 1000 * (2000 * 2e-6 + 300 * 8e-6) = 1000 * (0.004 + 0.0024) = 6.4
    expected = cost_per_1k_from_tokens(2000, 300, 2e-6, 8e-6)
    assert expected == pytest.approx(6.4)
    assert cost_per_1k(usages) == pytest.approx(expected)


def test_cost_per_1k_rejects_empty() -> None:
    with pytest.raises(ValueError):
        cost_per_1k([])


def test_template_length_is_stable_and_monotone() -> None:
    short = "Answer the question."
    long = short + " Think step by step and show your work before the final answer."
    assert template_length(short) == template_length(short) > 0
    assert template_length(long) > template_length(short)
    assert count_tokens("") == 0
    assert tokenizer_backend().startswith(("tiktoken:", "fallback:"))


def test_repo_pricing_yaml_loads_and_is_flagged_unverified() -> None:
    pricing = load_pricing()
    assert "xai/grok-3" in pricing.models
    assert pricing.default.input_per_mtok > 0
    assert set(pricing.unverified()) <= set(pricing.models)
    assert "openai/gpt-4.1-mini" in pricing.unverified()  # still to be checked by a human
    assert "gemini/gemini-3.1-flash-lite" not in pricing.unverified()  # verified against Google's page
