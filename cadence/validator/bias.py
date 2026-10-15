"""Judge bias probes: position bias (swap test) and verbosity bias (Spearman)."""
from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel
from scipy.stats import spearmanr

from cadence.judge import PairwiseJudge
from cadence.schemas import Example


class PositionBias(BaseModel):
    n: int
    flip_rate: float  # fraction of pairs whose verdict changes when the order is swapped
    first_slot_rate: float  # of decisive verdicts, fraction that picked slot A (0.5 = no preference)


def position_bias(judge: PairwiseJudge, pairs: Sequence[tuple[Example, str, str]], rubric: str) -> PositionBias:
    if not pairs:
        raise ValueError("no pairs")
    flips = 0
    slot_a = decisive = 0
    for ex, x, y in pairs:
        v1 = judge.compare(ex, x, y, rubric)  # x in slot A
        v2 = judge.compare(ex, y, x, rubric)  # x in slot B
        w1 = {"A": "x", "B": "y", "tie": "tie"}[v1]
        w2 = {"A": "y", "B": "x", "tie": "tie"}[v2]
        flips += w1 != w2
        for v in (v1, v2):
            if v != "tie":
                decisive += 1
                slot_a += v == "A"
    return PositionBias(n=len(pairs), flip_rate=flips / len(pairs),
                        first_slot_rate=slot_a / decisive if decisive else 0.5)


class VerbosityBias(BaseModel):
    n: int
    rho: float
    p_value: float


def verbosity_bias(scores: Sequence[float], lengths: Sequence[float]) -> VerbosityBias:
    """Spearman correlation between judge score and response length. Only meaningful when true quality is
    independent of length (e.g. correct answers only), otherwise a positive rho may be legitimate."""
    if len(scores) != len(lengths) or len(scores) < 3:
        raise ValueError("need >= 3 paired observations")
    if len(set(scores)) < 2 or len(set(lengths)) < 2:
        return VerbosityBias(n=len(scores), rho=0.0, p_value=1.0)
    rho, p = spearmanr(scores, lengths)
    return VerbosityBias(n=len(scores), rho=float(rho), p_value=float(p))
