"""Audio metrics: word error rate, speaking rate, and the combined quality score."""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace (WER should not punish commas)."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalised text. Empty hypothesis = 1.0; empty reference raises ValueError."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("reference text is empty")
    if not hyp:
        return 1.0
    import jiwer

    return float(jiwer.wer(ref, hyp))


def chars_per_second(text: str, seconds: float) -> float:
    return len(text) / seconds if seconds > 0 else 0.0


def rate_score(cps: float, lo: float = 10.0, hi: float = 18.0, floor: float = 4.0, ceil: float = 30.0) -> float:
    """1.0 inside [lo, hi] chars/second, decaying linearly to 0 at `floor` and `ceil`. cps<=0 scores 0."""
    if cps <= floor or cps >= ceil:
        return 0.0
    if cps < lo:
        return (cps - floor) / (lo - floor)
    if cps > hi:
        return (ceil - cps) / (ceil - hi)
    return 1.0


def audio_quality(w: float, rate: float, accuracy_weight: float = 0.8) -> float:
    """quality = 0.8 * (1 - min(WER, 1)) + 0.2 * rate_score."""
    return accuracy_weight * max(0.0, 1.0 - min(w, 1.0)) + (1.0 - accuracy_weight) * rate
