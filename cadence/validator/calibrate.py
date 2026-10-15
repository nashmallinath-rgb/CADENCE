"""Judge calibration against outputs whose correctness is known (e.g. via exact match)."""
from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from cadence.judge import Judge
from cadence.schemas import Example, Usage
from cadence.validator.agreement import binarize, cohen_kappa, icc_2_1


class CalibItem(BaseModel):
    example: Example
    response_text: str
    correct: bool  # ground truth from exact match or a human label
    rubric: str


class JudgeHealth(BaseModel):
    kappa: float
    icc: float | None  # across repeated trials; None when undefined
    n: int
    trials: int
    kappa_min: float
    flagged: bool
    rechecked: bool = False
    usage: Usage = Field(default_factory=Usage)


def calibrate(judge: Judge, items: Sequence[CalibItem], kappa_min: float = 0.60,
              threshold: float = 5.0) -> JudgeHealth:
    """Judge rates reference-free; its binarized verdict is compared with ground truth via Cohen's kappa."""
    if len(items) < 4:
        raise ValueError("calibration needs at least 4 items")
    truth = [int(i.correct) for i in items]
    if len(set(truth)) < 2:
        raise ValueError("calibration set needs both correct and incorrect items")
    scores: list[float] = []
    matrix: list[list[float]] = []
    usage = Usage()
    for it in items:
        r = judge.score(it.example, it.response_text, it.rubric)
        usage = usage + r.usage
        scores.append(r.score_0_10)
        row = [t.score for t in r.trials]
        row += [r.score_0_10] * (judge.trials - len(row))  # pad failed trials with the mean
        matrix.append(row)
    kappa = cohen_kappa(truth, binarize(scores, threshold))
    icc: float | None = None
    if judge.trials >= 2:
        v = icc_2_1(matrix)
        icc = None if math.isnan(v) else v
    return JudgeHealth(kappa=kappa, icc=icc, n=len(items), trials=judge.trials, kappa_min=kappa_min,
                       flagged=kappa < kappa_min, usage=usage)


def calibrate_with_recheck(judge: Judge, items: Sequence[CalibItem], kappa_min: float = 0.60,
                           threshold: float = 5.0, trial_multiplier: int = 2) -> tuple[JudgeHealth, Judge]:
    """If flagged, double the trials and re-check once. Returns the final health and the judge to keep using."""
    health = calibrate(judge, items, kappa_min, threshold)
    if not health.flagged:
        return health, judge
    stronger = judge.with_trials(judge.trials * trial_multiplier)
    again = calibrate(stronger, items, kappa_min, threshold)
    again.rechecked = True
    again.usage = again.usage + health.usage
    return again, stronger
