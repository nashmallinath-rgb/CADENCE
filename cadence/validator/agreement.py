"""Chance-corrected agreement statistics."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable, Literal

import numpy as np


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable],
                weights: Literal["linear", "quadratic"] | None = None) -> float:
    """Cohen's kappa. Weighted variants need orderable labels (sorted order = scale order).

    Degenerate case (expected disagreement is zero, e.g. both raters constant): 1.0 if identical else 0.0.
    """
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be non-empty and the same length")
    labels = sorted(set(a) | set(b))  # type: ignore[type-var]
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    n = conf.sum()
    expected = np.outer(conf.sum(1), conf.sum(0)) / n
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j).astype(float)
    elif weights == "quadratic":
        w = ((i - j) ** 2).astype(float)
    else:
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0 if list(a) == list(b) else 0.0
    return float(1.0 - (w * conf).sum() / denom)


def binarize(scores: Sequence[float], threshold: float = 5.0) -> list[int]:
    return [int(s >= threshold) for s in scores]


def icc_2_1(matrix: Sequence[Sequence[float]]) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss).

    `matrix` is n subjects x k raters (or repeated trials). NaN if there is no variance at all.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("need at least 2 subjects and 2 raters")
    n, k = x.shape
    grand = x.mean()
    ssr = k * ((x.mean(1) - grand) ** 2).sum()
    ssc = n * ((x.mean(0) - grand) ** 2).sum()
    sst = ((x - grand) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return float("nan")
    return float((msr - mse) / denom)
