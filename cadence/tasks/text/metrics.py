"""Answer extraction and exact-match style metrics for the text stream."""
from __future__ import annotations

import re
import string

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _clean_num(s: str) -> str:
    return s.replace(",", "").replace("$", "")


def gsm8k_reference(answer: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    return _clean_num(answer.split("####")[-1].strip())


def extract_number(text: str) -> str | None:
    """Final numeric answer from a model reply: '####', \\boxed{}, 'answer is', else the last number."""
    if "####" in text:
        m = re.search(r"####\s*(" + _NUM + ")", text)
        if m:
            return _clean_num(m.group(1))
    m = re.findall(r"\\boxed\{\s*(" + _NUM + r")\s*\}", text)
    if m:
        return _clean_num(m[-1])
    m = re.findall(r"answer\s*(?:is|:)\s*\**\s*(" + _NUM + ")", text, flags=re.I)
    if m:
        return _clean_num(m[-1])
    nums = re.findall(_NUM, text)
    return _clean_num(nums[-1]) if nums else None


def numeric_equal(a: str | None, b: str | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except ValueError:
        return False


def normalize_answer(s: str) -> str:
    """SQuAD-style: lowercase, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def first_answer_line(text: str) -> str:
    """Short-answer tasks: first non-empty line, minus a leading 'Answer:' label."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^[\*_]*answer[\*_]*\s*[:\-]\s*[\*_]*\s*", "", line, flags=re.I)
            return line.strip().strip("*_").strip()
    return ""


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def sentiment_label(text: str) -> str | None:
    """'positive' / 'negative' by first occurrence; None if neither word appears."""
    t = text.lower()
    hits = [(t.find(w), w) for w in ("positive", "negative") if w in t]
    return min(hits)[1] if hits else None
