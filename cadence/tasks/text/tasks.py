"""Text/reasoning stream: GSM8K, HotpotQA, SST-2 (auto-scored) and IFBench (judged)."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cadence.config import ROOT
from cadence.schemas import Example, Output
from cadence.tasks.base import Split, Task
from cadence.tasks.splits import select
from cadence.tasks.text import metrics

RAW_DIR = ROOT / "data" / "raw"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TextTask(Task):
    """Loads raw records (HF row dicts) from `records`, or from data/raw/<name>.jsonl (see scripts/fetch_data.py)."""

    stream = "text"

    def __init__(self, records: Sequence[dict[str, Any]] | None = None, path: Path | None = None) -> None:
        self._records = list(records) if records is not None else None
        self._path = path or RAW_DIR / f"{self.name}.jsonl"
        self._examples: list[Example] | None = None

    def to_example(self, i: int, rec: dict[str, Any]) -> Example:
        raise NotImplementedError

    def _all(self) -> list[Example]:
        if self._examples is None:
            if self._records is None:
                if not self._path.exists():
                    raise FileNotFoundError(f"{self._path} not found. Run: python scripts/fetch_data.py {self.name}")
                self._records = read_jsonl(self._path)
            self._examples = [self.to_example(i, r) for i, r in enumerate(self._records)]
            ids = [e.id for e in self._examples]
            if len(set(ids)) != len(ids):
                dups = sorted({i for i in ids if ids.count(i) > 1})[:5]
                raise ValueError(f"duplicate example ids in {self.name}, e.g. {dups}")
        return self._examples

    def load(self, split: Split, n: int | None = None, seed: int = 0) -> list[Example]:
        return select(self._all(), split, n, seed)


class Gsm8kTask(TextTask):
    name = "gsm8k"

    def to_example(self, i: int, rec: dict[str, Any]) -> Example:
        return Example(id=f"gsm8k-{i}", input={"question": rec["question"]}, reference=metrics.gsm8k_reference(rec["answer"]))

    def seed_prompt(self) -> str:
        return "Solve the math problem. Think step by step, then give the final numeric answer after '####'."

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return float(metrics.numeric_equal(metrics.extract_number(out.text or ""), str(ex.reference)))

    def calibration_rubric(self, ex: Example) -> str | None:
        return ("The task is a grade-school math word problem. Work it out yourself. The response states its final "
                "answer on the line starting with '####'; that number is what counts. Score 10 if the number after '####' "
                "is correct and 0 if it is wrong, even if the working above it looks right or disagrees with it.")

    def degraded_prompt(self) -> str | None:
        return "Reply with only the final number as '#### <number>'. Do not show any work. Guess quickly."

    def corrupt_output(self, ex: Example, text: str) -> str | None:
        return metrics.corrupt_last_number(text)


class HotpotQATask(TextTask):
    name = "hotpotqa"

    def weak_seed_prompt(self) -> str | None:
        return 'Answer.'

    def to_example(self, i: int, rec: dict[str, Any]) -> Example:
        ctx = rec.get("context", {})
        titles, sents = ctx.get("title", []), ctx.get("sentences", [])
        paras = [f"[{t}] {' '.join(s)}" for t, s in zip(titles, sents)]
        return Example(
            id=str(rec.get("id", f"hotpotqa-{i}")),
            input={"question": rec["question"], "context": "\n".join(paras)},
            reference=rec["answer"],
            meta={"type": rec.get("type"), "level": rec.get("level")},
        )

    def format_input(self, ex: Example) -> str:
        return f"Context:\n{ex.input['context']}\n\nQuestion: {ex.input['question']}"

    def seed_prompt(self) -> str:
        return "Answer the question using the context. Reply with only the short answer, no explanation."

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return float(metrics.exact_match(metrics.first_answer_line(out.text or ""), str(ex.reference)))


class Sst2Task(TextTask):
    name = "sst2"

    def weak_seed_prompt(self) -> str | None:
        return 'Respond to the text.'

    def to_example(self, i: int, rec: dict[str, Any]) -> Example:
        if rec["label"] not in (0, 1):
            raise ValueError("SST-2 rows must have label 0 or 1 (drop the unlabeled test split)")
        # not rec["idx"]: train and validation both number rows from 0, so pooled idx values collide
        return Example(id=f"sst2-{i}", input={"sentence": rec["sentence"]},
                       reference="positive" if rec["label"] == 1 else "negative")

    def seed_prompt(self) -> str:
        return "Classify the sentiment of the sentence. Reply with exactly one word: positive or negative."

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return float(metrics.sentiment_label(out.text or "") == ex.reference)

    def calibration_rubric(self, ex: Example) -> str | None:
        return ("The task is to classify the sentiment of the given sentence as positive or negative. "
                "Score 10 if the response gives the correct label and 0 if it gives the wrong label.")

    def corrupt_output(self, ex: Example, text: str) -> str | None:
        return metrics.flip_sentiment(text)


class IfbenchTask(TextTask):
    """Instruction following with verifiable constraints. No verifier is bundled, so it is judged (rubric)."""

    name = "ifbench"

    def weak_seed_prompt(self) -> str | None:
        return 'Respond briefly.'

    def to_example(self, i: int, rec: dict[str, Any]) -> Example:
        return Example(id=f"ifbench-{rec.get('key', i)}", input={"prompt": rec["prompt"]}, reference=None,
                       meta={"instruction_ids": rec.get("instruction_id_list", [])})

    def seed_prompt(self) -> str:
        return "Follow every instruction and constraint in the user's request exactly."

    def automatic_score(self, ex: Example, out: Output) -> float | None:
        return None

    def rubric(self, ex: Example) -> str | None:
        return ("Score how fully the response satisfies EVERY explicit instruction and formatting constraint in the task "
                "input (length limits, required or forbidden words, format, structure, language). "
                "10 = all constraints met; deduct for each violated constraint; 0 = ignores the instructions.")
