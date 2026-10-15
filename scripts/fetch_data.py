"""Download the text-stream datasets from Hugging Face into data/raw/<name>.jsonl (gitignored).

Run from the repo root (needs internet; tests never do):
    python scripts/fetch_data.py                # all four
    python scripts/fetch_data.py gsm8k sst2     # some
    python scripts/fetch_data.py --max-rows 800

The dataset ids and column names below come from memory and were NOT verified against the Hub.
If one fails, the error lists what the Hub actually returned; fix SOURCES and tell Claude.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"

# name -> (hf id, config or None, splits to pool, columns to keep)
SOURCES: dict[str, tuple[str, str | None, list[str], list[str]]] = {
    "gsm8k": ("openai/gsm8k", "main", ["train", "test"], ["question", "answer"]),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", ["validation"], ["id", "question", "answer", "type", "level", "context"]),
    "sst2": ("stanfordnlp/sst2", None, ["train", "validation"], ["idx", "sentence", "label"]),
    "ifbench": ("allenai/IFBench_test", None, ["train"], ["key", "prompt", "instruction_id_list"]),
}


def fetch(name: str, max_rows: int, seed: int) -> int:
    from datasets import load_dataset

    hf_id, config, splits, cols = SOURCES[name]
    rows: list[dict] = []
    for sp in splits:
        ds = load_dataset(hf_id, config, split=sp) if config else load_dataset(hf_id, split=sp)
        missing = [c for c in cols if c not in ds.column_names]
        if missing:
            raise KeyError(f"{hf_id}:{sp} lacks columns {missing}; it has {ds.column_names}")
        rows.extend({c: r[c] for c in cols} for r in ds)
    if name == "sst2":
        rows = [r for r in rows if r["label"] in (0, 1)]
    random.Random(seed).shuffle(rows)
    rows = rows[:max_rows]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(SOURCES))
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    code = 0
    for n in args.names:
        try:
            print(f"{n}: wrote {fetch(n, args.max_rows, args.seed)} rows to data/raw/{n}.jsonl")
        except Exception as exc:  # noqa: BLE001
            code = 1
            print(f"{n}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
    return code


if __name__ == "__main__":
    sys.exit(main())
