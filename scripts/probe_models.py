"""Try candidate models with one tiny call each. Prints status only, never keys.

Run from the repo root:  python scripts/probe_models.py
or with your own list:   python scripts/probe_models.py gemini/gemini-3.5-flash openai/gpt-4.1-mini
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing the package

from dotenv import load_dotenv

from cadence.check_setup import KEY_ENVS, _call, _redact
from cadence.config import ROOT
from cadence.judge import parse_trial

CANDIDATES = [
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-flash-latest",
    "gemini/gemini-flash-lite-latest",
    "gemini/gemma-4-31b-it",
    "gemini/gemma-4-26b-a4b-it",
]
SKIP = ("whisper", "tts", "guard", "playai", "embed", "orpheus", "transcribe", "safeguard")


def groq_chat_models() -> list[str]:
    """Discover Groq chat models from the key's own model list (None if no key / lookup fails)."""
    import json
    import urllib.request

    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        return []
    try:
        req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:  # noqa: BLE001
        print("(could not list Groq models; check GROQ_API_KEY)")
        return []
    return ["groq/" + i for i in sorted(ids) if not any(w in i.lower() for w in SKIP)]


JUDGE_PROMPT = ('Rate "2+2=4" for correctness. Reply with ONLY JSON: '
                '{"score": <0-10>, "confidence": <0-1>, "rationale": "<short>"}')


def main(models: list[str]) -> None:
    load_dotenv(ROOT / ".env")
    secrets = [os.getenv(v) or "" for v in KEY_ENVS.values()]
    print(f"{'model':40} result")
    for m in models:
        t0 = time.time()
        try:
            text = _call(m, JUDGE_PROMPT, 400)
            ok = parse_trial(text) is not None
            res = f"OK  judge-JSON {'parsed' if ok else 'NOT parsed'}  {time.time() - t0:.1f}s"
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "status_code", "")
            res = f"FAIL {type(exc).__name__} {code}: {_redact(str(exc), secrets)[:90]!r}"
        print(f"{m:40} {res}")
        time.sleep(1)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    main(sys.argv[1:] or CANDIDATES + groq_chat_models())
