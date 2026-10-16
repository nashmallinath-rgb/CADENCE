"""List the model IDs your API keys can use. Prints model names only, never keys.

Run from the repo root:  python scripts/list_models.py
Reads GEMINI_API_KEY, OPENAI_API_KEY and GROQ_API_KEY from .env (or the environment).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def get(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gemini(key: str) -> list[str]:
    data = get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", {"x-goog-api-key": key})
    return sorted(
        m["name"].removeprefix("models/")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )


def openai(key: str) -> list[str]:
    data = get("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    return sorted(m["id"] for m in data.get("data", []))


def groq(key: str) -> list[str]:
    data = get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
    return sorted(m["id"] for m in data.get("data", []))


def show(title: str, env: str, fn) -> None:  # type: ignore[no-untyped-def]
    print(f"\n== {title} ==")
    key = os.getenv(env, "").strip()
    if not key:
        print(f"{env} is not set (put it in .env).")
        return
    try:
        names = fn(key)
    except urllib.error.HTTPError as e:  # message only; the request headers (and key) are never printed
        print(f"HTTP {e.code}: the key was rejected or lacks access.")
        return
    except Exception as e:  # noqa: BLE001
        print(f"Failed: {type(e).__name__}")
        return
    print(f"{len(names)} models:")
    for n in names:
        print("  " + n)


if __name__ == "__main__":
    show("Gemini (usable for chat)", "GEMINI_API_KEY", gemini)
    show("OpenAI", "OPENAI_API_KEY", openai)
    show("Groq (use as groq/<id>)", "GROQ_API_KEY", groq)
    sys.exit(0)
