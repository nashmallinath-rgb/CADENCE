"""Find which image / TTS / ASR / vision-judge models your keys can actually use. Prints model names only, never keys.

Run from the repo root:  python scripts/probe_media.py
It discovers candidates from your Gemini and Groq model lists, makes ONE tiny call per candidate (a few cents at most
on a paid key, free on free tiers), and prints a PASS/FAIL table plus the models.yaml lines to paste.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from cadence.config import load_models_config  # noqa: E402
from cadence.llm.media import data_url, gemini_image, gemini_speech, litellm_transcribe, wav_duration  # noqa: E402
from cadence.llm.mock import tiny_png  # noqa: E402

RESULTS: dict[str, list[str]] = {}


def _get(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "cadence-setup/0.1", **headers})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gemini_models() -> list[str]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return []
    try:
        data = _get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", {"x-goog-api-key": key})
    except Exception as e:  # noqa: BLE001
        print(f"  could not list Gemini models: {type(e).__name__}")
        return []
    return sorted(m["name"].removeprefix("models/") for m in data.get("models", []))


def groq_models() -> list[str]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return []
    try:
        data = _get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
    except Exception as e:  # noqa: BLE001
        print(f"  could not list Groq models: {type(e).__name__}")
        return []
    return sorted(m["id"] for m in data.get("data", []))


def try_call(role: str, model: str, fn) -> object | None:  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace(os.getenv("GEMINI_API_KEY", "\0"), "[KEY]").replace(os.getenv("GROQ_API_KEY", "\0"), "[KEY]")
        print(f"  FAIL  {role:<12} {model:<48} {type(e).__name__}: {msg[:140]}")
        return None
    print(f"  PASS  {role:<12} {model:<48} {time.perf_counter() - t0:.1f}s")
    RESULTS.setdefault(role, []).append(model)
    return out


def tone_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(int(8000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True)
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return buf.getvalue()


def main() -> int:
    roles = load_models_config().roles
    gem, groq = gemini_models(), groq_models()
    print(f"Gemini models visible: {len(gem)}   Groq models visible: {len(groq)}")
    if not gem and not groq:
        print("No keys found. Fill GEMINI_API_KEY / GROQ_API_KEY in .env first.")
        return 1

    print("\n== image generation (Gemini REST) ==")
    cands = [m for m in gem if ("image" in m or "imagen" in m) and "embed" not in m][:4]
    configured = roles.image_gen.removeprefix("gemini/")
    cands = ([configured] if configured not in cands else []) + cands
    for m in cands:
        try_call("image_gen", "gemini/" + m, lambda m=m: gemini_image("gemini/" + m, "A red apple on a white table.", {}))

    print("\n== text to speech (Gemini REST) ==")
    wav: bytes | None = None
    cands = [m for m in gem if "tts" in m][:3]
    configured = roles.tts.removeprefix("gemini/")
    cands = ([configured] if configured not in cands else []) + cands
    for m in cands:
        out = try_call("tts", "gemini/" + m, lambda m=m: gemini_speech("gemini/" + m, "Turn left at the next junction.", {}))
        if out is not None and wav is None:
            wav = out.data  # type: ignore[attr-defined]
            print(f"        audio length {wav_duration(wav):.1f}s")

    print("\n== speech to text (Groq Whisper via LiteLLM) ==")
    audio = wav or tone_wav()
    cands = [m for m in groq if "whisper" in m][:3]
    configured = roles.asr.removeprefix("groq/")
    cands = ([configured] if configured not in cands else []) + cands
    for m in cands:
        out = try_call("asr", "groq/" + m, lambda m=m: litellm_transcribe("groq/" + m, audio, {}))
        if out is not None:
            print(f"        transcript: {out.text[:60]!r}")  # type: ignore[attr-defined]

    print("\n== vision judge and same-family judge (LiteLLM chat) ==")
    import litellm

    litellm.suppress_debug_info = True
    img = tiny_png(bytes((200, 30, 30)))
    msgs = [{"role": "user", "content": [{"type": "text", "text": "What colour is this image? One word."},
                                          {"type": "image_url", "image_url": {"url": data_url(img, "image/png")}}]}]
    for role, model in (("vision_judge", roles.vision_judge), ("vision_judge", roles.vision_judge_strong)):
        out = try_call(role, model, lambda model=model: litellm.completion(model=model, messages=msgs, max_tokens=50, timeout=60))
        if out is not None:
            print(f"        answer: {out.choices[0].message.content!r}")  # type: ignore[attr-defined]
    try_call("same_family", roles.same_family_judge, lambda: litellm.completion(
        model=roles.same_family_judge, messages=[{"role": "user", "content": "Reply with the word OK."}], max_tokens=20, timeout=60))

    print("\n== paste into configs/models.yaml (first PASS per role; edit if you prefer another) ==")
    for role, key in (("image_gen", "image_gen"), ("tts", "tts"), ("asr", "asr"), ("same_family", "same_family_judge")):
        got = RESULTS.get(role)
        print(f"  {key}: {got[0]}" if got else f"  # {key}: no working model found")
    v = RESULTS.get("vision_judge", [])
    print(f"  vision_judge: {v[0]}" if v else "  # vision_judge: none worked")
    print(f"  vision_judge_strong: {v[1] if len(v) > 1 else (v[0] if v else '(none)')}")
    if "image_gen" not in RESULTS:
        print("\nNo image model worked: the image stream stays mock-only until one does (do not switch to a paid model silently).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
