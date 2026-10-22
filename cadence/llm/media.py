"""Media helpers and provider adapters (image generation, speech, transcription).

- `MediaResponse`: what a media backend returns.
- WAV helpers (`wav_duration`, `pcm_to_wav`) use only the standard library.
- Gemini image/TTS go through the REST API with `urllib` (LiteLLM's Gemini media support is uncertain, and this adds
  no dependency). Everything else uses LiteLLM. All of this is imported only by `cadence.llm`.
- NONE OF THE LIVE ADAPTERS HAVE BEEN RUN AGAINST A REAL PROVIDER YET; `scripts/probe_media.py` is how you find out.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Any

_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class MediaResponse:
    data: bytes = b""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    units: dict[str, float] = field(default_factory=dict)


def wav_duration(data: bytes) -> float:
    """Seconds of audio in WAV bytes; 0.0 if the bytes are not a readable WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


def pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def data_url(data: bytes, mime: str | None = None) -> str:
    return f"data:{mime or image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


# ---- Gemini REST (image + TTS) -------------------------------------------------------------------------------------
def _gemini_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
    return key


def _post_gemini(model: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    from cadence.llm.client import TransientLLMError  # late import: client imports this module

    name = model.removeprefix("gemini/")
    key = _gemini_key()
    req = urllib.request.Request(
        f"{_GEMINI}/{name}:generateContent", data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key, "User-Agent": "cadence/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:300].replace(key, "[KEY]")
        if e.code in (408, 429, 500, 502, 503, 504):
            raise TransientLLMError(f"Gemini HTTP {e.code}: {msg}") from None
        raise RuntimeError(f"Gemini HTTP {e.code}: {msg}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise TransientLLMError(f"Gemini connection problem: {type(e).__name__}") from None


def _first_inline(resp: dict[str, Any], prefix: str) -> tuple[bytes, str]:
    for cand in resp.get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and str(inline.get("mimeType") or inline.get("mime_type", "")).startswith(prefix):
                return base64.b64decode(inline["data"]), str(inline.get("mimeType") or inline.get("mime_type"))
    reason = (resp.get("promptFeedback") or {}).get("blockReason") or "no media part in the response"
    raise RuntimeError(f"Gemini returned no {prefix.rstrip('/')} data ({reason})")


def _usage_tokens(resp: dict[str, Any]) -> tuple[int, int]:
    u = resp.get("usageMetadata") or {}
    return int(u.get("promptTokenCount", 0) or 0), int(u.get("candidatesTokenCount", 0) or 0)


def gemini_image(model: str, prompt: str, params: dict[str, Any]) -> MediaResponse:
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
    resp = _post_gemini(model, body)
    data, _mime = _first_inline(resp, "image/")
    i, o = _usage_tokens(resp)
    return MediaResponse(data=data, input_tokens=i, output_tokens=o, units={"images": 1.0})


def gemini_speech(model: str, text: str, params: dict[str, Any]) -> MediaResponse:
    voice = str(params.get("voice", "Kore"))
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    resp = _post_gemini(model, body)
    pcm, mime = _first_inline(resp, "audio/")
    m = re.search(r"rate=(\d+)", mime)
    wav = pcm if pcm[:4] == b"RIFF" else pcm_to_wav(pcm, rate=int(m.group(1)) if m else 24000)
    i, o = _usage_tokens(resp)
    return MediaResponse(data=wav, input_tokens=i, output_tokens=o, units={"audio_seconds": wav_duration(wav)})


# ---- LiteLLM paths ---------------------------------------------------------------------------------------------------
def litellm_image(model: str, prompt: str, params: dict[str, Any]) -> MediaResponse:
    if model.startswith("gemini/") and "imagen" not in model:
        return gemini_image(model, prompt, params)
    import litellm

    litellm.suppress_debug_info = True
    resp = litellm.image_generation(model=model, prompt=prompt, n=1, timeout=120, num_retries=0, **params)
    item = resp.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        data = base64.b64decode(b64)
    else:  # a URL result
        with urllib.request.urlopen(item.url, timeout=60) as r:  # noqa: S310 - provider-returned URL
            data = r.read()
    return MediaResponse(data=data, units={"images": 1.0})


def litellm_speech(model: str, text: str, params: dict[str, Any]) -> MediaResponse:
    if model.startswith("gemini/"):
        return gemini_speech(model, text, params)
    import litellm

    litellm.suppress_debug_info = True
    resp = litellm.speech(model=model, input=text, voice=params.get("voice", "alloy"), response_format="wav",
                          timeout=120, num_retries=0)
    data = resp.content
    return MediaResponse(data=data, input_tokens=0, units={"audio_seconds": wav_duration(data)})


def litellm_transcribe(model: str, audio: bytes, params: dict[str, Any]) -> MediaResponse:
    import litellm

    litellm.suppress_debug_info = True
    buf = io.BytesIO(audio)
    buf.name = "audio.wav"
    resp = litellm.transcription(model=model, file=buf, timeout=120, num_retries=0, **params)
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else "") or ""
    return MediaResponse(text=str(text), units={"audio_seconds": wav_duration(audio)})
