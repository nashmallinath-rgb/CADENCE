"""Provider smoke checks via LiteLLM. Never print secret values."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

from cadence.config import ROOT, load_models_config

GEMINI_KEY_ENV = "GEMINI_API_KEY"
XAI_KEY_ENV = "XAI_API_KEY"

_PROVIDERS: dict[str, tuple[str, str]] = {
    "gemini": (GEMINI_KEY_ENV, "gemini"),
    "xai": (XAI_KEY_ENV, "xai"),
}


def _redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _provider_of(model: str) -> str | None:
    prefix = model.split("/", 1)[0]
    if prefix in _PROVIDERS:
        return prefix
    return None


def _pick_probe_model(roles: dict[str, str], provider: str) -> str | None:
    preferred = (
        "cheap_judge",
        "generator",
        "reflector",
        "strong_judge",
        "second_family_judge",
    )
    for name in preferred:
        model = roles.get(name, "")
        if _provider_of(model) == provider:
            return model
    for model in roles.values():
        if _provider_of(model) == provider:
            return model
    return None


def _tiny_completion(model: str) -> None:
    import litellm

    litellm.drop_params = True
    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=8,
        temperature=0,
    )


def _list_gemini_models(api_key: str) -> list[str]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    names: list[str] = []
    for item in payload.get("models", []):
        raw = str(item.get("name", ""))
        short = raw.split("/")[-1] if raw else ""
        if short:
            names.append(f"gemini/{short}")
    return names


def _list_xai_models(api_key: str) -> list[str]:
    url = "https://api.x.ai/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    names: list[str] = []
    for item in payload.get("data", []):
        ident = str(item.get("id", ""))
        if ident:
            names.append(ident if ident.startswith("xai/") else f"xai/{ident}")
    return names


def run_check_setup() -> int:
    """Print OK/FAIL per provider. Return process exit code."""
    load_dotenv(ROOT / ".env")
    secrets = [os.getenv(GEMINI_KEY_ENV) or "", os.getenv(XAI_KEY_ENV) or ""]
    env_path = ROOT / ".env"
    if not env_path.exists():
        print("WARN: .env not found. Copy .env.example to .env and add keys.")

    try:
        models_cfg = load_models_config()
    except Exception as exc:
        print(f"FAIL: could not load configs/models.yaml ({type(exc).__name__})")
        return 1

    roles = models_cfg.roles.model_dump()
    exit_code = 0

    for provider, (env_name, _) in _PROVIDERS.items():
        key = os.getenv(env_name) or ""
        probe = _pick_probe_model(roles, provider)
        print(f"--- provider: {provider} ---")
        if not key.strip():
            print(f"FAIL: {env_name} is missing or empty")
            if probe:
                print(f"model (configured, not called): {probe}")
            exit_code = 1
            continue
        if not probe:
            print(f"FAIL: no {provider} model configured in configs/models.yaml")
            exit_code = 1
            continue
        try:
            _tiny_completion(probe)
            print(f"OK: tiny completion succeeded")
            print(f"model used: {probe}")
        except Exception as exc:
            exit_code = 1
            msg = _redact(f"{type(exc).__name__}: {exc}", secrets)
            print(f"FAIL: tiny completion ({msg})")
            print(f"model used: {probe}")

        try:
            listed = (
                _list_gemini_models(key) if provider == "gemini" else _list_xai_models(key)
            )
            print(f"models available to this key ({len(listed)}):")
            for name in listed:
                print(f"  {name}")
            if not listed:
                print("  (none returned)")
        except urllib.error.HTTPError as exc:
            body = _redact(exc.read().decode("utf-8", errors="replace")[:300], secrets)
            print(f"FAIL: model listing HTTP {exc.code} {_redact(body, secrets)}")
        except Exception as exc:
            print(f"FAIL: model listing ({_redact(f'{type(exc).__name__}: {exc}', secrets)})")

    return exit_code
