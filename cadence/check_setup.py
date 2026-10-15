"""Live smoke check: one tiny call per configured role. Never prints secret values."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from cadence.config import ROOT, load_models_config

# LiteLLM provider prefix -> env var holding its key
KEY_ENVS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
}
JUDGE_ROLES = ("cheap_judge", "strong_judge", "second_family_judge")
TEXT_ROLES = ("generator", "reflector", *JUDGE_ROLES)  # media roles are checked in Phase 7


def _redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return text


def _call(model: str, prompt: str, max_tokens: int) -> str:
    import litellm

    litellm.drop_params = True  # e.g. reasoning models reject temperature
    resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


def run_check_setup() -> int:
    """Print OK/FAIL per role. Returns a process exit code (0 = everything usable)."""
    from cadence.judge import parse_trial

    load_dotenv(ROOT / ".env")
    secrets = [os.getenv(v) or "" for v in KEY_ENVS.values()]
    if not (ROOT / ".env").exists():
        print("WARN: .env not found. Copy .env.example to .env and add keys.")
    try:
        roles = load_models_config().roles.model_dump()
    except Exception as exc:
        print(f"FAIL: could not load configs/models.yaml ({type(exc).__name__})")
        return 1

    code = 0
    for role in TEXT_ROLES:
        model = roles.get(role, "")
        provider = model.split("/", 1)[0]
        env = KEY_ENVS.get(provider)
        print(f"--- {role}: {model} ---")
        if env is None:
            print(f"FAIL: unknown provider prefix '{provider}'")
            code = 1
            continue
        if not (os.getenv(env) or "").strip():
            print(f"FAIL: {env} is missing or empty")
            code = 1
            continue
        t0 = time.time()
        try:
            if role in JUDGE_ROLES:
                text = _call(model, 'Rate "2+2=4" for correctness. Reply with ONLY JSON: '
                                    '{"score": <0-10>, "confidence": <0-1>, "rationale": "<short>"}', 400)
                ok = parse_trial(text) is not None
                print(("OK" if ok else "WARN") + f": judge JSON {'parsed' if ok else 'NOT parseable'} ({time.time() - t0:.1f}s)")
                code = code or (0 if ok else 1)
            else:
                text = _call(model, "Reply with the single word: pong", 200)
                ok = bool(text.strip())
                print(("OK" if ok else "WARN") + f": reply {'received' if ok else 'was empty'} ({time.time() - t0:.1f}s)")
                code = code or (0 if ok else 1)
        except Exception as exc:
            code = 1
            status = getattr(exc, "status_code", "")
            print(f"FAIL: {type(exc).__name__} {status} {_redact(str(exc), secrets)[:200]}")
    return code
