"""LLMClient: the only place CADENCE talks to a model provider.

Every call is cached, throttled per provider, retried on transient errors, and
checked against the run budget. Tests use MockClient (same interface), never this.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from cadence.cost.pricing import Pricing
from cadence.cost.tracker import count_tokens
from cadence.llm.budget import BudgetGuard
from cadence.llm import media as _media
from cadence.llm.cache import CacheBackend, DiskCache, MemoryCache, make_cache_key
from cadence.llm.mock import MockClient
from cadence.schemas import GenRequest, Usage


class TransientLLMError(RuntimeError):
    """A failure worth retrying (rate limit, timeout, 5xx)."""


_TRANSIENT_NAMES = frozenset(
    {
        "TransientLLMError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
        "TimeoutError",
        "BadGatewayError",
    }
)
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    if any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


@dataclass
class BackendResponse:
    text: str
    input_tokens: int
    output_tokens: int


Backend = Callable[[str, list[dict[str, Any]], dict[str, Any]], BackendResponse]
MediaBackend = Callable[[str, str, dict[str, Any]], _media.MediaResponse]  # (model, prompt/text, params)
TranscribeBackend = Callable[[str, bytes, dict[str, Any]], _media.MediaResponse]  # (model, audio, params)
IMAGE_TOKEN_ESTIMATE = 1500  # budget pre-check cost of one attached image (base64 text must not be counted as tokens)


def prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Token estimate for a message list. Multimodal parts: text is counted, each image is a flat estimate."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens(str(part.get("text", "")))
                else:
                    total += IMAGE_TOKEN_ESTIMATE
        else:
            total += count_tokens(str(content))
    return total


def litellm_backend(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> BackendResponse:
    """Real provider call through LiteLLM. Imported lazily so tests never load it."""
    import litellm

    litellm.drop_params = True  # reasoning models reject e.g. temperature
    litellm.suppress_debug_info = True  # silence the 'Give Feedback / Get Help' banner printed on every retry
    # Fail slow calls fast (the tenacity retry layer handles Timeout) and keep all retrying in one place so the
    # budget guard and retry stats see every call; LiteLLM's own retries are disabled.
    params = {"timeout": 90, "num_retries": 0, **params}
    resp = litellm.completion(model=model, messages=messages, **params)
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return BackendResponse(
        text=text,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


@dataclass
class ClientStats:
    live_calls: int = 0
    cache_hits: int = 0
    retries: int = 0


class LLMClient:
    """complete(req) -> (text, Usage).

    The Usage returned on a cache hit is the ORIGINAL call's usage, so a candidate's cost
    objective does not change between a fresh run and a cached rerun. Only live calls are
    charged to the budget.
    """

    def __init__(
        self,
        roles: Mapping[str, str],
        *,
        budget: BudgetGuard,
        pricing: Pricing,
        cache: CacheBackend | None = None,
        backend: Backend | None = None,
        image_backend: MediaBackend | None = None,
        speech_backend: MediaBackend | None = None,
        transcribe_backend: TranscribeBackend | None = None,
        rpm: Mapping[str, float] | None = None,
        max_retries: int = 4,
        est_output_tokens: int = 512,
        seed: int | None = None,
        retry_wait: wait_base | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.roles = dict(roles)
        self.budget = budget
        self.pricing = pricing
        self.cache: CacheBackend = cache if cache is not None else MemoryCache()
        self.backend: Backend = backend or litellm_backend
        self.image_backend: MediaBackend = image_backend or _media.litellm_image
        self.speech_backend: MediaBackend = speech_backend or _media.litellm_speech
        self.transcribe_backend: TranscribeBackend = transcribe_backend or _media.litellm_transcribe
        self.rpm = dict(rpm or {})
        self.max_retries = max_retries
        self.est_output_tokens = est_output_tokens
        self.seed = seed
        self.retry_wait = retry_wait or wait_exponential(multiplier=1, max=30)
        self._clock = clock
        self._sleeper = sleeper
        self._next_slot: dict[str, float] = {}
        self._throttle_lock = threading.Lock()
        self.stats = ClientStats()

    # -- construction from the repo configs ------------------------------------------
    @classmethod
    def from_config(cls, *, seed: int | None = None, cache_dir: Path | None = None,
                    budget: BudgetGuard | None = None) -> LLMClient:
        from dotenv import load_dotenv

        from cadence.config import ROOT, load_budget_config, load_models_config, load_run_config
        from cadence.cost.pricing import load_pricing

        load_dotenv()
        run = load_run_config()
        directory = cache_dir or (ROOT / run.llm.cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return cls(
            load_models_config().roles.model_dump(),
            budget=budget or BudgetGuard.from_config(load_budget_config()),
            pricing=load_pricing(),
            cache=DiskCache(directory),
            rpm=run.llm.rpm,
            max_retries=run.llm.max_retries,
            est_output_tokens=run.llm.est_output_tokens,
            seed=run.seed if seed is None else seed,
        )

    # -- internals -------------------------------------------------------------------
    def model_for(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"unknown role {role!r}; configured roles: {sorted(self.roles)}") from None

    def _throttle(self, model: str) -> None:
        provider = model.split("/", 1)[0]
        rpm = self.rpm.get(provider)
        if not rpm:
            return
        interval = 60.0 / rpm
        with self._throttle_lock:
            now = self._clock()
            start = max(now, self._next_slot.get(provider, 0.0))
            self._next_slot[provider] = start + interval
            wait = start - now
        if wait > 0:
            self._sleeper(wait)

    def _retry(self, model: str, fn: Callable[[], Any]) -> Any:
        def _note_retry(_state: Any) -> None:
            self.stats.retries += 1

        retrying = Retrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=self.retry_wait,
            retry=retry_if_exception(is_transient),
            before_sleep=_note_retry,
            sleep=self._sleeper,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                self._throttle(model)
                return fn()
        raise RuntimeError("unreachable")  # pragma: no cover

    def _call_with_retry(self, model: str, req: GenRequest) -> BackendResponse:
        return self._retry(model, lambda: self.backend(model, req.messages, dict(req.params)))

    # -- public API ------------------------------------------------------------------
    def complete(self, req: GenRequest) -> tuple[str, Usage]:
        model = self.model_for(req.role)
        key = make_cache_key(model, req.messages, req.params, self.seed)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit["text"], Usage.model_validate(hit["usage"])

        est_in = prompt_tokens(req.messages)
        est_out = int(req.params.get("max_tokens") or self.est_output_tokens)
        self.budget.check(
            est_tokens=est_in + est_out,
            est_usd=self.pricing.price_tokens(model, est_in, est_out),
        )

        resp = self._call_with_retry(model, req)
        usage = Usage(
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            usd=self.pricing.price_tokens(model, resp.input_tokens, resp.output_tokens),
        )
        self.stats.live_calls += 1
        self.budget.record(usage)
        self.cache.set(key, {"text": resp.text, "usage": usage.model_dump()})
        return resp.text, usage


    # -- media ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _last_text(req: GenRequest) -> str:
        content = req.messages[-1].get("content", "") if req.messages else ""
        if isinstance(content, list):
            return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content)

    def _media_call(self, kind: str, req: GenRequest, extra: dict[str, Any], est_units: dict[str, float],
                    live: Callable[[str], _media.MediaResponse]) -> tuple[_media.MediaResponse, Usage]:
        """Shared path for image/speech/transcribe: cache (bytes kept), budget pre-check, throttle+retry, record."""
        model = self.model_for(req.role)
        key = make_cache_key(model, req.messages, {**req.params, "__kind": kind, **extra}, self.seed)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return _media.MediaResponse(data=hit["data"], text=hit.get("text", "")), Usage.model_validate(hit["usage"])
        est_in = prompt_tokens(req.messages)
        est_usd = self.pricing.price(Usage(input_tokens=est_in, units=est_units), model)
        self.budget.check(est_tokens=est_in, est_usd=est_usd)
        resp: _media.MediaResponse = self._retry(model, lambda: live(model))
        usage = Usage(input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, units=dict(resp.units))
        usage.usd = self.pricing.price(usage, model)
        self.stats.live_calls += 1
        self.budget.record(usage)
        self.cache.set(key, {"data": resp.data, "text": resp.text, "usage": usage.model_dump()})
        return resp, usage

    def image(self, req: GenRequest) -> tuple[bytes, Usage]:
        """Generate one image from the last user message. Returns (PNG/JPEG bytes, Usage with units['images']=1)."""
        prompt = self._last_text(req)
        resp, usage = self._media_call("image", req, {}, {"images": 1.0},
                                       lambda m: self.image_backend(m, prompt, dict(req.params)))
        return resp.data, usage

    def speech(self, req: GenRequest) -> tuple[bytes, Usage]:
        """Text-to-speech of the last user message. Returns (WAV bytes, Usage with units['audio_seconds'])."""
        text = self._last_text(req)
        est_seconds = max(1.0, len(text) / 12.0)
        resp, usage = self._media_call("speech", req, {}, {"audio_seconds": est_seconds},
                                       lambda m: self.speech_backend(m, text, dict(req.params)))
        return resp.data, usage

    def transcribe(self, audio: bytes, req: GenRequest) -> tuple[str, Usage]:
        """Speech-to-text. `req.role` selects the model; the audio hash is part of the cache key."""
        import hashlib

        est_seconds = max(1.0, _media.wav_duration(audio))
        resp, usage = self._media_call("transcribe", req, {"audio": hashlib.sha256(audio).hexdigest()},
                                       {"audio_seconds": est_seconds},
                                       lambda m: self.transcribe_backend(m, audio, dict(req.params)))
        return resp.text, usage


# -- convenience entry point used throughout the codebase ---------------------------
_DEFAULT: LLMClient | MockClient | None = None


def get_client() -> LLMClient | MockClient:
    """Process-wide client. CADENCE_MOCK=1 returns an offline MockClient."""
    global _DEFAULT
    if _DEFAULT is None:
        if os.environ.get("CADENCE_MOCK", "").lower() in {"1", "true", "yes"}:
            _DEFAULT = MockClient()
        else:
            _DEFAULT = LLMClient.from_config()
    return _DEFAULT


def reset_default_client() -> None:
    global _DEFAULT
    _DEFAULT = None


def cached_complete(role: str, messages: list[dict[str, Any]], **params: Any) -> tuple[str, Usage]:
    """Cached, retried, budget-checked completion through the default client."""
    return get_client().complete(GenRequest(role=role, messages=messages, params=params))
