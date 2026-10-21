"""Phase 7: media client calls, mock media, vision judging, image and audio tasks. Offline only."""
from __future__ import annotations

import json
import threading
import urllib.error
from pathlib import Path

import pytest
from tenacity import wait_none

from cadence.cascade import Cascade
from cadence.config import CascadeConfig, load_models_config
from cadence.cost import ModelPrice, Pricing, load_pricing
from cadence.judge import Judge
from cadence.llm import BudgetExceeded, BudgetGuard, LLMClient, MemoryCache, MockClient, TransientLLMError
from cadence.llm import media
from cadence.llm.client import IMAGE_TOKEN_ESTIMATE, prompt_tokens
from cadence.llm.mock import mock_wav, mock_wav_text, tiny_png
from cadence.schemas import Example, GenRequest, Output
from cadence.tasks import get_task
from cadence.tasks.audio import AudioTask, metrics
from cadence.tasks.image import ImageTask

ROLES = {"image_gen": "gemini/img", "tts": "gemini/tts", "asr": "groq/whisper", "vision_judge": "gemini/v"}
PRICING = Pricing(default=ModelPrice(), models={
    "gemini/img": ModelPrice(units={"images": 0.05}),
    "gemini/tts": ModelPrice(units={"audio_seconds": 0.001}),
    "groq/whisper": ModelPrice(units={"audio_seconds": 0.0001}),
})
ROOT = Path(__file__).resolve().parents[1]


def _client(budget: BudgetGuard | None = None, failures: list[BaseException] | None = None):  # type: ignore[no-untyped-def]
    state = {"img": 0, "tts": 0, "asr": 0, "fail": list(failures or [])}

    def image(model, prompt, params):  # type: ignore[no-untyped-def]
        state["img"] += 1
        if state["fail"]:
            raise state["fail"].pop(0)
        return media.MediaResponse(data=tiny_png(prompt.encode().ljust(4, b"x")), units={"images": 1.0})

    def speech(model, text, params):  # type: ignore[no-untyped-def]
        state["tts"] += 1
        w = mock_wav(text, 2.0)
        return media.MediaResponse(data=w, units={"audio_seconds": media.wav_duration(w)})

    def transcribe(model, audio, params):  # type: ignore[no-untyped-def]
        state["asr"] += 1
        return media.MediaResponse(text=mock_wav_text(audio), units={"audio_seconds": media.wav_duration(audio)})

    c = LLMClient(ROLES, budget=budget or BudgetGuard(10.0, 100, 1_000_000), pricing=PRICING, cache=MemoryCache(),
                  image_backend=image, speech_backend=speech, transcribe_backend=transcribe,
                  retry_wait=wait_none(), sleeper=lambda s: None)
    return c, state


def _req(role: str, text: str = "a red apple") -> GenRequest:
    return GenRequest(role=role, messages=[{"role": "user", "content": text}])


# ------------------------------------------------------------------ helpers
def test_wav_helpers_roundtrip_and_garbage() -> None:
    w = media.pcm_to_wav(b"\x00\x00" * 24000, rate=24000)
    assert media.wav_duration(w) == pytest.approx(1.0)
    assert media.wav_duration(b"not a wav") == 0.0
    assert media.wav_duration(b"") == 0.0


def test_image_mime_and_data_url() -> None:
    png = tiny_png(b"abc")
    assert media.image_mime(png) == "image/png" and media.image_mime(b"\xff\xd8\xffabc") == "image/jpeg"
    assert media.image_mime(b"zzz") == "application/octet-stream"
    assert media.data_url(png).startswith("data:image/png;base64,")


def test_prompt_tokens_does_not_count_base64_images() -> None:
    big = "A" * 3_000_000  # a 2 MB image as base64 must not look like 750k tokens
    msgs = [{"role": "user", "content": [{"type": "text", "text": "rate it"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + big}}]}]
    assert prompt_tokens(msgs) < IMAGE_TOKEN_ESTIMATE + 20
    assert prompt_tokens([{"role": "user", "content": "plain text"}]) > 0


# ------------------------------------------------------------------ LLMClient media
def test_image_usage_units_and_cost() -> None:
    c, st = _client()
    data, u = c.image(_req("image_gen"))
    assert data[:4] == b"\x89PNG" and u.units == {"images": 1.0} and u.usd == pytest.approx(0.05)
    assert st["img"] == 1 and c.budget.calls == 1 and c.budget.spent_usd == pytest.approx(0.05)


def test_media_cache_hit_is_free_and_returns_same_bytes_and_usage() -> None:
    c, st = _client()
    d1, u1 = c.image(_req("image_gen"))
    d2, u2 = c.image(_req("image_gen"))
    assert d1 == d2 and u1 == u2 and st["img"] == 1 and c.budget.calls == 1 and c.stats.cache_hits == 1
    c.image(_req("image_gen", "something else"))
    assert st["img"] == 2


def test_speech_and_transcribe_round_trip_and_audio_in_cache_key() -> None:
    c, st = _client()
    wav, tu = c.speech(_req("tts", "Turn left at the gate."))
    assert tu.units["audio_seconds"] == pytest.approx(2.0, abs=0.01) and tu.usd == pytest.approx(0.002, abs=1e-5)
    text, au = c.transcribe(wav, GenRequest(role="asr", messages=[]))
    assert text == "Turn left at the gate." and au.units["audio_seconds"] > 0
    wav2, _ = c.speech(_req("tts", "Different words."))
    text2, _ = c.transcribe(wav2, GenRequest(role="asr", messages=[]))
    assert text2 == "Different words." and st["asr"] == 2  # same request, different audio => not a cache hit


def test_media_budget_precheck_blocks_before_the_call() -> None:
    c, st = _client(BudgetGuard(0.06, 100, 1_000_000))
    c.image(_req("image_gen", "one"))  # 0.05 spent
    with pytest.raises(BudgetExceeded):
        c.image(_req("image_gen", "two"))  # would reach 0.10 > 0.06
    assert st["img"] == 1


def test_media_retries_transient_errors_then_succeeds_and_bubbles_permanent_ones() -> None:
    c, st = _client(failures=[TransientLLMError("503"), TransientLLMError("503")])
    d, _ = c.image(_req("image_gen"))
    assert d and st["img"] == 3 and c.stats.retries == 2
    c2, _ = _client(failures=[RuntimeError("bad request")])
    with pytest.raises(RuntimeError):
        c2.image(_req("image_gen"))


def test_media_calls_are_thread_safe() -> None:
    c, st = _client()
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            c.image(_req("image_gen", f"prompt {i}"))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and c.budget.calls == 16 and c.budget.spent_usd == pytest.approx(0.8)


# ------------------------------------------------------------------ Gemini REST adapter (no network)
def test_gemini_adapter_parses_inline_data_and_wraps_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    png = tiny_png(b"xyz")
    pcm = b"\x00\x00" * 12000
    sent: dict = {}

    def fake_post(model, body, timeout=120.0):  # type: ignore[no-untyped-def]
        sent["body"] = body
        if "IMAGE" in body["generationConfig"]["responseModalities"]:
            part = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}
        else:
            part = {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": base64.b64encode(pcm).decode()}}
        return {"candidates": [{"content": {"parts": [{"text": "ok"}, part]}}], "usageMetadata": {"promptTokenCount": 7}}

    monkeypatch.setattr(media, "_post_gemini", fake_post)
    img = media.gemini_image("gemini/x", "an apple", {})
    assert img.data == png and img.units == {"images": 1.0} and img.input_tokens == 7
    sp = media.gemini_speech("gemini/x", "hello", {"voice": "Puck"})
    assert media.wav_duration(sp.data) == pytest.approx(0.5) and sp.units["audio_seconds"] == pytest.approx(0.5)
    assert sent["body"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"


def test_gemini_adapter_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="no image"):
        media._first_inline({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, "image/")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET123")

    class Boom:
        def __init__(self, code: int) -> None:
            self.code = code

        def __call__(self, req, timeout=0):  # type: ignore[no-untyped-def]
            import io

            raise urllib.error.HTTPError("u", self.code, "m", {}, io.BytesIO(b"quota SECRET123 used"))  # type: ignore[arg-type]

    monkeypatch.setattr(media.urllib.request, "urlopen", Boom(429))
    with pytest.raises(TransientLLMError):
        media._post_gemini("gemini/x", {})
    monkeypatch.setattr(media.urllib.request, "urlopen", Boom(400))
    with pytest.raises(RuntimeError) as ei:
        media._post_gemini("gemini/x", {})
    assert "SECRET123" not in str(ei.value) and "[KEY]" in str(ei.value)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        media._post_gemini("gemini/x", {})


# ------------------------------------------------------------------ MockClient media
def test_mock_media_deterministic_priced_budgeted() -> None:
    m = MockClient(unit_prices={"images": 0.04, "audio_seconds": 0.001}, budget=BudgetGuard(1.0, 100, 10**6))
    a, ua = m.image(_req("image_gen"))
    b, _ = m.image(_req("image_gen"))
    assert a == b and ua.usd == pytest.approx(0.04) and m.budget.spent_usd == pytest.approx(0.08)
    wav, us = m.speech(_req("tts", "Speak calmly: hello there my friend"))
    assert mock_wav_text(wav) == "hello there my friend"  # directive is not spoken by default
    assert us.units["audio_seconds"] == pytest.approx(len("hello there my friend") / 15.0, abs=0.01)
    loud = MockClient(reads_directive_aloud=True)
    assert mock_wav_text(loud.speech(_req("tts", "Speak calmly: hi"))[0]) == "Speak calmly: hi"
    tight = MockClient(unit_prices={"images": 0.5}, budget=BudgetGuard(0.6, 100, 10**6))
    tight.image(_req("image_gen"))
    with pytest.raises(BudgetExceeded):
        tight.image(_req("image_gen", "x"))


def test_mock_transcribe_script_can_corrupt() -> None:
    m = MockClient(script={"asr": lambda req: req.messages[0]["content"].replace("left", "right")})
    wav, _ = m.speech(_req("tts", "turn left now"))
    assert m.transcribe(wav, GenRequest(role="asr", messages=[]))[0] == "turn right now"
    assert MockClient().transcribe(b"junk", GenRequest(role="asr", messages=[]))[0] == ""


# ------------------------------------------------------------------ vision judging
def test_judge_attaches_image_as_multimodal_content() -> None:
    j = Judge(MockClient(), "vision_judge", trials=2)
    png = tiny_png(b"abc")
    r = j.build_request(Example(id="e", input={"concept": "x"}), "(img)", "rubric", 0, (png, "image/png"))
    content = r.messages[-1]["content"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    plain = j.build_request(Example(id="e", input={}), "t", "rubric", 0)
    assert isinstance(plain.messages[-1]["content"], str)  # text path unchanged
    res = j.score(Example(id="e", input={"concept": "x"}), "(img)", "rubric", media=(png, "image/png"))
    assert 0 <= res.score_0_10 <= 10


def test_cascade_passes_image_only_for_image_outputs(tmp_path: Path) -> None:
    seen: list[object] = []

    class Spy:
        trials = 1

        def __init__(self) -> None:
            self.role = "spy"

        def score(self, ex, text, rubric, **kw):  # type: ignore[no-untyped-def]
            seen.append(kw.get("media"))
            from cadence.judge import JudgeResult

            return JudgeResult(score_0_10=8.0, confidence_0_1=0.9, spread=0.0)

    class T:
        def automatic_score(self, ex, out):  # type: ignore[no-untyped-def]
            return None

        def rubric(self, ex):  # type: ignore[no-untyped-def]
            return "r"

    casc = Cascade(Spy(), Spy())  # type: ignore[arg-type]
    ex = Example(id="e", input={})
    p = tmp_path / "i.png"
    p.write_bytes(tiny_png(b"q"))
    casc.score(T(), ex, Output(kind="image", path=str(p)))
    casc.score(T(), ex, Output(kind="text", text="hello"))
    assert seen[0] is not None and seen[0][1] == "image/png" and seen[1] is None


# ------------------------------------------------------------------ image task
def test_image_dataset_is_valid() -> None:
    recs = [json.loads(line) for line in (ROOT / "data/tasks/image_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 50 and len({r["id"] for r in recs}) == 50 and len({r["concept"] for r in recs}) == 50
    assert len({r["category"] for r in recs}) >= 5
    for r in recs:
        assert 6 <= len(r["concept"].split()) <= 40 and 2 <= len(r["must_include"]) <= 4
        assert all(m.lower() in r["concept"].lower() for m in r["must_include"])


def test_image_task_generate_writes_png_and_scores_via_judge(tmp_path: Path) -> None:
    t = ImageTask(media_dir=tmp_path)
    c = MockClient(unit_prices={"images": 0.04})
    ex = t.load("opt", n=1)[0]
    out = t.generate(c, t.build_request(t.seed_prompt(), ex))
    assert out.kind == "image" and Path(out.path).read_bytes()[:4] == b"\x89PNG" and out.usage.usd == pytest.approx(0.04)
    assert tmp_path in Path(out.path).parents
    assert t.automatic_score(ex, out) is None and ex.reference is None
    rub = t.rubric(ex)
    assert all(m in rub for m in ex.meta["must_include"])
    out2 = t.generate(c, t.build_request("Different prefix.", ex))
    assert out2.path != out.path  # different template => different folder, no overwrite
    cas = Cascade.from_config(c, CascadeConfig(), "vision_judge", "vision_judge_strong")
    sc = cas.score(t, ex, out)
    assert sc.judged and 0 <= sc.quality <= 1


def test_image_task_splits_and_errors(tmp_path: Path) -> None:
    t = ImageTask(media_dir=tmp_path)
    ids = [e.id for s in ("opt", "heldout", "calib") for e in t.load(s)]
    assert len(ids) == 50 == len(set(ids))
    with pytest.raises(FileNotFoundError):
        ImageTask(path=tmp_path / "nope.jsonl").load("opt")
    with pytest.raises(ValueError):
        ImageTask(records=[{"id": "a", "concept": "x"}, {"id": "a", "concept": "y"}]).load("opt")


# ------------------------------------------------------------------ audio metrics + task
def test_wer_maths() -> None:
    assert metrics.wer("turn left now", "Turn left, now!") == 0.0  # case and punctuation ignored
    assert metrics.wer("a b c d", "a x c d") == pytest.approx(0.25)
    assert metrics.wer("a b", "") == 1.0
    assert metrics.wer("a b", "a b c d") == pytest.approx(1.0)  # two insertions on two words
    with pytest.raises(ValueError):
        metrics.wer("", "anything")


@pytest.mark.parametrize("cps,expected", [(0, 0.0), (4, 0.0), (7, 0.5), (10, 1.0), (14, 1.0), (18, 1.0), (24, 0.5),
                                          (30, 0.0), (99, 0.0)])
def test_rate_score_boundaries(cps: float, expected: float) -> None:
    assert metrics.rate_score(cps) == pytest.approx(expected)


def test_audio_quality_formula_and_cps() -> None:
    assert metrics.audio_quality(0.0, 1.0) == pytest.approx(1.0)
    assert metrics.audio_quality(1.0, 1.0) == pytest.approx(0.2)
    assert metrics.audio_quality(3.0, 0.0) == 0.0  # WER above 1 is clipped, never negative
    assert metrics.chars_per_second("abcd", 2.0) == 2.0 and metrics.chars_per_second("abcd", 0.0) == 0.0


def test_audio_dataset_is_valid() -> None:
    recs = [json.loads(line) for line in (ROOT / "data/tasks/audio_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 50 and len({r["id"] for r in recs}) == 50 and len({r["text"] for r in recs}) == 50
    for r in recs:
        assert r["text"].isascii() and 8 <= len(r["text"].split()) <= 40 and r["text"][-1] in ".!?"


def test_audio_task_round_trip_cost_and_components(tmp_path: Path) -> None:
    t = AudioTask(media_dir=tmp_path)
    c = MockClient(unit_prices={"audio_seconds": 0.001})
    ex = t.load("opt", n=1)[0]
    out = t.generate(c, t.build_request("Speak calmly", ex))
    assert out.kind == "audio" and out.text == ex.input["text"] and Path(out.path).exists()
    secs = out.usage.units["audio_seconds"]
    assert out.usage.usd == pytest.approx(secs * 0.001)  # TTS only; the ASR call is not deployment cost
    assert c.calls[-1].role == "asr"
    q = t.automatic_score(ex, out)
    comps = t.auto_components(ex, out)
    assert comps["wer"] == 0.0 and comps["chars_per_second"] == pytest.approx(15.0, abs=0.5)
    assert q == pytest.approx(0.8 + 0.2 * comps["rate_score"]) and q > 0.95
    casc = Cascade.from_config(c, CascadeConfig())
    sc = casc.score(t, ex, out)
    assert not sc.judged and sc.components["wer"] == 0.0 and sc.quality == pytest.approx(q)


def test_audio_quality_drops_when_asr_hears_wrong_or_directive_read_aloud(tmp_path: Path) -> None:
    t = AudioTask(media_dir=tmp_path)
    ex = t.load("opt", n=1)[0]
    req = t.build_request("Speak calmly", ex)
    good = t.automatic_score(ex, t.generate(MockClient(), req))
    noisy = MockClient(script={"asr": lambda r: " ".join(r.messages[0]["content"].split()[:-3])})
    assert t.automatic_score(ex, t.generate(noisy, req)) < good
    loud = MockClient(reads_directive_aloud=True)
    assert t.automatic_score(ex, t.generate(loud, req)) < good
    empty = MockClient(script={"asr": lambda r: ""})
    assert t.automatic_score(ex, t.generate(empty, req)) <= 0.2 + 1e-9


def test_audio_task_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AudioTask(path=tmp_path / "nope.jsonl").load("opt")
    with pytest.raises(ValueError):
        AudioTask(records=[{"id": "a", "text": "x y"}, {"id": "a", "text": "z"}]).load("opt")


# ------------------------------------------------------------------ config + registry + CLI
def test_roles_and_pricing_config_load() -> None:
    r = load_models_config().roles
    assert r.vision_judge and r.vision_judge_strong and r.asr and r.same_family_judge
    p = load_pricing()
    assert p.for_model(r.image_gen).units.get("images", 0) >= 0
    assert "gemini/gemini-2.5-flash-preview-tts" in p.models  # single definition, no duplicate-key surprise


def test_registry_builds_media_tasks_with_kwargs(tmp_path: Path) -> None:
    assert get_task("image", media_dir=tmp_path).media_dir == tmp_path
    assert get_task("audio", media_dir=tmp_path).stream == "audio"
    assert get_task("toy").name == "toy"
    with pytest.raises(KeyError):
        get_task("nope")


@pytest.mark.e2e
@pytest.mark.parametrize("stream", ["image", "audio"])
def test_cli_run_mock_media_streams(stream: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import cadence.config as config
    from cadence.cli import main

    monkeypatch.setattr(config, "ROOT", tmp_path)
    rc = main(["run", "--stream", stream, "--task", stream, "--mock", "--no-mlflow", "--max-gen", "2",
               "--minibatch", "3", "--children", "2", "--max-calls", "5000", "--run-id", "t"])
    out = capsys.readouterr().out
    assert rc == 0 and "Pareto menu" in out
    assert list((tmp_path / "runs" / "t" / "media").rglob("*.*"))  # media was written under the run directory
