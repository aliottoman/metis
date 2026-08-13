"""The brief, read aloud — and the normalizer everything spoken stands on.

Two things are pinned. What comes out of the normalizer contains nothing a
listener would hear as noise: no markup, no addresses, no bracketed reference
markers, no table pipes. And what the Listen button costs: one synthesis per
distinct rendition, with the key covering every input that changes the audio.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import AttentionItemV1, MorningBriefV1
from waqil_api.main import create_app
from waqil_api.model_provider import ModelProviderError
from waqil_api.spoken_text import SPOKEN_MAX_CHARS, brief_to_speech, to_speech
from waqil_api.voice_audio import SpokenAudioCache


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        **overrides,
    )


def _brief(**overrides) -> MorningBriefV1:
    now = datetime(2026, 8, 13, 6, 30, tzinfo=UTC)
    return MorningBriefV1(
        generated_at=now,
        since=now - timedelta(hours=24),
        **overrides,
    )


class FakeSpeech:
    def __init__(self, *, available: bool = True, error: str = "") -> None:
        self.available = available
        self.error = error
        self.spoken: list[str] = []

    async def synthesize(self, text: str, *, voice_id: str = "", model: str = ""):
        if self.error:
            raise ModelProviderError(self.error)
        self.spoken.append(text)
        return b"ID3" + text.encode("utf-8")[:16], "audio/mpeg"


class FakeRouter:
    def __init__(self, elevenlabs) -> None:
        self.elevenlabs = elevenlabs


def test_the_normalizer_leaves_nothing_a_listener_would_hear_as_noise() -> None:
    written = (
        "## Where things stand\n"
        "The **Batelco** sizing is ready [1], per the "
        "[service request](https://example.com/sr/9912).\n"
        "- Send it today\n"
        "- Then book the review\n"
        "```python\nprint('never spoken')\n```\n"
        "See https://example.com/more for detail.\n"
    )
    spoken = to_speech(written, max_sentences=8)
    for noise in ("#", "**", "```", "http", "[1]", "(", ")", "- "):
        assert noise not in spoken
    assert "Where things stand" in spoken
    assert "Batelco" in spoken
    assert "service request" in spoken  # the label survives, the address does not
    assert "never spoken" not in spoken
    assert "Send it today." in spoken


def test_a_table_becomes_its_dimensions() -> None:
    written = (
        "Here is the split.\n\n"
        "| Region | Cores | Cost |\n"
        "| --- | --- | --- |\n"
        "| Dubai | 32 | 1200 |\n"
        "| Muscat | 16 | 640 |\n"
    )
    spoken = to_speech(written, max_sentences=6)
    assert "|" not in spoken
    assert "table of 2 rows and 3 columns" in spoken


def test_mermaid_and_inline_code_never_reach_the_voice() -> None:
    written = (
        "The flow is simple.\n"
        "```mermaid\ngraph TD; A-->B;\n```\n"
        "Run `make verify` when you are done.\n"
    )
    spoken = to_speech(written, max_sentences=6)
    assert "graph TD" not in spoken
    assert "`" not in spoken
    assert "Run make verify when you are done." in spoken


def test_the_ceiling_is_whole_sentences_where_it_can_be() -> None:
    written = " ".join(f"Sentence number {index}." for index in range(1, 9))
    assert to_speech(written) == (
        "Sentence number 1. Sentence number 2. Sentence number 3. Sentence number 4."
    )

    # One sentence longer than the whole budget still ends cleanly, on a word
    # boundary, rather than trailing off mid-word.
    runaway = "word " * 400
    clipped = to_speech(runaway)
    assert len(clipped) <= SPOKEN_MAX_CHARS
    assert clipped.endswith(".")
    assert not clipped.endswith("wor.")


def test_an_empty_answer_produces_no_speech() -> None:
    assert to_speech("") == ""
    assert to_speech("   \n\n  ") == ""
    assert to_speech("```\nonly code\n```") == ""


def test_the_brief_speaks_its_counts_without_reading_punctuation() -> None:
    brief = _brief(
        waiting_total=3,
        narrative="Two accounts moved. **Batelco** is the one that matters [2].",
        recommendation="Send the Batelco sizing before the standup.",
        changed=["Recorded a win: Ledger — BAPCO ($120,000 ARR)", "2 runs completed"],
        focus=[
            AttentionItemV1(
                key="customer_action:1",
                kind="customer_action",
                kind_label="Commitment",
                title="Send the sizing to Batelco",
                overdue=True,
            )
        ],
    )
    spoken = brief_to_speech(brief)
    assert spoken.startswith("3 things waiting.")
    assert "**" not in spoken and "[2]" not in spoken
    assert "commitment, Send the sizing to Batelco, which is overdue" in spoken
    assert "In the last day, Recorded a win" in spoken
    assert spoken.endswith("Send the Batelco sizing before the standup.")


def test_the_brief_is_sayable_with_no_model_and_nothing_waiting() -> None:
    """The facts are the brief. Prose is a bonus, and speech must not need it."""
    spoken = brief_to_speech(_brief(waiting_total=0))
    assert spoken == "Nothing is waiting for a decision."


def test_one_thing_waiting_is_said_as_one_thing() -> None:
    assert brief_to_speech(_brief(waiting_total=1)).startswith("1 thing waiting.")


def test_the_rendition_key_covers_everything_that_changes_the_audio() -> None:
    base = dict(scope="2026-08-13", voice_id="voice-a", model="flash")
    key = SpokenAudioCache.key("three things waiting", **base)
    assert key == SpokenAudioCache.key("three things waiting", **base)
    assert key != SpokenAudioCache.key("four things waiting", **base)
    assert key != SpokenAudioCache.key(
        "three things waiting", **{**base, "scope": "2026-08-14"}
    )
    assert key != SpokenAudioCache.key(
        "three things waiting", **{**base, "voice_id": "voice-b"}
    )
    assert key != SpokenAudioCache.key(
        "three things waiting", **{**base, "model": "turbo"}
    )


def test_the_cache_round_trips_and_a_miss_is_not_an_error(tmp_path) -> None:
    cache = SpokenAudioCache(tmp_path / "voice-audio")
    key = SpokenAudioCache.key("hello", scope="s", voice_id="v", model="m")
    assert cache.read(key) is None
    cache.write(key, b"ID3audio", "audio/mpeg")
    assert cache.read(key) == (b"ID3audio", "audio/mpeg")
    # An unknown container is not stored rather than stored under a lie.
    other = SpokenAudioCache.key("hi", scope="s", voice_id="v", model="m")
    cache.write(other, b"???", "application/octet-stream")
    assert cache.read(other) is None


def test_listening_renders_once_and_replays_from_disk(tmp_path) -> None:
    settings = _settings(tmp_path, elevenlabs_api_key="e")
    app = create_app(settings)
    with TestClient(app) as client:
        speech = FakeSpeech()
        app.state.runtime.model = FakeRouter(speech)

        first = client.get("/api/v1/attention/brief/audio")
        assert first.status_code == 200
        assert first.headers["content-type"] == "audio/mpeg"
        assert first.headers["cache-control"] == "no-store"
        assert first.content.startswith(b"ID3")

        second = client.get("/api/v1/attention/brief/audio")
        assert second.status_code == 200
        assert second.content == first.content
        # One synthesis, two plays: the second press cost nothing but a disk read.
        assert len(speech.spoken) == 1
        assert "waiting" in speech.spoken[0]


def test_listening_without_a_key_says_which_key(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        app.state.runtime.model = FakeRouter(FakeSpeech(available=False))
        response = client.get("/api/v1/attention/brief/audio")
        assert response.status_code == 503
        assert "WAQIL_ELEVENLABS_API_KEY" in response.json()["detail"]


def test_a_failed_synthesis_is_reported_rather_than_cached(tmp_path) -> None:
    settings = _settings(tmp_path, elevenlabs_api_key="e")
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.runtime.model = FakeRouter(FakeSpeech(error="HTTP 429: slow down"))
        response = client.get("/api/v1/attention/brief/audio")
        assert response.status_code == 502
        assert "slow down" in response.json()["detail"]
        # Nothing was written, so the next press is a real retry rather than a
        # replay of a failure.
        cached = (
            list(settings.voice_cache_dir.glob("*"))
            if settings.voice_cache_dir.exists()
            else []
        )
        assert cached == []


@pytest.mark.asyncio
async def test_the_spoken_brief_never_invents_a_number() -> None:
    """The one thing a spoken brief must not do.

    Everything the voice says about how much is waiting comes from the same
    host-counted fields the page shows; the model only ever wrote the prose
    around them, and that prose is rendered verbatim minus its markup.
    """
    brief = _brief(waiting_total=2, narrative="You closed nine actions.")
    spoken = brief_to_speech(brief)
    assert spoken.startswith("2 things waiting.")
    # The model's sentence is carried, not corrected — but it is also not the
    # source of the count, which is what keeps the headline honest.
    assert "You closed nine actions." in spoken
