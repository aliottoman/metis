"""Which service hears you, and what that changes downstream.

Two things are pinned here. The choice itself — persisted, degraded when its
key disappears, refused when it names a provider with nothing behind it, and
never carrying a key back out to the browser. And the consequence of the
choice: the dictation route's size ceiling and container handling are the
selected provider's own, with the Cohere path unchanged from the day it was
the only path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.main import create_app
from waqil_api.speech_preference import VOICE_MODELS, SpeechPreferenceStore


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


class FakeSpeechProvider:
    """A transcriber that records what it was handed."""

    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self.available = available
        self.calls: list[tuple[str, str, int]] = []

    async def transcribe(
        self, audio: bytes, filename: str, media_type: str, *, language: str = ""
    ) -> str:
        self.calls.append((filename, media_type, len(audio)))
        return f"heard by {self.name}"


class FakeRouter:
    def __init__(self, cohere=None, elevenlabs=None) -> None:
        self.cohere = cohere
        self.elevenlabs = elevenlabs


def test_the_default_is_cohere_and_no_key_travels_with_it(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        cohere_api_key="sk-cohere-canary-0001",
        elevenlabs_api_key="sk-eleven-canary-0002",
    )
    store = SpeechPreferenceStore(settings)
    preference = store.load()
    assert preference.stt_provider == "cohere"
    assert preference.spoken_confirmation is False
    assert preference.cohere_available and preference.elevenlabs_available
    assert preference.voice_models == list(VOICE_MODELS)
    assert preference.voice_model == VOICE_MODELS[0]

    # Availability is a boolean, never the credential behind it — neither in
    # what is returned nor in what is written to disk.
    serialized = preference.model_dump_json()
    stored = store.save("cohere")
    on_disk = settings.speech_preference_path.read_text(encoding="utf-8")
    for canary in ("sk-cohere-canary-0001", "sk-eleven-canary-0002"):
        assert canary not in serialized
        assert canary not in stored.model_dump_json()
        assert canary not in on_disk


def test_a_choice_persists_and_is_refused_without_a_key(tmp_path) -> None:
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    store = SpeechPreferenceStore(settings)
    saved = store.save("elevenlabs", spoken_confirmation=True)
    assert saved.stt_provider == "elevenlabs"
    assert saved.spoken_confirmation is True
    assert SpeechPreferenceStore(settings).load().stt_provider == "elevenlabs"

    keyless = SpeechPreferenceStore(_settings(tmp_path / "b", cohere_api_key="c"))
    with pytest.raises(ValueError, match="WAQIL_ELEVENLABS_API_KEY"):
        keyless.save("elevenlabs")
    with pytest.raises(ValueError, match="cohere.*elevenlabs"):
        keyless.save("whisper")


def test_a_stored_choice_degrades_when_its_key_goes_away(tmp_path) -> None:
    """A key can be removed long after the choice was saved.

    Failing with "not configured" while a working transcriber sits beside it
    is a worse answer than using the one that works — the same degrade the
    model preference makes when a lane loses its key.
    """
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    SpeechPreferenceStore(settings).save("elevenlabs")

    settings.elevenlabs_api_key = ""
    assert SpeechPreferenceStore(settings).load().stt_provider == "cohere"

    # And with no ElevenLabs key at all this can only ever be Cohere, which is
    # exactly the behavior that existed before there was a choice.
    settings.cohere_api_key = ""
    assert SpeechPreferenceStore(settings).load().stt_provider == "elevenlabs"


def test_the_voice_model_comes_from_a_server_owned_allowlist(tmp_path) -> None:
    settings = _settings(tmp_path, cohere_api_key="c")
    store = SpeechPreferenceStore(settings)

    chosen = VOICE_MODELS[1]
    assert store.save("cohere", voice_model=chosen).voice_model == chosen
    # None keeps what is stored, so a surface that only knows about dictation
    # cannot silently reset the voice model.
    assert store.save("cohere").voice_model == chosen
    # "" returns to the configured default.
    assert store.save("cohere", voice_model="").voice_model == VOICE_MODELS[0]
    with pytest.raises(ValueError, match="not a voice model"):
        store.save("cohere", voice_model="anything-i-like:cloud")

    # A configured default off the allowlist is not honored either.
    settings.voice_model = "some-unmeasured-model:cloud"
    assert SpeechPreferenceStore(settings).voice_model() == VOICE_MODELS[0]


def test_the_settings_route_reads_and_writes_the_choice(tmp_path) -> None:
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/settings/speech").json()["stt_provider"] == "cohere"
        response = client.put(
            "/api/v1/settings/speech",
            json={"stt_provider": "elevenlabs", "spoken_confirmation": True},
        )
        assert response.status_code == 200
        assert response.json()["stt_provider"] == "elevenlabs"
        assert client.get("/api/v1/settings/speech").json()["spoken_confirmation"]

        refused = client.put(
            "/api/v1/settings/speech",
            json={"stt_provider": "elevenlabs", "voice_model": "gpt-9:cloud"},
        )
        assert refused.status_code == 422
        # The refusal did not take: the stored choice is still the last good one.
        assert client.get("/api/v1/settings/speech").json()["voice_model"] in (
            VOICE_MODELS
        )


def test_a_provider_with_no_key_is_refused_by_the_route(tmp_path) -> None:
    settings = _settings(tmp_path, cohere_api_key="c")
    with TestClient(create_app(settings)) as client:
        response = client.put(
            "/api/v1/settings/speech", json={"stt_provider": "elevenlabs"}
        )
        assert response.status_code == 422
        assert "WAQIL_ELEVENLABS_API_KEY" in response.json()["detail"]


def test_dictation_reaches_the_selected_provider_untranscoded(tmp_path) -> None:
    """The whole point of the per-provider container rule.

    Chrome records WebM. On the Cohere path that has to be decoded to WAV on
    the host first; Scribe reads it as-is, so the clip arrives in the
    container the browser actually produced.
    """
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    app = create_app(settings)
    with TestClient(app) as client:
        eleven = FakeSpeechProvider("elevenlabs")
        app.state.runtime.model = FakeRouter(
            cohere=FakeSpeechProvider("cohere"), elevenlabs=eleven
        )
        client.put("/api/v1/settings/speech", json={"stt_provider": "elevenlabs"})

        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("dictation.webm", b"OggS" * 300, "audio/webm")},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "heard by elevenlabs"
        assert eleven.calls == [("dictation.webm", "audio/webm", 1200)]


def test_the_cohere_path_still_transcodes_what_cohere_refuses(
    tmp_path, monkeypatch
) -> None:
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    app = create_app(settings)
    converted: list[str] = []

    async def fake_to_wav(audio: bytes, filename: str, media_type: str, **kwargs):
        converted.append(filename)
        return b"RIFFconverted"

    monkeypatch.setattr("waqil_api.api.to_wav", fake_to_wav)
    with TestClient(app) as client:
        cohere = FakeSpeechProvider("cohere")
        app.state.runtime.model = FakeRouter(
            cohere=cohere, elevenlabs=FakeSpeechProvider("elevenlabs")
        )
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("dictation.webm", b"OggS" * 300, "audio/webm")},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "heard by cohere"
        assert converted == ["dictation.webm"]
        assert cohere.calls == [("dictation.wav", "audio/wav", len(b"RIFFconverted"))]


def test_each_provider_enforces_its_own_ceiling(tmp_path) -> None:
    settings = _settings(tmp_path, cohere_api_key="c", elevenlabs_api_key="e")
    settings.cohere_transcribe_max_bytes = 2048
    settings.elevenlabs_transcribe_max_bytes = 8192
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.runtime.model = FakeRouter(
            cohere=FakeSpeechProvider("cohere"),
            elevenlabs=FakeSpeechProvider("elevenlabs"),
        )
        clip = {"file": ("dictation.wav", b"RIFF" + b"x" * 4000, "audio/wav")}
        assert client.post("/api/v1/transcribe", files=clip).status_code == 413

        client.put("/api/v1/settings/speech", json={"stt_provider": "elevenlabs"})
        assert client.post("/api/v1/transcribe", files=clip).status_code == 200


def test_dictation_says_which_key_is_missing(tmp_path) -> None:
    settings = _settings(tmp_path, elevenlabs_api_key="e")
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.runtime.model = FakeRouter(
            cohere=FakeSpeechProvider("cohere", available=False),
            elevenlabs=FakeSpeechProvider("elevenlabs", available=False),
        )
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("dictation.wav", b"RIFF" * 300, "audio/wav")},
        )
        assert response.status_code == 503
        # No Cohere key, so the load-time degrade already chose ElevenLabs, and
        # the message names the key that is actually missing.
        assert "ELEVENLABS" in response.json()["detail"]
