"""The ElevenLabs transport: ears and mouth, and nothing that reasons.

Mirrors the Cohere provider's own seams — the multipart upload, the ceiling
enforced before a round trip is spent, the client whose headers must not
displace a multipart boundary — and adds the one thing that is unique here:
this provider is carried by the router but can never be selected by it, so no
run can be routed to a transport with no answer in it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.model_provider import (
    ElevenLabsSpeechProvider,
    ModelProviderError,
    OCIResponsesModelProvider,
    RoutedModelProvider,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        elevenlabs_api_key="test-key",
    )


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict | None = None,
        content: bytes = b"",
        headers: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.sent: dict = {}

    async def post(self, url, **kwargs):
        self.sent = {"url": url, **kwargs}
        return self.response


@pytest.mark.asyncio
async def test_transcribe_posts_multipart_and_returns_only_the_text(tmp_path) -> None:
    provider = ElevenLabsSpeechProvider(_settings(tmp_path))
    client = FakeClient(FakeResponse(payload={"text": "  move the workshop  "}))
    provider._client_instance = client

    assert await provider.transcribe(b"OggS...", "clip.webm", "audio/webm") == (
        "move the workshop"
    )
    assert client.sent["url"] == "/v1/speech-to-text"
    # Multipart, not JSON — the same constraint the Cohere upload lives under.
    assert client.sent["files"] == {"file": ("clip.webm", b"OggS...", "audio/webm")}
    assert client.sent["data"]["model_id"] == "scribe_v1"
    # English is stated rather than detected: every clip in this build is known
    # to be English, and a stated code is a better answer than a guess.
    assert client.sent["data"]["language_code"] == "en"


@pytest.mark.asyncio
async def test_transcribe_refuses_empty_and_oversized_audio(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.elevenlabs_transcribe_max_bytes = 2048
    provider = ElevenLabsSpeechProvider(settings)

    with pytest.raises(ModelProviderError, match="No audio"):
        await provider.transcribe(b"", "clip.webm", "audio/webm")
    # No client is ever built here: an oversized clip costs no round trip.
    with pytest.raises(ModelProviderError, match="past the"):
        await provider.transcribe(b"x" * 2049, "clip.webm", "audio/webm")


@pytest.mark.asyncio
async def test_transcribe_reports_a_failed_call_rather_than_guessing(tmp_path) -> None:
    provider = ElevenLabsSpeechProvider(_settings(tmp_path))
    provider._client_instance = FakeClient(
        FakeResponse(status_code=422, text="unsupported model_id")
    )
    with pytest.raises(ModelProviderError, match="HTTP 422"):
        await provider.transcribe(b"OggS...", "clip.webm", "audio/webm")

    provider._client_instance = FakeClient(FakeResponse(payload={"words": []}))
    with pytest.raises(ModelProviderError, match="no transcript"):
        await provider.transcribe(b"OggS...", "clip.webm", "audio/webm")


@pytest.mark.asyncio
async def test_synthesize_returns_audio_bytes_and_the_voice_it_used(tmp_path) -> None:
    provider = ElevenLabsSpeechProvider(_settings(tmp_path))
    client = FakeClient(
        FakeResponse(content=b"ID3audio", headers={"content-type": "audio/mpeg"})
    )
    provider._client_instance = client

    audio, media_type = await provider.synthesize("Three things are waiting.")
    assert audio == b"ID3audio"
    assert media_type == "audio/mpeg"
    # The stock voice from the brief's recorded decision, on the path itself.
    assert client.sent["url"] == "/v1/text-to-speech/r1KmysJdVYZjJCm4mL3b"
    assert client.sent["json"]["model_id"] == "eleven_flash_v2_5"
    assert client.sent["json"]["text"] == "Three things are waiting."


@pytest.mark.asyncio
async def test_synthesize_refuses_silence_and_a_missing_voice(tmp_path) -> None:
    settings = _settings(tmp_path)
    provider = ElevenLabsSpeechProvider(settings)
    with pytest.raises(ModelProviderError, match="nothing to say"):
        await provider.synthesize("   ")

    settings.elevenlabs_voice_id = ""
    with pytest.raises(ModelProviderError, match="VOICE_ID"):
        await provider.synthesize("anything at all")


def test_the_client_leaves_content_type_to_each_request(tmp_path) -> None:
    """A client-level Content-Type wins httpx's merge and breaks multipart."""
    provider = ElevenLabsSpeechProvider(_settings(tmp_path))
    client = asyncio.run(provider._client())
    try:
        assert "xi-api-key" in client.headers
        assert "content-type" not in client.headers
    finally:
        asyncio.run(provider.close())


def test_an_absent_key_makes_the_provider_unavailable(tmp_path) -> None:
    provider = ElevenLabsSpeechProvider(Settings(_env_file=None, data_dir=tmp_path))
    assert provider.available is False
    with pytest.raises(ModelProviderError, match="WAQIL_ELEVENLABS_API_KEY"):
        asyncio.run(provider._client())


def test_the_router_carries_the_speech_provider_but_cannot_select_it(
    tmp_path,
) -> None:
    """The invariant that keeps a speechless transport out of reasoning.

    `_selected` resolves whichever provider a run's aliases name. ElevenLabs
    has no `generate`, so a run routed here would have nothing to answer with;
    the guarantee is that no alias — including one that spells its own name —
    can reach it.
    """
    settings = _settings(tmp_path)
    speech = ElevenLabsSpeechProvider(settings)
    router = RoutedModelProvider(
        local=object(),  # type: ignore[arg-type]
        oci=OCIResponsesModelProvider(settings),
        elevenlabs=speech,
    )
    assert router.elevenlabs is speech
    for provider in ("elevenlabs", "cohere", "cline", "oci", "local", ""):
        assert router._selected({"_provider": provider}) is not speech
    assert router._selected(None) is not speech
