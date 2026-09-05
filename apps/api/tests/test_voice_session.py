"""Sessions, leases, and what stops running when nobody is talking.

The lease is the load-bearing idea and most of these tests are about it. A
browser that crashes cannot tell anyone it crashed, so nothing that depends on
the browser being polite is allowed to be the thing that closes an outbound
tunnel. What closes it is a lease nobody renewed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import VoiceRenditionV1
from waqil_api.main import create_app
from waqil_api.speech_preference import SpeechPreferenceStore
from waqil_api.voice_session import (
    VoiceSessionExpired,
    VoiceSessionService,
    VoiceUnavailable,
)

READY = dict(
    elevenlabs_api_key="e",
    elevenlabs_agent_id="agent_1",
    voice_tunnel_hostname="voice.example.com",
    voice_tunnel_authorized=True,
    # The readiness check asks whether the connector binary is on PATH, and
    # this machine has no cloudflared. Any real executable satisfies the
    # question being asked; nothing here ever runs it.
    cloudflared_path="/bin/echo",
)


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


class FakeSpeech:
    available = True


class FakeRouter:
    def __init__(self) -> None:
        self.elevenlabs = FakeSpeech()


class FakeGraph:
    """Answers without reasoning, so these tests are about the lifecycle."""

    def __init__(self) -> None:
        self.turns: list = []
        self.forgotten: list[str] = []

    def forget(self, voice_session_id: str) -> None:
        self.forgotten.append(voice_session_id)

    async def answer(self, turn, *, on_spoken=None):
        self.turns.append(turn)
        if on_spoken is not None:
            await on_spoken("Three things are waiting.")
        return VoiceRenditionV1(
            written="Three things are waiting.",
            spoken="Three things are waiting.",
            intent="read",
            transcript=turn.transcript,
            voice_session_id=turn.voice_session_id,
            turn_id=turn.turn_id,
        )


class FakeProcess:
    """A child process that records whether it was asked to stop."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def _service(tmp_path, **overrides) -> tuple[VoiceSessionService, dict]:
    settings = _settings(tmp_path, **{**READY, **overrides})
    graph = FakeGraph()
    service = VoiceSessionService(
        settings,
        graph=graph,  # type: ignore[arg-type]
        speech_preference=SpeechPreferenceStore(settings),
        model=FakeRouter(),
        database=None,
    )
    started: dict = {"ingress": [], "connector": []}

    async def fake_ingress():
        process = FakeProcess()
        started["ingress"].append(process)
        return process

    async def fake_connector():
        process = FakeProcess()
        started["connector"].append(process)
        return process

    async def fake_ingress_ready():
        return None

    async def fake_token():
        return "conv-token-short-lived"

    service._spawn_ingress = fake_ingress  # type: ignore[assignment]
    service._spawn_connector = fake_connector  # type: ignore[assignment]
    service._wait_for_ingress = fake_ingress_ready  # type: ignore[assignment]
    service._conversation_token = fake_token  # type: ignore[assignment]
    return service, started


# -- readiness ---------------------------------------------------------------


def test_each_missing_piece_says_what_is_missing(tmp_path) -> None:
    def reason(**overrides) -> str:
        service, _ = _service(tmp_path, **overrides)
        return service.unavailable_reason()

    assert reason() == ""
    assert "WAQIL_ELEVENLABS_AGENT_ID" in reason(elevenlabs_agent_id="")
    assert "TUNNEL_HOSTNAME" in reason(voice_tunnel_hostname="")
    assert "switched off" in reason(voice_enabled=False)


def test_the_connector_will_not_start_without_an_explicit_go_ahead(tmp_path) -> None:
    """A.3, enforced rather than recorded.

    The brief's appendix authorizes a connector in principle and asks for one
    further go-ahead at the moment one is first started on this managed
    device. The flag is that go-ahead; without it voice refuses and says so.
    """
    service, started = _service(tmp_path, voice_tunnel_authorized=False)
    assert "explicit" in service.unavailable_reason()
    assert "WAQIL_VOICE_TUNNEL_AUTHORIZED" in service.unavailable_reason()


@pytest.mark.asyncio
async def test_starting_without_authorization_launches_nothing(tmp_path) -> None:
    service, started = _service(tmp_path, voice_tunnel_authorized=False)
    with pytest.raises(VoiceUnavailable, match="explicit"):
        await service.start()
    assert started == {"ingress": [], "connector": []}


# -- the session -------------------------------------------------------------


@pytest.mark.asyncio
async def test_starting_opens_both_processes_and_returns_the_url_once(
    tmp_path,
) -> None:
    service, started = _service(tmp_path)
    opened = await service.start()

    assert opened.conversation_token == "conv-token-short-lived"
    assert opened.session.state == "live"
    assert len(started["ingress"]) == 1 and len(started["connector"]) == 1

    # A status poll never hands the short-lived credential back: the contract
    # it returns has no field for it.
    status = await service.status(opened.session.id)
    assert not hasattr(status, "conversation_token")
    assert status.id == opened.session.id
    await service.shutdown()


@pytest.mark.asyncio
async def test_prewarm_fetches_once_and_start_consumes_the_prepared_token(
    tmp_path,
) -> None:
    service, started = _service(tmp_path)
    calls = 0

    async def counted_token():
        nonlocal calls
        calls += 1
        return f"prepared-{calls}"

    service._conversation_token = counted_token  # type: ignore[assignment]
    await service.prewarm()
    await service.prewarm()

    assert calls == 1
    assert len(started["ingress"]) == 1 and len(started["connector"]) == 1
    opened = await service.start()
    assert opened.conversation_token == "prepared-1"
    assert calls == 1
    await service.shutdown()


@pytest.mark.asyncio
async def test_an_unused_prewarm_expires_and_releases_both_processes(tmp_path) -> None:
    service, started = _service(tmp_path)
    await service.prewarm()
    service._prepared_until = datetime.now(UTC) - timedelta(seconds=1)

    await service.sweep()

    assert started["ingress"][0].terminated
    assert started["connector"][0].terminated
    assert service._prepared_token == ""


@pytest.mark.asyncio
async def test_the_api_key_never_reaches_the_browser(tmp_path) -> None:
    settings = _settings(tmp_path, **{**READY, "elevenlabs_api_key": "sk-canary-0001"})
    app = create_app(settings)
    with TestClient(app) as client:
        service = app.state.runtime.voice
        service._spawn_ingress = lambda: _resolved(FakeProcess())  # type: ignore
        service._spawn_connector = lambda: _resolved(FakeProcess())  # type: ignore
        service._wait_for_ingress = lambda: _resolved(None)  # type: ignore
        service._conversation_token = lambda: _resolved("conv-token")  # type: ignore
        service.model = FakeRouter()

        response = client.post("/api/v1/voice/sessions")
        assert response.status_code == 200
        assert "sk-canary-0001" not in response.text
        # And neither does the ingress bearer, which is not a user credential
        # but is still not the browser's business.
        assert service.shared_secret() not in response.text


def _resolved(value):
    async def resolve():
        return value

    return resolve()


# -- leases ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unrenewed_lease_ends_the_session_and_stops_the_connector(
    tmp_path,
) -> None:
    """The browser-crash path, which is the only one that really matters."""
    service, started = _service(tmp_path)
    opened = await service.start()
    connector = started["connector"][0]
    ingress = started["ingress"][0]

    # Nobody renews. Time passes.
    session = service._sessions[opened.session.id]
    session.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await service.sweep()

    assert session.state == "ended"
    assert "quiet" in session.reason
    assert connector.terminated and ingress.terminated


@pytest.mark.asyncio
async def test_renewing_keeps_the_session_and_the_processes(tmp_path) -> None:
    service, started = _service(tmp_path)
    opened = await service.start()
    service._sessions[opened.session.id].lease_expires_at = datetime.now(
        UTC
    ) - timedelta(seconds=1)

    renewed = await service.renew(opened.session.id)
    assert renewed.lease_expires_at > datetime.now(UTC)
    await service.sweep()
    assert service._sessions[opened.session.id].state == "live"
    assert not started["connector"][0].terminated
    await service.shutdown()


@pytest.mark.asyncio
async def test_two_tabs_do_not_stop_a_connector_the_other_still_needs(
    tmp_path,
) -> None:
    service, started = _service(tmp_path)
    first = await service.start()
    second = await service.start()
    # One connector serves both: the second session found it already running.
    assert len(started["connector"]) == 1

    await service.end(first.session.id)
    assert not started["connector"][0].terminated, "the second tab still needs it"

    await service.end(second.session.id)
    assert started["connector"][0].terminated
    assert started["ingress"][0].terminated


@pytest.mark.asyncio
async def test_a_session_past_its_ceiling_ends_on_the_next_renewal(tmp_path) -> None:
    service, started = _service(tmp_path, voice_session_max_seconds=60)
    opened = await service.start()
    session = service._sessions[opened.session.id]
    session.started_at = datetime.now(UTC) - timedelta(seconds=120)

    ended = await service.renew(opened.session.id)
    assert ended.state == "ended"
    assert "time limit" in ended.reason
    assert started["connector"][0].terminated


@pytest.mark.asyncio
async def test_shutdown_ends_every_session_and_leaves_nothing_running(
    tmp_path,
) -> None:
    service, started = _service(tmp_path)
    await service.start()
    await service.shutdown()
    assert started["connector"][0].terminated
    assert started["ingress"][0].terminated


# -- turns -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_turn_binds_its_provider_conversation_and_carries_identity(
    tmp_path,
) -> None:
    service, _ = _service(tmp_path)
    opened = await service.start()

    rendition = await service.turn(
        provider_conversation_id="conv_9",
        transcript="What's waiting on me?",
    )
    assert rendition.voice_session_id == opened.session.id
    assert rendition.turn_id
    assert service._sessions[opened.session.id].provider_conversation_id == "conv_9"

    # A second utterance on the same provider conversation finds the same session.
    again = await service.turn(
        provider_conversation_id="conv_9", transcript="And after that?"
    )
    assert again.voice_session_id == opened.session.id
    await service.shutdown()


@pytest.mark.asyncio
async def test_two_tabs_bind_to_the_explicit_metis_session_not_start_order(
    tmp_path,
) -> None:
    service, _ = _service(tmp_path)
    first = await service.start()
    second = await service.start()

    # The second provider call arrives first. Explicit identity keeps it on the
    # second tab instead of the oldest-unbound-session fallback.
    rendition = await service.turn(
        provider_conversation_id="conv_second",
        metis_session_id=second.session.id,
        transcript="What's waiting?",
    )
    assert rendition.voice_session_id == second.session.id
    assert not service._sessions[first.session.id].provider_conversation_id
    await service.shutdown()


@pytest.mark.asyncio
async def test_an_utterance_with_no_open_session_is_refused_not_answered(
    tmp_path,
) -> None:
    service, _ = _service(tmp_path)
    with pytest.raises(VoiceSessionExpired):
        await service.turn(
            provider_conversation_id="conv_unknown", transcript="What's waiting?"
        )


@pytest.mark.asyncio
async def test_the_browser_stream_carries_the_written_answer_and_the_handoff(
    tmp_path,
) -> None:
    """What the tunnel does not carry has to arrive some other way."""
    service, _ = _service(tmp_path)
    opened = await service.start()
    session = service._sessions[opened.session.id]

    import asyncio

    queue: asyncio.Queue = asyncio.Queue()
    session.listeners.append(queue)
    await service.turn(provider_conversation_id="c", transcript="What's waiting?")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event["type"] == "voice.turn"
    assert event["rendition"]["written"] == "Three things are waiting."
    await service.shutdown()


@pytest.mark.asyncio
async def test_the_ingress_stream_carries_each_sentence_then_a_close(
    tmp_path,
) -> None:
    """What crosses back through the tunnel: spoken sentences, and nothing else."""
    service, _ = _service(tmp_path)
    await service.start()
    frames = [
        frame
        async for frame in service.turn_stream(
            provider_conversation_id="c", transcript="What's waiting?"
        )
    ]
    assert frames == [
        {"type": "spoken", "text": "Three things are waiting."},
        {"type": "done"},
    ]
    await service.shutdown()


@pytest.mark.asyncio
async def test_a_stream_for_an_unknown_session_is_one_error_frame(tmp_path) -> None:
    service, _ = _service(tmp_path)
    frames = [
        frame
        async for frame in service.turn_stream(
            provider_conversation_id="conv_unknown", transcript="What's waiting?"
        )
    ]
    assert [frame["type"] for frame in frames] == ["error"]


@pytest.mark.asyncio
async def test_a_refused_build_publishes_the_verbatim_transcript_for_the_composer(
    tmp_path,
) -> None:
    import asyncio

    service, _ = _service(tmp_path)

    class RefusingGraph(FakeGraph):
        async def answer(self, turn, *, on_spoken=None):
            return VoiceRenditionV1(
                written="I can't build from voice",
                spoken="I can't build from voice",
                intent="refuse_build",
                transcript=turn.transcript,
                voice_session_id=turn.voice_session_id,
                turn_id=turn.turn_id,
            )

    service.graph = RefusingGraph()  # type: ignore[assignment]
    opened = await service.start()
    queue: asyncio.Queue = asyncio.Queue()
    service._sessions[opened.session.id].listeners.append(queue)

    said = "Build me a tool that summarises meeting notes"
    await service.turn(provider_conversation_id="c", transcript=said)

    events = [await asyncio.wait_for(queue.get(), timeout=1) for _ in range(2)]
    deferred = next(item for item in events if item["type"] == "voice.build_deferred")
    # Verbatim, so the composer is prefilled with what was said rather than a
    # paraphrase of it.
    assert deferred["transcript"] == said
    assert deferred["hand_off"] is True
    assert deferred["voice_session_id"] == opened.session.id
    await service.shutdown()


# -- the loopback contract ---------------------------------------------------


def test_the_turn_route_refuses_a_caller_without_the_ingress_bearer(tmp_path) -> None:
    settings = _settings(tmp_path, **READY)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.runtime.voice.model = FakeRouter()
        body = {"provider_conversation_id": "c", "transcript": "hello"}
        assert client.post("/api/v1/voice/turn", json=body).status_code == 401
        assert (
            client.post(
                "/api/v1/voice/turn",
                json=body,
                headers={"authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            client.post("/api/v1/voice/post-call", json={"payload": {}}).status_code
            == 401
        )


def test_availability_explains_itself_without_leaking_configuration(
    tmp_path,
) -> None:
    app = create_app(_settings(tmp_path, elevenlabs_api_key="sk-canary-0002"))
    with TestClient(app) as client:
        app.state.runtime.voice.model = FakeRouter()
        body = client.get("/api/v1/voice").json()
        assert body["available"] is False
        assert "WAQIL_ELEVENLABS_AGENT_ID" in body["reason"]
        assert "sk-canary-0002" not in body["reason"]


@pytest.mark.asyncio
async def test_a_post_call_is_stored_as_evidence_and_creates_no_records(
    tmp_path,
) -> None:
    """Everything a verified webhook leaves behind, and everything it does not.

    The provider's own analysis is stored under a column that says whose
    opinion it is. Nothing here becomes a customer fact, an action, a memory
    or an account link — those stay proposals a person accepts.
    """
    from waqil_api.database import Database

    settings = _settings(tmp_path, **READY)
    database = Database(settings.database_path)
    await database.open()
    try:
        service, _ = _service(tmp_path)
        service.database = database
        payload = {
            "event_type": "post_call_transcription",
            "provider_conversation_id": "conv_77",
            "body_sha256": "a" * 64,
            "payload": {
                "type": "post_call_transcription",
                "data": {
                    "conversation_id": "conv_77",
                    "transcript": [{"role": "user", "message": "the workshop moved"}],
                    "analysis": {
                        "call_successful": "success",
                        "data_collection_results": {"account": "Batelco"},
                    },
                },
            },
        }
        first = await service.post_call(payload)
        assert first["status"] == "stored"
        assert first["record_id"]

        # ElevenLabs retries; a retry is the same conversation, not a second one.
        second = await service.post_call(payload)
        assert second["record_id"] == first["record_id"]

        stored = await database.list_voice_post_calls()
        assert len(stored) == 1
        row = stored[0]
        assert row["provider_conversation_id"] == "conv_77"
        assert row["status"] == "stored"
        assert "workshop moved" in row["transcript_json"]
        # Named as the provider's, so nothing downstream reads it as a decision.
        assert "provider_analysis_json" in row
        assert "Batelco" in row["provider_analysis_json"]

        # And the thing it must never have done: no customer record appeared.
        accounts = await database.list_customer_accounts()
        assert accounts == []
    finally:
        await database.close()


def test_the_locally_minted_secret_is_stable_and_kept_to_the_owner(tmp_path) -> None:
    service, _ = _service(tmp_path)
    secret = service.shared_secret()
    assert len(secret) >= 32
    # Stable across reads, and across a fresh service on the same data dir.
    assert service.shared_secret() == secret
    again, _ = _service(tmp_path)
    assert again.shared_secret() == secret

    path = service.settings.voice_secret_path
    assert path.is_file()
    assert oct(path.stat().st_mode)[-3:] == "600"
