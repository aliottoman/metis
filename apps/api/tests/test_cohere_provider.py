"""The Cohere transport: Command A through the same tool-calling seam.

Cohere is the fourth provider and deliberately brings no fourth protocol: the
project roster, the create_file narrowing, and the function-call conversion
are the shared ones, so these tests pin the adapter's own surface — reply
parsing (thinking blocks excluded), the structured function-call decode with
its bounded repair, routing by the run's aliases, and the preference gates.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api.contracts import ModelRequestV1, PROJECT_TOOL_REQUIRED_ARGUMENTS
from waqil_api.model_preference import ModelPreferenceStore
from waqil_api.model_provider import (
    CohereModelProvider,
    ModelProviderError,
    RoutedModelProvider,
)
from waqil_api.project_tools import FINISH_TOOL_NAME


class ScriptedCohere(CohereModelProvider):
    def __init__(self, settings: Settings, *replies: dict) -> None:
        super().__init__(settings)
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def _chat(self, payload: dict) -> dict:
        self.requests.append(payload)
        return self.replies.pop(0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        cohere_api_key="test-key",
    )


def _tool_reply(name: str, arguments) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "planning the call"}],
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}
            ],
        }
    }


@pytest.mark.asyncio
async def test_a_project_step_sends_the_shared_roster_and_converts(tmp_path) -> None:
    provider = ScriptedCohere(
        _settings(tmp_path),
        _tool_reply(
            "create_file", json.dumps({"path": "app/main.py", "content": "x\n"})
        ),
    )
    step = await provider.project_step(
        {"build_turn": True, "files_still_to_write": ["app/main.py", ".env.example"]}
    )
    assert step.status == "tool"
    assert step.tool_call is not None and step.tool_call.name == "create_file"
    sent = provider.requests[0]
    names = [tool["function"]["name"] for tool in sent["tools"]]
    assert set(names) == set(PROJECT_TOOL_REQUIRED_ARGUMENTS) | {FINISH_TOOL_NAME}
    create = next(
        tool["function"]
        for tool in sent["tools"]
        if tool["function"]["name"] == "create_file"
    )
    assert create["parameters"]["properties"]["path"]["enum"] == [
        "app/main.py",
        ".env.example",
    ]


@pytest.mark.asyncio
async def test_an_unknown_tool_is_refused_and_prose_becomes_a_completion(
    tmp_path,
) -> None:
    provider = ScriptedCohere(_settings(tmp_path), _tool_reply("rm_rf", "{}"))
    with pytest.raises(ModelProviderError, match="unsupported project tool"):
        await provider.project_step({})

    prose = ScriptedCohere(
        _settings(tmp_path),
        {
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "All the files are in place."},
                ]
            }
        },
    )
    step = await prose.project_step({})
    # Judged by the loop's provider-independent premature-finish guard, like OCI.
    assert step.status == "complete"
    assert step.response == "All the files are in place."
    assert "hidden" not in step.response


@pytest.mark.asyncio
async def test_structured_decode_rides_one_function_with_a_bounded_repair(
    tmp_path,
) -> None:
    provider = ScriptedCohere(
        _settings(tmp_path),
        _tool_reply("return_projectbuildplanv1", {"files": 12}),  # wrong type → repair
        _tool_reply("return_projectbuildplanv1", {"files": ["app/main.py"]}),
    )
    plan = await provider.project_plan_files({})
    assert plan.files == ["app/main.py"]
    assert len(provider.requests) == 2
    assert "failed validation" in provider.requests[1]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_a_json_answer_in_text_is_judged_on_its_merits(tmp_path) -> None:
    provider = ScriptedCohere(
        _settings(tmp_path),
        {"message": {"content": [{"type": "text", "text": '{"files": ["a.py"]}'}]}},
    )
    assert (await provider.project_plan_files({})).files == ["a.py"]


@pytest.mark.asyncio
async def test_generate_returns_text_and_never_the_thinking_channel(tmp_path) -> None:
    provider = ScriptedCohere(
        _settings(tmp_path),
        {
            "id": "resp_1",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": "The answer."},
                ]
            },
        },
    )
    from waqil_api.contracts import ModelRequestV1

    result = await provider.generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert result.content == "The answer."
    assert "internal" not in result.content


def _frame(kind: str, data: dict) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()


async def _collect(sink: list[str], delta: str) -> None:
    sink.append(delta)


@pytest.mark.asyncio
async def test_generate_streams_text_and_thinking_as_events_arrive(tmp_path) -> None:
    emitted: list[str] = []
    thinking: list[str] = []
    requests: list[dict] = []

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _frame("message-start", {"id": "resp_stream"})
            yield _frame("content-delta", {"delta": {"message": {"content": {"thinking": "Checking"}}}})
            assert thinking == ["Checking"] and emitted == []
            yield _frame("content-delta", {"delta": {"message": {"content": {"text": "Hel"}}}})
            assert emitted == ["Hel"]
            yield _frame("content-delta", {"delta": {"message": {"content": {"text": "lo <co"}}}})
            assert emitted == ["Hel", "lo "]
            yield _frame("content-delta", {"delta": {"message": {"content": {"text": ">world</co: 0:[0]>!"}}}})
            assert emitted == ["Hel", "lo ", "world!"]
            yield _frame("message-end", {"delta": {"finish_reason": "COMPLETE"}})

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Chunks())

    client = httpx.AsyncClient(
        base_url="https://example.test", transport=httpx.MockTransport(respond)
    )
    provider = CohereModelProvider(_settings(tmp_path))
    provider._client_instance = client
    try:
        result = await provider.generate(
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
            on_token=lambda delta: _collect(emitted, delta),
            on_reasoning=lambda delta: _collect(thinking, delta),
        )
    finally:
        await client.aclose()
    assert result.content == "Hello world!"
    assert result.structured["response_id"] == "resp_stream"
    assert emitted == ["Hel", "lo ", "world!"]
    assert thinking == ["Checking"]
    assert requests[0]["stream"] is True


@pytest.mark.asyncio
async def test_stream_rejects_truncated_and_limited_cohere_replies(tmp_path) -> None:
    async def run(frames: bytes) -> tuple[str, list[str]]:
        emitted: list[str] = []
        client = httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=frames
                )
            ),
        )
        provider = CohereModelProvider(_settings(tmp_path))
        provider._client_instance = client
        try:
            with pytest.raises(ModelProviderError) as failure:
                await provider.generate(
                    ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                    on_token=lambda delta: _collect(emitted, delta),
                )
            return str(failure.value), emitted
        finally:
            await client.aclose()

    partial = _frame("content-delta", {"delta": {"message": {"content": {"text": "partial"}}}})
    truncated, emitted = await run(partial)
    assert "before completion" in truncated and emitted == ["partial"]
    limited, emitted = await run(partial + _frame("message-end", {"delta": {"finish_reason": "MAX_TOKENS"}}))
    assert "MAX_TOKENS" in limited and emitted == ["partial"]
    failed, emitted = await run(partial + _frame("error", {"message": "generation failed"}))
    assert "generation failed" in failed and emitted == ["partial"]


@pytest.mark.asyncio
async def test_stream_preserves_cohere_callback_cancellation(tmp_path) -> None:
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_frame("content-delta", {"delta": {"message": {"content": {"text": "first"}}}}),
            )
        ),
    )
    provider = CohereModelProvider(_settings(tmp_path))
    provider._client_instance = client

    async def cancel(_: str) -> None:
        raise asyncio.CancelledError()

    try:
        with pytest.raises(asyncio.CancelledError):
            await provider.generate(
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=cancel,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_the_router_selects_cohere_only_by_the_runs_aliases(tmp_path) -> None:
    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(
            self, request, on_token=None, *, model_aliases=None, on_reasoning=None
        ):
            return self.name

    routed = RoutedModelProvider(Named("local"), Named("oci"), cohere=Named("cohere"))  # type: ignore[arg-type]
    assert (
        await routed.generate(None, model_aliases={"_provider": "cohere"}) == "cohere"
    )
    assert await routed.generate(None, model_aliases={"_provider": "oci"}) == "oci"
    assert await routed.generate(None, model_aliases={}) == "local"
    # Without a Cohere instance the alias degrades to local instead of crashing.
    bare = RoutedModelProvider(Named("local"), Named("oci"))  # type: ignore[arg-type]
    assert await bare.generate(None, model_aliases={"_provider": "cohere"}) == "local"


def test_cohere_continuous_pins_its_own_provider_and_is_gated_on_the_key(
    tmp_path,
) -> None:
    """The third project mode: Command A+ leads every bounded step.

    Two things have to hold together — the mode names the provider the run
    will use, and it is refused where it is chosen when that provider has no
    key. Refusing later means a build that starts and cannot make one call.
    """
    from fastapi import HTTPException

    from waqil_api.api import PROJECT_MODE_PROVIDER, _require_project_mode_available

    assert PROJECT_MODE_PROVIDER["cohere_continuous"] == "cohere"
    assert PROJECT_MODE_PROVIDER["grok_continuous"] == "oci"
    # Local-led mode names no provider, so it is never gated on a cloud key.
    assert "grok_bootstrap_local" not in PROJECT_MODE_PROVIDER

    class App:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings
            self.model_preference = ModelPreferenceStore(settings)

    keyed = App(_settings(tmp_path))
    _require_project_mode_available(keyed, "cohere_continuous")  # no raise

    unkeyed = App(Settings(_env_file=None, data_dir=tmp_path / "b"))
    with pytest.raises(HTTPException, match="WAQIL_COHERE_API_KEY") as refusal:
        _require_project_mode_available(unkeyed, "cohere_continuous")
    assert refusal.value.status_code == 409
    # The local-led mode still opens without any cloud key at all.
    _require_project_mode_available(unkeyed, "grok_bootstrap_local")


@pytest.mark.asyncio
async def test_a_cohere_project_session_survives_a_round_trip(tmp_path) -> None:
    """The mode is guarded by a SQLite CHECK, so storing it is its own risk."""
    from waqil_api.database import Database

    database = Database(tmp_path / "modes.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Command A+ build")
        stored = await database.set_conversation_project(
            conversation.id, "asset_" + "a" * 20, "cohere_continuous"
        )
        assert stored.mode == "cohere_continuous"
        assert (await database.get_conversation_project(conversation.id)).mode == (
            "cohere_continuous"
        )
        with pytest.raises(ValueError, match="unsupported project mode"):
            await database.set_conversation_project(
                conversation.id, "asset_" + "a" * 20, "cohere_once"
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_transcribe_posts_multipart_and_returns_only_the_text(tmp_path) -> None:
    """The one call on this provider that is not chat, so it has its own seam."""
    sent: dict = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {"text": "  book the sizing review  "}

    class FakeClient:
        async def post(self, url, *, data=None, files=None, **kwargs):
            sent.update(url=url, data=data, files=files)
            return FakeResponse()

    provider = CohereModelProvider(_settings(tmp_path))
    provider._client_instance = FakeClient()

    assert await provider.transcribe(b"RIFF...", "clip.wav", "audio/wav") == (
        "book the sizing review"
    )
    assert sent["url"] == "/v2/audio/transcriptions"
    # Multipart, not JSON: a JSON body here is a 4xx from Cohere, and the
    # client's own headers must not be carrying a Content-Type that would
    # displace the boundary.
    assert sent["files"] == {"file": ("clip.wav", b"RIFF...", "audio/wav")}
    assert sent["data"]["model"] == "cohere-transcribe-03-2026"
    assert sent["data"]["language"] == "en"


@pytest.mark.asyncio
async def test_transcribe_refuses_empty_and_oversized_audio(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.cohere_transcribe_max_bytes = 2048
    provider = CohereModelProvider(settings)

    with pytest.raises(ModelProviderError, match="No audio"):
        await provider.transcribe(b"", "clip.wav", "audio/wav")
    # Refused before the client is ever built, so an oversized clip costs no
    # round trip — note there is no _client_instance set on this provider.
    with pytest.raises(ModelProviderError, match="past the"):
        await provider.transcribe(b"x" * 2049, "clip.wav", "audio/wav")


def test_the_cohere_client_leaves_content_type_to_each_request(tmp_path) -> None:
    """A client-level Content-Type wins httpx's merge and breaks multipart.

    This is the whole reason the audio upload works, and it is invisible at
    the call site, so it is pinned here rather than left to a live 400.
    """
    import asyncio

    provider = CohereModelProvider(_settings(tmp_path))
    client = asyncio.run(provider._client())
    try:
        assert "authorization" in client.headers
        assert "content-type" not in client.headers
    finally:
        asyncio.run(provider.close())


def test_the_preference_gates_cohere_on_the_key(tmp_path) -> None:
    keyed = ModelPreferenceStore(_settings(tmp_path))
    assert keyed.cohere_available
    saved = keyed.save("split", None, provider="cohere")
    assert saved.provider == "cohere" and saved.cohere_available

    unkeyed = ModelPreferenceStore(Settings(_env_file=None, data_dir=tmp_path / "b"))
    with pytest.raises(ValueError, match="WAQIL_COHERE_API_KEY"):
        unkeyed.save("split", None, provider="cohere")
    # A stored cohere preference degrades to local when the key disappears.
    unkeyed._settings.model_preference_path.parent.mkdir(parents=True, exist_ok=True)
    unkeyed._settings.model_preference_path.write_text(
        json.dumps({"mode": "split", "model": None, "provider": "cohere"}),
        encoding="utf-8",
    )
    assert unkeyed.load().provider == "local"


def test_citation_markup_never_reaches_content() -> None:
    """Command A+ emits its own grounded-generation markup inside structured
    output. Left in, `<co>…</co: 0:[0]>` prints verbatim on a rendered slide."""
    from waqil_api.model_provider import _cohere_message_text

    message = {"content": [{"type": "text", "text": "<co>hello</co: 1:[2]> there"}]}
    assert _cohere_message_text(message) == "hello there"


def test_citation_markup_is_cleaned_after_json_decoding() -> None:
    """Cohere escapes the markup on the wire (\\u003cco\\u003e), so a strip that
    runs on the raw arguments matches nothing — it has to run on the decoded
    payload, or the markup prints verbatim in a generated document."""
    import json

    from waqil_api.model_provider import _clean_cohere_payload, _strip_cohere_citations

    wire = (
        '{"body": "gives \\u003cco\\u003efull isolation\\u003c/co: 0:[0]\\u003e today"}'
    )
    assert "<co>" not in wire  # escaped, so a pre-parse strip cannot see it
    assert _strip_cohere_citations(wire) == wire
    cleaned = _clean_cohere_payload(json.loads(wire))
    assert cleaned == {"body": "gives full isolation today"}


def test_payload_cleaning_reaches_nested_lists_and_dicts() -> None:
    from waqil_api.model_provider import _clean_cohere_payload

    payload = {
        "sections": [{"bullets": ["<co>one</co: 0:[0]>", "two"], "n": 3}],
        "keep": None,
    }
    assert _clean_cohere_payload(payload) == {
        "sections": [{"bullets": ["one", "two"], "n": 3}],
        "keep": None,
    }


def test_thinking_blocks_are_separated_from_the_answer() -> None:
    """Command A+ thinks by default and bills the tokens either way; the
    reasoning belongs on its own channel, never concatenated into the answer."""
    from waqil_api.model_provider import _cohere_message_text, _cohere_thinking_text

    message = {
        "content": [
            {"type": "thinking", "thinking": "let me compute 17*23"},
            {"type": "text", "text": "391"},
        ]
    }
    assert _cohere_message_text(message) == "391"
    assert _cohere_thinking_text(message) == "let me compute 17*23"


class _Reply:
    """The parts of an httpx response the retry path reads."""

    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._body


class _CountingClient:
    def __init__(self, *replies: _Reply) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def post(self, path: str, json: dict) -> _Reply:  # noqa: A002 - httpx's name
        self.calls += 1
        return self.replies.pop(0)


def _provider_with(settings: Settings, client: _CountingClient) -> CohereModelProvider:
    provider = CohereModelProvider(settings)
    provider._client_instance = client  # type: ignore[assignment]
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        _Reply(500, {"message": "internal server error"}),
        _Reply(503, {"message": "unavailable"}),
        _Reply(422, {"error_type": "NO_VALID_RESPONSE_GENERATED"}),
        _Reply(422, {"error_type": "INVALID_TOOL_GENERATION"}),
        _Reply(422, {"error_type": "HALLUCINATED_ALL_TOOL_CALLS"}),
    ],
)
async def test_a_transient_service_failure_is_retried_not_surfaced(
    tmp_path, failure: _Reply
) -> None:
    """One live battery run lost a build turn, a web-research turn and a deck
    turn to three of these, each on the first attempt, with nothing wrong on
    this side. They are the service failing to generate, not a bad request."""
    client = _CountingClient(failure, _Reply(200, {"message": {"content": []}}))
    provider = _provider_with(_settings(tmp_path), client)
    assert await provider._chat({"messages": []}) == {"message": {"content": []}}
    assert client.calls == 2


@pytest.mark.asyncio
async def test_a_request_this_side_got_wrong_fails_at_once(tmp_path) -> None:
    client = _CountingClient(_Reply(400, {"message": "invalid request"}))
    provider = _provider_with(_settings(tmp_path), client)
    with pytest.raises(ModelProviderError, match="HTTP 400"):
        await provider._chat({"messages": []})
    assert client.calls == 1


@pytest.mark.asyncio
async def test_a_transient_failure_that_never_clears_surfaces_its_real_status(
    tmp_path,
) -> None:
    client = _CountingClient(*(_Reply(500, {"message": "down"}) for _ in range(3)))
    provider = _provider_with(_settings(tmp_path), client)
    with pytest.raises(ModelProviderError, match="HTTP 500"):
        await provider._chat({"messages": []})
    assert client.calls == 3
