"""The Cohere transport: Command A through the same tool-calling seam.

Cohere is the fourth provider and deliberately brings no fourth protocol: the
project roster, the create_file narrowing, and the function-call conversion
are the shared ones, so these tests pin the adapter's own surface — reply
parsing (thinking blocks excluded), the structured function-call decode with
its bounded repair, routing by the run's aliases, and the preference gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS
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
        _tool_reply("create_file", json.dumps({"path": "app/main.py", "content": "x\n"})),
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
        tool["function"] for tool in sent["tools"] if tool["function"]["name"] == "create_file"
    )
    assert create["parameters"]["properties"]["path"]["enum"] == [
        "app/main.py",
        ".env.example",
    ]


@pytest.mark.asyncio
async def test_an_unknown_tool_is_refused_and_prose_becomes_a_completion(tmp_path) -> None:
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
async def test_structured_decode_rides_one_function_with_a_bounded_repair(tmp_path) -> None:
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


@pytest.mark.asyncio
async def test_the_router_selects_cohere_only_by_the_runs_aliases(tmp_path) -> None:
    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
            return self.name

    routed = RoutedModelProvider(Named("local"), Named("oci"), cohere=Named("cohere"))  # type: ignore[arg-type]
    assert await routed.generate(None, model_aliases={"_provider": "cohere"}) == "cohere"
    assert await routed.generate(None, model_aliases={"_provider": "oci"}) == "oci"
    assert await routed.generate(None, model_aliases={}) == "local"
    # Without a Cohere instance the alias degrades to local instead of crashing.
    bare = RoutedModelProvider(Named("local"), Named("oci"))  # type: ignore[arg-type]
    assert await bare.generate(None, model_aliases={"_provider": "cohere"}) == "local"


def test_cohere_continuous_pins_its_own_provider_and_is_gated_on_the_key(tmp_path) -> None:
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
