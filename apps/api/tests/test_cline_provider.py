"""The Cline lane: one key, two seats, and three quirks verified live.

Each quirk below was found by calling the real endpoint, not by reading
documentation, and each would have produced a failure that looked like a bad
model rather than a bad client.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from waqil_api.config import Settings
from waqil_api.contracts import (
    AssetRecipeV1,
    ModelRequestV1,
    PlanningRequestV1,
    ProjectBuildPlanV1,
    ProjectDirectionV1,
)
from waqil_api.model_provider import (
    ClineModelProvider,
    ModelProviderError,
    RoutedModelProvider,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        cline_api_key="test-key",
        allow_test_backends=True,
        **overrides,
    )


class _Response:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if not isinstance(body, str) else body
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class _Client:
    """A stand-in httpx client that records what the provider actually sent."""

    def __init__(self, *responses: _Response) -> None:
        self._responses = list(responses)
        self.sent: list[dict[str, Any]] = []

    async def post(self, path: str, json: dict[str, Any]) -> _Response:  # noqa: A002
        self.sent.append({"path": path, **json})
        return self._responses.pop(0) if self._responses else _Response(200, {})


def _provider(client: _Client, **overrides: Any) -> ClineModelProvider:
    provider = ClineModelProvider(_settings(**overrides))

    async def _client() -> _Client:
        return client

    provider._client = _client  # type: ignore[assignment]
    return provider


def _tool_reply(name: str, arguments: Any) -> _Response:
    """A reply in the gateway's own envelope: wrapped in `data`."""
    return _Response(
        200,
        {
            "data": {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": name, "arguments": arguments}}
                            ],
                        }
                    }
                ]
            }
        },
    )


# ── The three quirks ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_reply_is_unwrapped_from_its_data_envelope() -> None:
    """The gateway wraps what it proxies, so an OpenAI-shaped client reading
    `choices` off the top level finds nothing and reports an empty model."""
    client = _Client(
        _Response(200, {"data": {"choices": [{"message": {"content": "hello"}}]}})
    )
    result = await _provider(client).generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert result.content == "hello"
    assert client.sent[0]["stream"] is False
    assert result.structured is not None
    assert result.structured["provider"] == "cline"
    assert result.structured["streamed"] is False
    assert result.structured["first_text_seconds"] is None
    assert result.structured["generation_seconds"] >= 0
    assert result.structured["input_characters"] > len("u")
    assert result.structured["output_characters"] == len("hello")
    assert "response_headers_seconds" not in result.structured


@pytest.mark.asyncio
async def test_general_chat_prompt_allows_stable_knowledge_but_bounds_personal_and_current_claims() -> None:
    chat = _Client(
        _Response(200, {"data": {"choices": [{"message": {"content": "answer"}}]}})
    )
    await _provider(chat).generate(
        ModelRequestV1(
            role="planner",
            system_prompt="Be concise.",
            user_prompt="Explain a stable concept.",
        )
    )
    system = chat.sent[0]["messages"][0]["content"]
    assert "general, stable knowledge directly" in system
    assert "Ground claims about the user's" in system
    assert "Ground current or changing claims" in system
    assert "retrieved web evidence" in system
    assert "Never claim to have browsed" in system
    assert system.endswith("Be concise.")
    assert "Answer only from the bounded context" not in system

    project = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '{"path":"app/main.py","instruction":"write it"}',
        )
    )
    await _provider(project).project_direction({"planned_files": ["app/main.py"]})
    assert "Answer only from the bounded context" in project.sent[0]["messages"][0][
        "content"
    ]


@pytest.mark.asyncio
async def test_chat_stream_emits_each_delta_before_the_reply_finishes() -> None:
    emitted: list[str] = []
    requests: list[dict[str, Any]] = []

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"data":{"choices":[{"delta":{"content":"Hel"}}]}}\n\n'
            assert emitted == ["Hel"]
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            assert emitted == ["Hel", "lo"]
            yield b'data: {"data":{"choices":[{"finish_reason":"stop"}]}}\n\n'
            yield b'data: {"data":{"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}}\n\n'
            yield b"data: [DONE]\n\n"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunks()
        )

    client = httpx.AsyncClient(
        base_url="https://example.test", transport=httpx.MockTransport(respond)
    )
    provider = _provider(client)  # type: ignore[arg-type]
    try:
        result = await provider.generate(
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
            on_token=lambda delta: _collect(emitted, delta),
        )
    finally:
        await client.aclose()
    assert result.content == "Hello"
    assert emitted == ["Hel", "lo"]
    assert requests[0]["stream"] is True
    assert result.structured is not None
    timing = result.structured
    assert timing["provider"] == "cline"
    assert timing["streamed"] is True
    assert timing["output_characters"] == 5
    assert 0 <= timing["response_headers_seconds"] <= timing["first_event_seconds"]
    assert timing["first_event_seconds"] <= timing["first_text_seconds"]
    assert timing["first_text_seconds"] <= timing["generation_seconds"]
    assert provider.last_usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


async def _collect(sink: list[str], delta: str) -> None:
    sink.append(delta)


@pytest.mark.asyncio
@pytest.mark.parametrize("done_marker", [True, False])
async def test_cline_reasoning_is_batched_separately_from_answer_text(
    done_marker: bool,
) -> None:
    answer: list[str] = []
    reasoning: list[str] = []
    first = "a" * 90
    second = "b" * 90
    trailing = "c" * 30
    after_stop = "d" * 20

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield f'data: {{"choices":[{{"delta":{{"reasoning":"{first}"}}}}]}}\n\n'.encode()
            assert reasoning == []
            yield f'data: {{"choices":[{{"delta":{{"reasoning_content":"{second}"}}}}]}}\n\n'.encode()
            assert reasoning == [first + second]
            assert answer == []
            yield b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
            assert answer == ["Answer"]
            yield f'data: {{"choices":[{{"delta":{{"reasoning":"{trailing}"}}}}]}}\n\n'.encode()
            assert reasoning == [first + second]
            yield b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
            assert reasoning == [first + second, trailing]
            yield f'data: {{"choices":[{{"delta":{{"reasoning":"{after_stop}"}}}}]}}\n\n'.encode()
            assert reasoning == [first + second, trailing]
            if done_marker:
                yield b"data: [DONE]\n\n"

    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Chunks()
            )
        ),
    )
    try:
        result = await _provider(client).generate(  # type: ignore[arg-type]
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
            on_token=lambda delta: _collect(answer, delta),
            on_reasoning=lambda delta: _collect(reasoning, delta),
        )
    finally:
        await client.aclose()
    assert result.content == "Answer"
    assert reasoning == [first + second, trailing, after_stop]
    assert result.structured is not None
    assert result.structured["reasoning_characters"] == 230
    assert result.structured["first_reasoning_seconds"] <= result.structured[
        "first_reasoning_visible_seconds"
    ]
    assert result.structured["first_reasoning_visible_seconds"] <= result.structured[
        "first_text_seconds"
    ]


@pytest.mark.asyncio
async def test_cline_reasoning_callback_preserves_cancellation() -> None:
    thought = "x" * 180
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    f'data: {{"choices":[{{"delta":{{"reasoning":"{thought}"}}}}]}}\n\n'
                    'data: {"choices":[{"delta":{"content":"unseen"},"finish_reason":"stop"}]}\n\n'
                ),
            )
        ),
    )

    async def cancel(_: str) -> None:
        raise asyncio.CancelledError()

    answer: list[str] = []
    try:
        with pytest.raises(asyncio.CancelledError):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=lambda delta: _collect(answer, delta),
                on_reasoning=cancel,
            )
    finally:
        await client.aclose()
    assert answer == []


@pytest.mark.asyncio
async def test_chat_stream_rejects_a_truncated_reply_after_emitting_text() -> None:
    emitted: list[str] = []
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            )
        ),
    )
    try:
        with pytest.raises(ModelProviderError, match="ended before completion"):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=lambda delta: _collect(emitted, delta),
            )
    finally:
        await client.aclose()
    assert emitted == ["partial"]


@pytest.mark.asyncio
async def test_chat_stream_rejects_a_midstream_generation_error() -> None:
    emitted: list[str] = []
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                    'data: {"choices":[{"finish_reason":"error"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        ),
    )
    try:
        with pytest.raises(ModelProviderError, match="generation error"):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=lambda delta: _collect(emitted, delta),
            )
    finally:
        await client.aclose()
    assert emitted == ["partial"]


@pytest.mark.asyncio
async def test_chat_stream_rejects_a_length_limited_reply() -> None:
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        ),
    )
    try:
        with pytest.raises(ModelProviderError, match="ended with length"):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=lambda delta: _collect([], delta),
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_stream_keeps_permanent_http_errors() -> None:
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, json={"error": "bad key"})
        ),
    )
    try:
        with pytest.raises(ModelProviderError, match="WAQIL_CLINE_API_KEY"):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=lambda delta: _collect([], delta),
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_stream_preserves_cancellation() -> None:
    client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"choices":[{"delta":{"content":"first"}}]}\n\n',
            )
        ),
    )

    async def cancel(_: str) -> None:
        raise asyncio.CancelledError()

    try:
        with pytest.raises(asyncio.CancelledError):
            await _provider(client).generate(  # type: ignore[arg-type]
                ModelRequestV1(role="planner", system_prompt="s", user_prompt="u"),
                on_token=cancel,
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_the_budget_is_sent_as_max_completion_tokens() -> None:
    """With `max_tokens` the reasoning trace is charged against the budget and
    the content comes back empty with finish_reason=length — which reads
    exactly like a model that failed the task."""
    client = _Client(
        _Response(200, {"data": {"choices": [{"message": {"content": "x"}}]}})
    )
    await _provider(client).generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert "max_completion_tokens" in client.sent[0]
    assert "max_tokens" not in client.sent[0]

    limited = _Client(
        _Response(200, {"data": {"choices": [{"message": {"content": "x"}}]}})
    )
    result = await _provider(limited).generate(
        ModelRequestV1(
            role="planner",
            system_prompt="s",
            user_prompt="u",
            max_output_tokens=512,
        )
    )
    assert limited.sent[0]["max_completion_tokens"] == 512
    assert result.structured is not None
    assert result.structured["max_completion_tokens"] == 512


@pytest.mark.asyncio
async def test_an_unsubscribed_model_says_so_instead_of_retrying() -> None:
    """403 is the subscription, 401 is the key. Neither improves on a retry,
    and three identical failures teach the user nothing."""
    client = _Client(_Response(403, {"error": "not subscribed"}))
    with pytest.raises(ModelProviderError, match="subscription does not cover"):
        await _provider(client).generate(
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
        )
    assert len(client.sent) == 1

    client = _Client(_Response(401, {"error": "bad key"}))
    with pytest.raises(ModelProviderError, match="WAQIL_CLINE_API_KEY"):
        await _provider(client).generate(
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
        )
    assert len(client.sent) == 1


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried() -> None:
    client = _Client(
        _Response(429, {"error": "slow down"}),
        _Response(200, {"data": {"choices": [{"message": {"content": "ok"}}]}}),
    )
    result = await _provider(client).generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert result.content == "ok"
    assert len(client.sent) == 2


@pytest.mark.asyncio
async def test_weekly_provider_cap_is_not_retried_and_marks_health() -> None:
    """The ClinePass cap applies to the account, not the selected model."""
    client = _Client(
        _Response(
            429,
            {
                "error": {
                    "code": "INFERENCE_CAP_ERROR",
                    "message": (
                        "You have reached your weekly Clinepass limit. "
                        "The limit resets in 6d 2h"
                    ),
                }
            },
        )
    )
    provider = _provider(client)

    with pytest.raises(ModelProviderError) as caught:
        await provider.generate(
            ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
        )
    assert getattr(caught.value, "reason", "") == "provider_exhausted"
    assert len(client.sent) == 1

    health = await provider.health()
    assert health["configured"] is True
    assert health["reachable"] is False
    assert health["reason"] == "provider_exhausted"
    assert health["retry_after_seconds"] > 6 * 86_400

    # The in-process readiness signal prevents another known-doomed request;
    # the control plane can now skip to a different provider without spending
    # a second gateway call.
    with pytest.raises(ModelProviderError) as cached:
        await provider.generate(
            ModelRequestV1(role="coder", system_prompt="s", user_prompt="u")
        )
    assert getattr(cached.value, "reason", "") == "provider_exhausted"
    assert len(client.sent) == 1


# ── Two seats, two models ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_orchestrator_and_the_coder_are_different_models() -> None:
    """The whole point of this lane. Directing runs on the model that answers
    a planning question; writing runs on a separate ClinePass model."""
    client = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '{"path": "app/main.py", "instruction": "write it"}',
        ),
    )
    provider = _provider(client)
    await provider.project_direction({"planned_files": ["app/main.py"]})
    assert client.sent[0]["model"] == "cline-pass/qwen3.7-plus"

    client = _Client(_tool_reply("read_file", '{"path": "app/main.py"}'))
    provider = _provider(client)
    await provider.project_step({"files_still_to_write": [], "build_turn": False})
    assert client.sent[0]["model"] == "cline-pass/deepseek-v4-pro"


@pytest.mark.asyncio
async def test_the_one_shot_manifest_has_a_bounded_planner_budget() -> None:
    direction_client = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '{"path":"app/main.py","instruction":"write it"}',
        )
    )
    await _provider(direction_client, cline_max_output_tokens=32_768).project_direction(
        {}
    )
    # A compatibility checkpoint without the compact-plan marker falls back to
    # a full-context direction; a real repo made the tempting 4K cap empty twice.
    assert direction_client.sent[0]["max_completion_tokens"] == 32_768

    plan_client = _Client(
        _tool_reply(
            "return_projectbuildplanv1",
            '{"intent":"build","scope":"whole_app","files":["app/main.py"]}',
        )
    )
    await _provider(plan_client, cline_max_output_tokens=32_768).project_plan_files({})
    assert plan_client.sent[0]["max_completion_tokens"] == 8192


def test_the_manifest_schema_stays_small_enough_for_a_planner() -> None:
    properties = ProjectBuildPlanV1.model_json_schema()["properties"]
    assert set(properties) == {"intent", "scope", "files", "scenarios", "slices"}


@pytest.mark.asyncio
async def test_a_ladder_rung_may_name_the_model_for_that_seat() -> None:
    client = _Client(
        _tool_reply(
            "return_projectdirectionv1", '{"path": "a.py", "instruction": "go"}'
        )
    )
    provider = _provider(client)
    await provider.project_direction(
        {}, model_aliases={"_cline_model": "x-ai/grok-4.3"}
    )
    assert client.sent[0]["model"] == "x-ai/grok-4.3"


@pytest.mark.asyncio
async def test_routed_recipe_keeps_the_selected_cline_coder_rung() -> None:
    client = _Client(
        _tool_reply(
            "return_assetrecipev1",
            '{"entrypoint":"app.py","launch_command":["python","app.py"]}',
        )
    )
    cline = _provider(client)
    routed = RoutedModelProvider(object(), object(), cline=cline)  # type: ignore[arg-type]
    aliases = {
        "_provider": "cline",
        "_cline_model": "cline-pass/kimi-k2.7-code",
    }

    recipe = await routed.draft_asset_recipe(
        {"files": ["app.py"]}, model_aliases=aliases
    )

    assert isinstance(recipe, AssetRecipeV1)
    assert recipe.entrypoint == "app.py"
    assert client.sent[0]["model"] == "cline-pass/kimi-k2.7-code"


@pytest.mark.asyncio
async def test_cline_supports_selected_one_shot_authoring_capabilities() -> None:
    client = _Client(
        _tool_reply(
            "return_tooldefinitiondraftv1",
            '{"name":"Invoice Helper","description":"Extract invoice fields"}',
        ),
        _tool_reply(
            "return_architecturespecv1",
            '{"title":"Invoice Service","components":['
            '{"id":"api","label":"API","kind":"service"}]}',
        ),
    )
    provider = _provider(client)
    aliases = {"_cline_model": "cline-pass/qwen3.7-plus"}
    request = PlanningRequestV1(
        run_id="run_1",
        conversation_id="conversation_1",
        prompt="Create a reusable invoice extraction tool",
    )

    draft = await provider.draft_tool_definition(request, model_aliases=aliases)
    architecture = await provider.architecture_spec(
        "Draw the invoice service", "API service", model_aliases=aliases
    )

    assert draft.name == "Invoice Helper"
    assert architecture.components[0].id == "api"
    assert [call["model"] for call in client.sent] == [
        "cline-pass/qwen3.7-plus",
        "cline-pass/qwen3.7-plus",
    ]


# ── Decode ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_direction_decodes_through_the_advertised_function() -> None:
    client = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '{"path": "app/main.py", "instruction": "Define create_app().", "reuse": ["appkit.web.page"]}',
        )
    )
    direction = await _provider(client).project_direction(
        {"planned_files": ["app/main.py"]}
    )
    assert isinstance(direction, ProjectDirectionV1)
    assert direction.path == "app/main.py"
    assert direction.reuse == ["appkit.web.page"]
    assert client.sent[0]["tools"][0]["function"]["name"] == "return_projectdirectionv1"


@pytest.mark.asyncio
async def test_arguments_wrapped_in_a_fence_are_repaired_not_failed() -> None:
    """The repair layer reaches this lane too — one shared shape fix, not one
    per transport."""
    client = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '```json\n{"path": "app/main.py", "instruction": "go"}\n```',
        )
    )
    direction = await _provider(client).project_direction({})
    assert direction.path == "app/main.py"


@pytest.mark.asyncio
async def test_prose_is_a_completion_on_this_lane_too() -> None:
    """A lane that raised on prose spent three malformed strikes on a model
    that had simply answered the question."""
    client = _Client(
        _Response(
            200,
            {"data": {"choices": [{"message": {"content": "The app uses FastAPI."}}]}},
        )
    )
    step = await _provider(client).project_step({"build_turn": False})
    assert step.status == "complete"
    assert step.response == "The app uses FastAPI."


@pytest.mark.asyncio
async def test_a_project_step_becomes_a_tool_call() -> None:
    client = _Client(
        _tool_reply("create_file", '{"path": "app/main.py", "content": "x = 1\\n"}')
    )
    step = await _provider(client).project_step(
        {"build_turn": True, "files_still_to_write": ["app/main.py"]}
    )
    assert step.tool_call is not None
    assert step.tool_call.name == "create_file"
    assert step.tool_call.arguments["path"] == "app/main.py"


# ── Availability ───────────────────────────────────────────────────────────


def test_the_lane_is_unavailable_without_a_key() -> None:
    assert (
        ClineModelProvider(Settings(_env_file=None, cline_api_key="")).available
        is False
    )
    assert ClineModelProvider(_settings()).available is True


@pytest.mark.asyncio
async def test_a_missing_key_names_the_variable() -> None:
    provider = ClineModelProvider(Settings(_env_file=None, cline_api_key=""))
    with pytest.raises(ModelProviderError, match="WAQIL_CLINE_API_KEY"):
        await provider._client()


@pytest.mark.asyncio
async def test_no_credits_names_the_subscription_boundary() -> None:
    """The failure that silently stalled a live turn. ClinePass covers the
    cline-pass/* models; Anthropic and xAI bill against credits, and an empty
    balance is a configuration fact, not a model failing."""
    client = _Client(
        _Response(
            402,
            {
                "error": {
                    "code": "insufficient_credits",
                    "message": "Insufficient balance.",
                }
            },
        )
    )
    with pytest.raises(ModelProviderError) as caught:
        await _provider(client).project_direction({})
    message = str(caught.value)
    assert "credits" in message
    assert "cline-pass/*" in message
    assert "app.cline.bot/credits" in message
    # Not retried: an empty balance does not fill itself in one second.
    assert len(client.sent) == 1
