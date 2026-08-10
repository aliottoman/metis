"""The Cline lane: one key, two seats, and three quirks verified live.

Each quirk below was found by calling the real endpoint, not by reading
documentation, and each would have produced a failure that looked like a bad
model rather than a bad client.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import ModelRequestV1, ProjectDirectionV1
from waqil_api.model_provider import ClineModelProvider, ModelProviderError


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
    client = _Client(_Response(200, {"data": {"choices": [{"message": {"content": "hello"}}]}}))
    result = await _provider(client).generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_the_budget_is_sent_as_max_completion_tokens() -> None:
    """With `max_tokens` the reasoning trace is charged against the budget and
    the content comes back empty with finish_reason=length — which reads
    exactly like a model that failed the task."""
    client = _Client(_Response(200, {"data": {"choices": [{"message": {"content": "x"}}]}}))
    await _provider(client).generate(
        ModelRequestV1(role="planner", system_prompt="s", user_prompt="u")
    )
    assert "max_completion_tokens" in client.sent[0]
    assert "max_tokens" not in client.sent[0]


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


# ── Two seats, two models ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_orchestrator_and_the_coder_are_different_models() -> None:
    """The whole point of this lane. Directing runs on the model that answers
    a planning question in seventeen tokens; writing runs on the subscription."""
    client = _Client(
        _tool_reply("return_projectdirectionv1", '{"path": "app/main.py", "instruction": "write it"}'),
    )
    provider = _provider(client)
    await provider.project_direction({"planned_files": ["app/main.py"]})
    assert client.sent[0]["model"] == "anthropic/claude-opus-4.5"

    client = _Client(_tool_reply("read_file", '{"path": "app/main.py"}'))
    provider = _provider(client)
    await provider.project_step({"files_still_to_write": [], "build_turn": False})
    assert client.sent[0]["model"] == "cline-pass/deepseek-v4-pro"


@pytest.mark.asyncio
async def test_a_ladder_rung_may_name_the_model_for_that_seat() -> None:
    client = _Client(
        _tool_reply("return_projectdirectionv1", '{"path": "a.py", "instruction": "go"}')
    )
    provider = _provider(client)
    await provider.project_direction({}, model_aliases={"_cline_model": "x-ai/grok-4.3"})
    assert client.sent[0]["model"] == "x-ai/grok-4.3"


# ── Decode ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_direction_decodes_through_the_advertised_function() -> None:
    client = _Client(
        _tool_reply(
            "return_projectdirectionv1",
            '{"path": "app/main.py", "instruction": "Define create_app().", "reuse": ["appkit.web.page"]}',
        )
    )
    direction = await _provider(client).project_direction({"planned_files": ["app/main.py"]})
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
        _Response(200, {"data": {"choices": [{"message": {"content": "The app uses FastAPI."}}]}})
    )
    step = await _provider(client).project_step({"build_turn": False})
    assert step.status == "complete"
    assert step.response == "The app uses FastAPI."


@pytest.mark.asyncio
async def test_a_project_step_becomes_a_tool_call() -> None:
    client = _Client(_tool_reply("create_file", '{"path": "app/main.py", "content": "x = 1\\n"}'))
    step = await _provider(client).project_step({"build_turn": True, "files_still_to_write": ["app/main.py"]})
    assert step.tool_call is not None
    assert step.tool_call.name == "create_file"
    assert step.tool_call.arguments["path"] == "app/main.py"


# ── Availability ───────────────────────────────────────────────────────────


def test_the_lane_is_unavailable_without_a_key() -> None:
    assert ClineModelProvider(Settings(_env_file=None, cline_api_key="")).available is False
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
        _Response(402, {"error": {"code": "insufficient_credits", "message": "Insufficient balance."}})
    )
    with pytest.raises(ModelProviderError) as caught:
        await _provider(client).project_direction({})
    message = str(caught.value)
    assert "credits" in message
    assert "cline-pass/*" in message
    assert "app.cline.bot/credits" in message
    # Not retried: an empty balance does not fill itself in one second.
    assert len(client.sent) == 1
