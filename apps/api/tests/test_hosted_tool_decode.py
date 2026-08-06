"""The hosted transport: tool-calling decode for models the grammar cannot reach.

Ollama Cloud ignores ``format`` on every model family measured, and enforces
tool calling on the same endpoint — so the decode protocol is a property of
the transport, picked by the model name. These tests pin the whole seam
without a network: a hosted name takes the tool path with the same function
schemas the OCI provider sends, a local name keeps its grammar untouched, the
three measured failure modes surface as ``ModelProviderError`` so the loop's
malformed-reply handling covers them, and a fabricated completion is declined
by the same provider-independent guard that declines it on the OCI path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import PROJECT_TOOL_REQUIRED_ARGUMENTS
from waqil_api.model_provider import (
    ModelProviderError,
    OCIResponsesModelProvider,
    OllamaModelProvider,
)
from waqil_api.project_tools import (
    FINISH_TOOL_NAME,
    chat_tool_format,
    narrowed_project_tools,
    unrestricted_project_tools,
)

HOSTED = {"coder": "gpt-oss:120b-cloud", "planner": "gpt-oss:120b-cloud"}
LOCAL = {"coder": "qwen3-coder:30b", "planner": "qwen3.6:35b-mlx"}


class ToolReply:
    """A chat reply shaped like the AIMessage LangChain hands back."""

    def __init__(self, tool_calls=None, content="", invalid_tool_calls=None):
        self.tool_calls = tool_calls or []
        self.invalid_tool_calls = invalid_tool_calls or []
        self.content = content


class ScriptedChat:
    """Serves scripted replies and records every request it was sent."""

    def __init__(self, replies, requests, parameters):
        self.replies = replies
        self.requests = requests
        self.parameters = parameters

    async def ainvoke(self, messages, tools=None):
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "format": self.parameters.get("format"),
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def provider_with(settings, *replies) -> tuple[OllamaModelProvider, list[dict]]:
    provider = OllamaModelProvider(settings)
    queue = list(replies)
    requests: list[dict] = []
    provider._chat_type = lambda **parameters: ScriptedChat(
        queue, requests, parameters
    )
    return provider, requests


def build_request(**overrides) -> dict:
    request = {
        "user_request": "Build out this project from scratch.",
        "build_turn": True,
        "files_still_to_write": ["app/main.py", "requirements.txt"],
        "step": 1,
    }
    request.update(overrides)
    return request


# --- M1: both providers advertise the same tool set --------------------------


def test_both_providers_advertise_the_same_tool_set(settings) -> None:
    oci = OCIResponsesModelProvider(
        Settings(
            _env_file=None,
            allow_oci_responses=True,
            oci_responses_project_id="ocid1.aiproject.oc1.test",
        )
    )
    shared = unrestricted_project_tools()
    assert oci._unrestricted_project_tools() == shared
    hosted = chat_tool_format(shared)
    assert [tool["function"]["name"] for tool in hosted] == [
        tool["name"] for tool in shared
    ]
    # Same parameters object by value, not a second hand-written list.
    for flat, nested in zip(shared, hosted):
        assert nested["function"]["parameters"] == flat["parameters"]
        assert nested["function"]["description"] == flat["description"]
    assert {tool["name"] for tool in shared} == set(
        PROJECT_TOOL_REQUIRED_ARGUMENTS
    ) | {FINISH_TOOL_NAME}


def test_owed_files_narrow_create_file_and_keep_finish(settings) -> None:
    """M4: the manifest binds the hosted path, and finishing stays legal.

    Withholding finish_project_task was tried on the OCI path and measured
    worse — a model with no legal move burns the whole budget — so the hosted
    transport matches it: create_file narrows to the owed files, finish stays.
    """
    owed = ["app/main.py", ".env.example"]
    tools = {tool["name"]: tool for tool in narrowed_project_tools(owed)}
    assert tools["create_file"]["parameters"]["properties"]["path"]["enum"] == owed
    assert FINISH_TOOL_NAME in tools
    # And the unrestricted list is rebuilt per call, never mutated in place.
    fresh = {tool["name"]: tool for tool in unrestricted_project_tools()}
    assert "enum" not in fresh["create_file"]["parameters"]["properties"]["path"]


# --- M2: the decode branch and its three measured failure modes ---------------


@pytest.mark.asyncio
async def test_a_hosted_build_step_sends_tools_and_converts_the_call(settings) -> None:
    provider, requests = provider_with(
        settings,
        ToolReply(
            tool_calls=[
                {
                    "name": "create_file",
                    "args": {"path": "app/main.py", "content": "print('hi')\n"},
                }
            ]
        ),
    )
    step = await provider.project_step(build_request(), model_aliases=HOSTED)
    assert step.status == "tool"
    assert step.tool_call is not None
    assert step.tool_call.name == "create_file"
    assert step.tool_call.arguments["path"] == "app/main.py"
    sent = requests[0]
    # The constraint travels as tools — format would be silently ignored.
    assert sent["format"] is None
    names = [tool["function"]["name"] for tool in sent["tools"]]
    assert set(names) == set(PROJECT_TOOL_REQUIRED_ARGUMENTS) | {FINISH_TOOL_NAME}
    # M4: on a build turn, create_file is narrowed to the files still owed.
    create = next(
        tool["function"]
        for tool in sent["tools"]
        if tool["function"]["name"] == "create_file"
    )
    assert create["parameters"]["properties"]["path"]["enum"] == [
        "app/main.py",
        "requirements.txt",
    ]


@pytest.mark.asyncio
async def test_prose_instead_of_a_tool_call_is_a_model_error(settings) -> None:
    provider, _ = provider_with(
        settings, ToolReply(content="I created the files you asked for.")
    )
    with pytest.raises(ModelProviderError, match="prose instead of a project tool call"):
        await provider.project_step(build_request(), model_aliases=HOSTED)


@pytest.mark.asyncio
async def test_string_arguments_are_parsed_not_refused(settings) -> None:
    provider, _ = provider_with(
        settings,
        ToolReply(
            tool_calls=[
                {
                    "name": "read_file",
                    "args": json.dumps({"path": "app/main.py"}),
                }
            ]
        ),
    )
    step = await provider.project_step(build_request(), model_aliases=HOSTED)
    assert step.tool_call is not None
    assert step.tool_call.arguments == {"path": "app/main.py"}


@pytest.mark.asyncio
async def test_unparseable_arguments_are_a_model_error(settings) -> None:
    provider, _ = provider_with(
        settings,
        ToolReply(
            invalid_tool_calls=[
                {"name": "create_file", "args": "{not json", "error": "parse"}
            ]
        ),
    )
    with pytest.raises(ModelProviderError, match="invalid project tool arguments"):
        await provider.project_step(build_request(), model_aliases=HOSTED)


@pytest.mark.asyncio
async def test_an_unknown_tool_name_is_a_model_error(settings) -> None:
    provider, _ = provider_with(
        settings,
        ToolReply(tool_calls=[{"name": "delete_everything", "args": {}}]),
    )
    with pytest.raises(ModelProviderError, match="unsupported project tool"):
        await provider.project_step(build_request(), model_aliases=HOSTED)


@pytest.mark.asyncio
async def test_finish_project_task_converts_to_a_completion(settings) -> None:
    provider, _ = provider_with(
        settings,
        ToolReply(
            tool_calls=[
                {
                    "name": FINISH_TOOL_NAME,
                    "args": {"response": "Both files are staged.", "learnings": []},
                }
            ]
        ),
    )
    step = await provider.project_step(
        build_request(build_turn=False, files_still_to_write=[]),
        model_aliases=HOSTED,
    )
    assert step.status == "complete"
    assert step.response == "Both files are staged."


@pytest.mark.asyncio
async def test_a_scripted_build_turn_runs_the_branch_end_to_end(settings) -> None:
    """M2 done-when: a full build turn through the branch, no network.

    Two writes and a finish, exactly the sequence a real hosted build takes;
    the narrowing follows the shrinking owed list step by step.
    """
    provider, requests = provider_with(
        settings,
        ToolReply(
            tool_calls=[
                {"name": "create_file", "args": {"path": "app/main.py", "content": "x\n"}}
            ]
        ),
        ToolReply(
            tool_calls=[
                {
                    "name": "create_file",
                    "args": {"path": "requirements.txt", "content": "flask\n"},
                }
            ]
        ),
        ToolReply(
            tool_calls=[
                {
                    "name": FINISH_TOOL_NAME,
                    "args": {"response": "Done: both planned files.", "learnings": []},
                }
            ]
        ),
    )
    owed = ["app/main.py", "requirements.txt"]
    staged: list[str] = []
    steps = []
    while True:
        remaining = [path for path in owed if path not in staged]
        step = await provider.project_step(
            build_request(
                build_turn=bool(remaining), files_still_to_write=remaining
            ),
            model_aliases=HOSTED,
        )
        steps.append(step)
        if step.status == "complete":
            break
        staged.append(step.tool_call.arguments["path"])
    assert [step.status for step in steps] == ["tool", "tool", "complete"]
    assert staged == owed
    enums = []
    for sent in requests:
        create = next(
            tool["function"]
            for tool in sent["tools"]
            if tool["function"]["name"] == "create_file"
        )
        enums.append(create["parameters"]["properties"]["path"].get("enum"))
    assert enums == [owed, ["requirements.txt"], None]


# --- M3: protocol selection by transport --------------------------------------


@pytest.mark.asyncio
async def test_the_transport_switches_mid_conversation_by_model_name(settings) -> None:
    """The same provider serves both dialects call by call; nothing is pinned."""
    provider, requests = provider_with(
        settings,
        # Hosted: the plan arrives as a function call.
        ToolReply(
            tool_calls=[{"name": "return_projectbuildplanv1", "args": {"files": ["a.py"]}}]
        ),
        # Local: the same contract arrives as grammar-constrained text.
        FakeText('{"files": ["b.py"]}'),
    )
    hosted_plan = await provider.project_plan_files({}, model_aliases=HOSTED)
    local_plan = await provider.project_plan_files({}, model_aliases=LOCAL)
    assert hosted_plan.files == ["a.py"]
    assert local_plan.files == ["b.py"]
    assert requests[0]["format"] is None and requests[0]["tools"] is not None
    assert requests[1]["format"] is not None and requests[1]["tools"] is None


@pytest.mark.asyncio
async def test_a_structured_decode_on_a_hosted_model_rides_one_function(settings) -> None:
    provider, requests = provider_with(
        settings,
        ToolReply(
            tool_calls=[{"name": "return_projectbuildplanv1", "args": {"files": ["app/main.py"]}}]
        ),
    )
    plan = await provider.project_plan_files({}, model_aliases=HOSTED)
    assert plan.files == ["app/main.py"]
    tools = requests[0]["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "return_projectbuildplanv1"
    assert tools[0]["function"]["parameters"]["properties"]["files"]


@pytest.mark.asyncio
async def test_a_hosted_json_answer_in_text_is_still_judged_on_its_merits(
    settings,
) -> None:
    """A model that ignored the function but answered with the right object
    should not spend a repair round-trip — validation is the authority."""
    provider, _ = provider_with(settings, ToolReply(content='{"files": ["a.py"]}'))
    assert (await provider.project_plan_files({}, model_aliases=HOSTED)).files == ["a.py"]


@pytest.mark.asyncio
async def test_preflight_never_probes_a_hosted_model(settings) -> None:
    """There is no grammar to compile on the hosted path, so preflight has
    nothing to ask the backend — and must not spend network calls asking."""
    provider = OllamaModelProvider(settings)

    def refuse(**parameters):
        raise AssertionError("preflight must not build a model for a hosted role")

    provider._chat_type = refuse
    failures = await provider.preflight_schemas(model_aliases=HOSTED)
    assert failures == {}


class FakeText:
    """A grammar-path reply: plain text content, no tool calls."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list = []
        self.invalid_tool_calls: list = []


# --- M4: a fabricated completion is declined by the shared guard --------------


async def _noop_emit(*args, **kwargs) -> None:
    return None


async def _empty_context(asset_id: str) -> dict:
    return {"name": "demo", "files": []}


@pytest.mark.asyncio
async def test_a_hosted_fabricated_completion_is_declined_by_the_guard(
    settings,
) -> None:
    """M4 done-when: the hosted transport fabricates a finish with nothing
    staged, and the same provider-independent guard that protects the OCI path
    declines it and routes the step back to the model as evidence."""
    from waqil_api.control_plane import ControlPlane

    provider, _ = provider_with(
        settings,
        ToolReply(
            tool_calls=[
                {
                    "name": FINISH_TOOL_NAME,
                    "args": {
                        "response": "I built the whole app: main.py, config.py.",
                        "learnings": [],
                    },
                }
            ]
        ),
    )

    class HostedModel:
        async def project_plan_files(self, request, *, model_aliases=None):
            return ["app/main.py", "app/config.py"]

        async def project_step(self, request, *, model_aliases=None):
            return await provider.project_step(request, model_aliases=model_aliases)

    plane = object.__new__(ControlPlane)
    plane.model = HostedModel()
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        # The spec-rewrite stage stands down here; these tests are about the
        # manifest and the guards, not the rewrite (test_spec_rewrite owns that).
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800,
        project_reference_enabled=True,
        project_reference_dir=Path("/nonexistent-reference"),
        project_reference_max_chars=14_000,
        project_reference_max_chars_local=6_000,
    )
    plane._guard = _noop_emit
    plane._stage = _noop_emit

    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Build out this project from scratch.",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x", **HOSTED},
            "project_iterations": 0,
        },
    )

    # Declined: no answer published, the streak advanced, and the refusal is
    # on the record as evidence for the next step.
    assert not result.get("response_text")
    assert result["project_empty_finish_streak"] == 1
    refusal = result["project_trace"][-1]
    assert refusal["tool"] == FINISH_TOOL_NAME
    assert refusal["result"]["ok"] is False
