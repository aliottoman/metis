"""The planner boundary: schema enforcement, bounded repair, and accounting.

A live run failed here twice in one turn -- malformed JSON on the first
attempt, a schema violation on the bounded repair -- and reported a planner
bill of zero, because only successful calls were ever recorded. These pin the
transport's decode contract and the accounting that has to survive a failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import (
    AcceptanceScenarioV1,
    ProjectBuildPlanV1,
    ProjectVerticalSliceV1,
)
from waqil_api.control_plane import ControlPlane, _planner_failure_stage
from waqil_api.model_provider import ModelProviderError, is_cloud_model


def test_provider_native_schema_enforcement_is_used_only_where_it_works() -> None:
    """Cloud models get a tool schema; local models get a decode grammar.

    Ollama Cloud ignores ``format`` -- the decode path documents this as a
    measured property of every model family tried -- so asking for native
    JSON-schema enforcement there would be a constraint the runtime silently
    drops, leaving unvalidated prose to look enforced. The split is the audit
    finding: enforcement where it is real, host validation where it is not.
    """

    assert is_cloud_model("glm-5.2:cloud") is True
    assert is_cloud_model("qwen3.6:35b-mlx") is False


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, kind: str, payload: dict
    ) -> None:
        del run_id, conversation_id
        self.items.append((kind, payload))


class Planner:
    """A planner whose every attempt is scripted, usage included."""

    def __init__(self, replies: list[Any], usage: dict[str, int] | None = None) -> None:
        self.replies = list(replies)
        self.calls = 0
        # `{}` means "this provider reports no usage" and must survive as
        # such; only an omitted argument gets the default.
        self.last_usage = dict(
            {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000}
            if usage is None
            else usage
        )

    async def project_plan_files(self, request, *, model_aliases=None):
        del request, model_aliases
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _plane(tmp_path, planner: Planner) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    plane.model = planner
    plane.events = Events()
    return plane


def _state(chain: list[dict[str, str]] | None = None) -> dict[str, Any]:
    import json

    return {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "prompt": "Build it.",
        "model_aliases": {
            "_provider": "local",
            "_chain_planner": json.dumps(
                chain or [{"provider": "local", "model": "glm-5.2:cloud"}]
            ),
        },
        "project_planner_chain_index": 0,
    }


def _valid_plan() -> ProjectBuildPlanV1:
    return ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py"],
        scenarios=[AcceptanceScenarioV1(name="one", path="/x")],
        slices=[
            ProjectVerticalSliceV1(
                name="Outcome", outcome="It works.", owned_files=["app/a.py"]
            )
        ],
    )


def test_the_failure_stage_of_an_attempt_is_classified_without_quoting_it() -> None:
    from pydantic import ValidationError

    try:
        ProjectVerticalSliceV1(
            name="x", outcome="y", owned_files=["a"], integration_files=["a"]
        )
    except ValidationError as error:
        assert _planner_failure_stage(error) == "schema"

    malformed = ValueError(
        "model response contains no valid JSON object: Expecting property name"
    )
    assert _planner_failure_stage(malformed) == "json"

    transport = ModelProviderError("Cline returned HTTP 500")
    transport.reason = "backend_unreachable"  # type: ignore[attr-defined]
    assert _planner_failure_stage(transport) == "transport"


@pytest.mark.asyncio
async def test_a_malformed_reply_is_recorded_as_a_failed_attempt_with_its_cost(
    tmp_path,
) -> None:
    """The exact live shape: no valid JSON, and the call still cost tokens."""

    planner = Planner(
        [ValueError("model response contains no valid JSON object: Expecting ...")]
    )
    plane = _plane(tmp_path, planner)
    state = _state()

    with pytest.raises(Exception):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})

    [attempt] = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert attempt["ok"] is False
    assert attempt["failure_stage"] == "json"
    assert attempt["reason"] == "invalid_planner_reply"
    assert attempt["model"] == "glm-5.2:cloud"
    assert attempt["attempt"] == 1
    # Usage captured BEFORE parsing, so a failed decode still carries its bill.
    assert attempt["usage"]["total_tokens"] == 1000
    assert attempt["usage_available"] is True
    assert state["project_planner_tokens"] == 1000
    assert state["project_planner_attempts"] == 1
    # Nothing from the reply itself is stored.
    assert "Expecting" not in str(attempt)


@pytest.mark.asyncio
async def test_a_failed_attempt_without_usage_is_never_recorded_as_free(
    tmp_path,
) -> None:
    planner = Planner([ValueError("no valid JSON object")], usage={})
    plane = _plane(tmp_path, planner)
    state = _state()

    with pytest.raises(Exception):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})

    [attempt] = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert attempt["ok"] is False
    assert attempt["usage"] == {}
    assert attempt["usage_available"] is False
    assert state["project_planner_tokens"] == 0

    from waqil_api.project_capability_eval import (
        TimelineEvent,
        combined_observed_tokens,
        summarize_planner_usage,
    )

    summary = summarize_planner_usage(
        [TimelineEvent(type="run.planner_attempt", payload=attempt)]
    )
    assert summary["calls"] == 1
    assert summary["failed_calls"] == 1
    assert summary["calls_without_usage"] == 1
    combined = combined_observed_tokens({"totalTokens": 0}, summary)
    # The whole point: unknown, not zero.
    assert combined["partially_unknown"] is True


@pytest.mark.asyncio
async def test_a_valid_reply_records_a_successful_attempt(tmp_path) -> None:
    planner = Planner([_valid_plan()])
    plane = _plane(tmp_path, planner)
    state = _state()

    plan = await ControlPlane._project_planner_call(
        plane, state, "project_plan_files", {}
    )

    assert plan.files == ["app/a.py"]
    [attempt] = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert attempt["ok"] is True
    assert "failure_stage" not in attempt
    assert attempt["usage"]["total_tokens"] == 1000
    assert state["project_planner_attempts"] == 1


@pytest.mark.asyncio
async def test_a_repaired_reply_records_both_attempts_and_their_combined_cost(
    tmp_path,
) -> None:
    """Failure then success: two attempts, two bills, one usable plan."""

    planner = Planner(
        [ValueError("model response contains no valid JSON object"), _valid_plan()]
    )
    plane = _plane(
        tmp_path,
        planner,
    )
    state = _state(
        [
            {"provider": "local", "model": "glm-5.2:cloud"},
            {"provider": "local", "model": "kimi-k3:cloud"},
        ]
    )

    plan = await ControlPlane._project_planner_call(
        plane, state, "project_plan_files", {}
    )

    assert plan.files == ["app/a.py"]
    attempts = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert [item["ok"] for item in attempts] == [False, True]
    assert [item["attempt"] for item in attempts] == [1, 2]
    # The second rung is a fallback, and says so.
    assert attempts[0]["fallback"] is False
    assert attempts[1]["fallback"] is True
    assert attempts[1]["model"] == "kimi-k3:cloud"
    # Both calls counted toward the turn's planner spend.
    assert state["project_planner_tokens"] == 2000
    assert state["project_planner_attempts"] == 2


@pytest.mark.asyncio
async def test_honest_exhaustion_records_every_attempt_before_raising(
    tmp_path,
) -> None:
    """The live failure: two invalid replies, then an honest stop."""

    planner = Planner(
        [
            ValueError("model response contains no valid JSON object"),
            ValueError("1 validation error for ProjectBuildPlanV1"),
        ]
    )
    plane = _plane(tmp_path, planner)
    state = _state(
        [
            {"provider": "local", "model": "glm-5.2:cloud"},
            {"provider": "local", "model": "glm-5.2:cloud"},
        ]
    )

    with pytest.raises(Exception):
        await ControlPlane._project_planner_call(plane, state, "project_plan_files", {})

    attempts = [p for k, p in plane.events.items if k == "run.planner_attempt"]
    assert len(attempts) == 2
    assert all(item["ok"] is False for item in attempts)
    assert [item["failure_stage"] for item in attempts] == ["json", "schema"]
    # Two real calls, two real bills -- never reported as a free failure.
    assert state["project_planner_tokens"] == 2000
    assert state["project_planner_attempts"] == 2


def test_redundant_field_disagreement_no_longer_fails_the_schema() -> None:
    """The other half of the live failure, now impossible by construction.

    The repair attempt was rejected for `owned_files and integration_files
    together must name exactly the slice's files`. `files` is now derived from
    ownership, so the two cannot disagree.
    """

    plan = ProjectBuildPlanV1(
        intent="edit",
        scope="narrow",
        files=["app/a.py", "app/b.py"],
        scenarios=[],
        slices=[
            {"name": "One", "outcome": "x", "owned_files": ["app/a.py"]},
            {
                "name": "Two",
                "outcome": "y",
                # A combined list that contradicts the ownership split.
                "files": ["app/b.py", "app/a.py", "app/ghost.py"],
                "owned_files": ["app/b.py"],
                "integration_files": ["app/a.py"],
            },
        ],
    )

    assert plan.slices[1].files == ["app/b.py", "app/a.py"]
    assert [item.owned_files for item in plan.slices] == [["app/a.py"], ["app/b.py"]]
