"""Model tools are read-only until a project has an accepted plan.

A live plan-only probe watched a planner-stage model call `replace_lines` on
app/db.py before any plan existed. It happened to be an in-scope file, but
only by luck: there was no scope yet for it to be in. Exploration is for
looking; the plan is what turns looking into permission to write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import (
    PROJECT_TOOL_REQUIRED_ARGUMENTS,
    READ_ONLY_PROJECT_TOOLS,
    ProjectAgentStepV1,
    ProjectToolCallV1,
)
from waqil_api.control_plane import (
    PROJECT_WRITE_TOOLS,
    ControlPlane,
    _write_authority_denied,
)


# The shape of a request the plan gate applies to -- the live FastAPI repair
# request is one of these.
BUILD_PROMPT = (
    "Repair the existing fault-injected Incident Continuity FastAPI service. "
    "Modify or create exactly these application-owned files: app/db.py, "
    "app/repository.py, app/routes.py, tests/test_incidents.py."
)


def test_every_tool_is_classified_as_a_read_or_a_write() -> None:
    """No tool may sit outside the classification and escape the gate."""

    classified = READ_ONLY_PROJECT_TOOLS | PROJECT_WRITE_TOOLS
    # The remainder are control tools that stage nothing: they end the turn,
    # ask, answer, or run a host check.
    control = {"run_check", "ask_user", "respond", "revise_plan"}
    assert classified | control == set(PROJECT_TOOL_REQUIRED_ARGUMENTS)
    assert not (READ_ONLY_PROJECT_TOOLS & PROJECT_WRITE_TOOLS)
    assert PROJECT_WRITE_TOOLS == {"create_file", "apply_patch", "replace_lines"}


def test_write_authority_follows_the_accepted_plan() -> None:
    build = {"prompt": BUILD_PROMPT}
    assert "read-only" in _write_authority_denied(build)
    assert "read-only" in _write_authority_denied(
        {**build, "project_plan_taken": False}
    )
    # A plan that names nothing is not a write scope.
    assert "no authorized write scope" in _write_authority_denied(
        {**build, "project_plan_taken": True, "project_planned_files": []}
    )
    # Accepted plan with files: writing is available.
    assert (
        _write_authority_denied(
            {**build, "project_plan_taken": True, "project_planned_files": ["app/a.py"]}
        )
        == ""
    )
    # The manifest this very step just took also grants authority.
    assert _write_authority_denied(build, ["app/a.py"]) == ""
    # A turn the plan gate never applies to is unaffected: a conversational
    # one-line edit has no plan to wait for and never did.
    assert _write_authority_denied({"prompt": "Fix the typo on line 4."}) == ""


def test_a_batched_read_reply_cannot_smuggle_a_write() -> None:
    """The documented reads-only rule for extra_calls, now enforced.

    The batch runs in order with no model step between its members and reports
    `staged: False` for each, so a write riding along would bypass the write
    pin, the one-write-per-step rule, and the gate below.
    """

    step = ProjectAgentStepV1(
        status="tool",
        tool_call=ProjectToolCallV1(name="read_file", arguments={"path": "a.py"}),
        extra_calls=[
            ProjectToolCallV1(
                name="create_file", arguments={"path": "x.py", "content": "X"}
            ),
            ProjectToolCallV1(name="replace_lines", arguments={"path": "y.py"}),
            ProjectToolCallV1(name="read_file", arguments={"path": "b.py"}),
        ],
    )

    assert [call.name for call in step.extra_calls] == ["read_file"]


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, conversation_id: str, kind: str, payload: dict
    ) -> None:
        del run_id, conversation_id
        self.items.append((kind, payload))


class ExplodingWorkspace:
    """Any staged execution at all is the defect under test."""

    async def execute_staged(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a write was executed before a plan existed")

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a write was executed before a plan existed")


async def _noop(*args: Any, **kwargs: Any) -> None:
    return None


def _plane(tmp_path: Path) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        allow_test_backends=True,
    )
    plane.events = Events()
    plane.projects = ExplodingWorkspace()
    plane._stage = _noop
    return plane


def _state(**overrides: Any) -> dict[str, Any]:
    return {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        # Build-shaped, like the live probe's request: the gate applies to
        # turns where a plan is expected. A conversational one-line edit that
        # never plans is out of scope and keeps writing, as it always could.
        "prompt": BUILD_PROMPT,
        "model_aliases": {"_project_id": "asset_x", "_coding_engine": "clinecore"},
        "project_trace": [],
        "project_staged": {},
        "project_iterations": 2,
        **overrides,
    }


async def _dispatch(
    plane: ControlPlane, state: dict[str, Any], call: ProjectToolCallV1
) -> dict[str, Any]:
    step = ProjectAgentStepV1(status="tool", tool_call=call)
    return await ControlPlane._project_step_result(
        plane,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        step,
        2,
        {},
    )


@pytest.mark.asyncio
async def test_the_exact_live_pre_plan_replace_lines_is_refused(tmp_path: Path) -> None:
    """The captured FastAPI probe call, refused, staging nothing."""

    plane = _plane(tmp_path)
    state = _state()
    call = ProjectToolCallV1(
        name="replace_lines",
        arguments={
            "path": "app/db.py",
            "start_line": 1,
            "end_line": 4,
            "replacement": "SECRET = 'do not stage me'\n",
        },
    )

    result = await _dispatch(plane, state, call)

    assert result["project_pending_call"] == {}
    assert result["project_refused_streak"] == 1
    [refused] = [p for k, p in plane.events.items if k == "project.step_refused"]
    assert refused["reason"] == "write_before_plan"
    assert refused["tool"] == "replace_lines"
    assert refused["path"] == "app/db.py"
    # The attempted BYTES are never recorded anywhere.
    blob = json.dumps({"events": plane.events.items, "result": result}, default=str)
    assert "do not stage me" not in blob
    # The refusal is fed back so the model can carry on reading.
    [note] = [
        item for item in result["project_trace"] if item["tool"] == "replace_lines"
    ]
    assert note["result"]["ok"] is False
    assert "read-only" in note["result"]["error"]
    assert "content" not in note["arguments"]
    assert "replacement" not in note["arguments"]


@pytest.mark.asyncio
async def test_every_write_tool_is_refused_before_a_plan(tmp_path: Path) -> None:
    for name in sorted(PROJECT_WRITE_TOOLS):
        plane = _plane(tmp_path)
        arguments = {"path": "app/a.py"}
        if name == "create_file":
            arguments["content"] = "X\n"
        if name == "apply_patch":
            arguments.update({"original": "a", "replacement": "b"})
        if name == "replace_lines":
            arguments.update({"start_line": 1, "end_line": 1, "replacement": "b"})

        result = await _dispatch(
            plane, _state(), ProjectToolCallV1(name=name, arguments=arguments)
        )

        assert result["project_pending_call"] == {}, name
        assert any(kind == "project.step_refused" for kind, _ in plane.events.items), (
            name
        )


@pytest.mark.asyncio
async def test_a_read_is_never_affected_by_the_gate(tmp_path: Path) -> None:
    plane = _plane(tmp_path)

    result = await _dispatch(
        plane,
        _state(),
        ProjectToolCallV1(name="read_file", arguments={"path": "app/db.py"}),
    )

    # Dispatched normally: the gate only ever sees write tools.
    assert result["project_pending_call"]["name"] == "read_file"
    assert not any(kind == "project.step_refused" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_writes_become_available_once_a_plan_with_files_is_accepted(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path)
    state = _state(
        project_plan_taken=True,
        project_planned_files=["app/db.py", "app/repository.py"],
    )

    result = await _dispatch(
        plane,
        state,
        ProjectToolCallV1(
            name="create_file", arguments={"path": "app/db.py", "content": "X\n"}
        ),
    )

    assert result["project_pending_call"]["name"] == "create_file"
    assert not any(kind == "project.step_refused" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_a_plan_naming_no_files_grants_no_write_authority(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path)
    state = _state(project_plan_taken=True, project_planned_files=[])

    result = await _dispatch(
        plane,
        state,
        ProjectToolCallV1(
            name="create_file", arguments={"path": "app/db.py", "content": "X\n"}
        ),
    )

    assert result["project_pending_call"] == {}
    [refused] = [p for k, p in plane.events.items if k == "project.step_refused"]
    assert refused["reason"] == "write_before_plan"


@pytest.mark.asyncio
async def test_a_carried_repair_overlay_survives_without_premature_authority(
    tmp_path: Path,
) -> None:
    """The overlay is preserved; the authority comes from its repair scope.

    A carried changeset with no plan behind it must not become a licence to
    edit -- the bytes stay, the pen does not.
    """

    carried = {"app/db.py": {"content": "SEED = 1\n", "bytes": 10}}
    plane = _plane(tmp_path)
    state = _state(project_staged=dict(carried))

    result = await _dispatch(
        plane,
        state,
        ProjectToolCallV1(
            name="replace_lines",
            arguments={
                "path": "app/db.py",
                "start_line": 1,
                "end_line": 1,
                "replacement": "SEED = 2\n",
            },
        ),
    )

    assert result["project_pending_call"] == {}
    assert state["project_staged"] == carried
    assert "project_staged" not in result

    # With the repair's own validated scope active, the same edit proceeds.
    active = _plane(tmp_path)
    allowed = await _dispatch(
        active,
        _state(
            project_staged=dict(carried),
            project_plan_taken=True,
            project_planned_files=["app/db.py"],
            project_repair_context={"files": ["app/db.py"], "findings": []},
        ),
        ProjectToolCallV1(
            name="replace_lines",
            arguments={
                "path": "app/db.py",
                "start_line": 1,
                "end_line": 1,
                "replacement": "SEED = 2\n",
            },
        ),
    )
    assert allowed["project_pending_call"]["name"] == "replace_lines"


def test_host_scaffold_is_not_a_model_write(tmp_path: Path) -> None:
    """Scaffolding never travels through a tool call, and never counts.

    The plan-only sample first reported Meridian's appkit scaffold as eight
    staged writes; production's own line is the one that matters.
    """

    from waqil_api.control_plane import _model_has_written

    scaffold = {
        ".env.example": {"content": "X\n"},
        "appkit/__init__.py": {"content": "\n"},
        "appkit/web.py": {"content": "\n"},
    }
    assert _model_has_written(scaffold) is False
    assert _model_has_written({**scaffold, "app/db.py": {"content": "\n"}}) is True
    # And no scaffold path is a tool the gate would ever see.
    assert not set(scaffold) & (PROJECT_WRITE_TOOLS | READ_ONLY_PROJECT_TOOLS)
