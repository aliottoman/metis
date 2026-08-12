"""Bounded strategy and model recovery for the project loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.contracts import (
    WHOLE_FILE_END_LINE,
    ProjectAgentStepV1,
    ProjectToolCallV1,
    project_whole_file_schema,
)
from waqil_api.control_plane import (
    _FOCUSED_NO_PROGRESS_SWITCH_STEPS,
    _MALFORMED_MODEL_SWITCH_STEPS,
    _PLAN_AFTER_STEPS,
    ControlPlane,
)
from waqil_api.model_provider import project_roster
from waqil_api.project_workspace import (
    ProjectWorkspaceError,
    _guard_whole_file_repair,
    _splice_lines,
)


async def _noop(*args: object, **kwargs: object) -> None:
    return None


def _step_plane(model: object, projects: object) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.model = model
    plane.projects = projects
    plane.events = SimpleNamespace(emit=_noop)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_verify_bonus_steps=0,
        project_staged_max_files=48,
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1_800,
        project_reference_enabled=False,
        project_reference_dir=Path("/none"),
        project_reference_max_chars=0,
        project_reference_max_chars_local=0,
        project_repo_map_enabled=False,
        project_orchestrator_enabled=False,
    )
    plane._guard = _noop
    plane._stage = _noop
    return plane


def _cline_chain() -> str:
    return json.dumps(
        [
            {"provider": "cline", "model": "cline-pass/deepseek-v4-pro"},
            {"provider": "cline", "model": "cline-pass/kimi-k3"},
            {"provider": "cline", "model": "cline-pass/kimi-k2.7-code"},
        ]
    )


@pytest.mark.asyncio
async def test_empty_manifest_advances_to_the_next_planner_model() -> None:
    calls: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    class Model:
        async def project_plan_files(self, request, *, model_aliases=None):
            model = model_aliases["_cline_model"]
            calls.append(model)
            files = [] if model.endswith("glm-5.2") else ["app/main.py"]
            return SimpleNamespace(
                files=files, scenarios=[], intent="build", scope="whole_app"
            )

    async def emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    plane = SimpleNamespace(
        model=Model(),
        settings=SimpleNamespace(project_staged_max_files=48),
        events=SimpleNamespace(emit=emit),
    )
    state = {
        "prompt": "Build a complete document extraction application from scratch.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "project_iterations": _PLAN_AFTER_STEPS,
        "project_planner_chain_index": 0,
        "model_aliases": {
            "_provider": "cline",
            "_chain_planner": json.dumps(
                [
                    {"provider": "cline", "model": "cline-pass/glm-5.2"},
                    {"provider": "cline", "model": "cline-pass/qwen3.7-plus"},
                ]
            ),
        },
    }

    plan = await ControlPlane._project_manifest(plane, state, {}, _PLAN_AFTER_STEPS, {})

    assert calls == ["cline-pass/glm-5.2", "cline-pass/qwen3.7-plus"]
    assert plan["files"] == ["app/main.py"]
    assert state["project_planner_chain_index"] == 1
    fallback = [payload for kind, payload in events if kind == "run.model_fallback"]
    assert fallback[0]["reason"] == "empty_manifest"


@pytest.mark.asyncio
async def test_repeated_malformed_reply_switches_before_the_next_model_call() -> None:
    calls: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []

    class Model:
        async def project_step(self, request, *, model_aliases=None):
            calls.append(model_aliases["_cline_model"])
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(name="list_files", arguments={}),
            )

    async def context(project_id: str) -> dict[str, Any]:
        return {"manifest": {"file_tree": []}, "metis_md": ""}

    async def emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    plane = _step_plane(Model(), SimpleNamespace(context=context))
    plane.events = SimpleNamespace(emit=emit)
    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Fix the validation bug in app/main.py.",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {
                "_project_id": "asset_x",
                "_provider": "cline",
                "_chain_coder": _cline_chain(),
            },
            "project_iterations": 4,
            "project_plan_taken": True,
            "project_build_intent": "edit",
            "project_malformed_streak": _MALFORMED_MODEL_SWITCH_STEPS,
            "project_chain_index": 0,
        },
    )

    assert calls == ["cline-pass/kimi-k3"]
    assert result["project_chain_index"] == 1
    assert result["project_malformed_streak"] == 0
    fallback = [payload for kind, payload in events if kind == "run.model_fallback"]
    assert fallback[0]["reason"] == "repeated_malformed_reply"


@pytest.mark.asyncio
async def test_focused_no_progress_switches_without_spending_another_model_call() -> (
    None
):
    events: list[tuple[str, dict[str, Any]]] = []

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"fallback should happen before model call {name}")

    async def context(project_id: str) -> dict[str, Any]:
        return {"manifest": {"file_tree": []}, "metis_md": ""}

    async def emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    plane = _step_plane(MustNotRun(), SimpleNamespace(context=context))
    plane.events = SimpleNamespace(emit=emit)
    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Fix app/main.py.",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {
                "_project_id": "asset_x",
                "_provider": "cline",
                "_chain_coder": _cline_chain(),
            },
            "project_iterations": 8,
            "project_planned_files": ["app/main.py"],
            "project_focus_path": "app/main.py",
            "project_stall_steps": _FOCUSED_NO_PROGRESS_SWITCH_STEPS,
            "project_chain_index": 0,
        },
    )

    assert result["project_chain_index"] == 1
    assert result["project_stall_steps"] == 0
    assert result["project_pending_call"] == {}
    fallback = [payload for kind, payload in events if kind == "run.model_fallback"]
    assert fallback[0]["reason"] == "focused_no_progress"


@pytest.mark.asyncio
async def test_exact_edit_refusal_selects_and_observes_whole_file_repair() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class Workspace:
        async def execute_staged(self, project_id, call, staged, next_paths=()):
            raise ProjectWorkspaceError(
                "patch context did not match",
                argument_shape=True,
                repair_strategy="whole_file",
            )

    async def emit(run_id, conversation_id, kind, payload):
        events.append((kind, payload))

    plane = object.__new__(ControlPlane)
    plane.projects = Workspace()
    plane.events = SimpleNamespace(emit=emit)
    plane.settings = SimpleNamespace(project_verify_max_runs=2)
    plane._guard = _noop
    plane._stage = _noop
    result = await ControlPlane._project_execute(
        plane,
        {
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_trace": [],
            "project_staged": {"app/main.py": {"content": "broken = True\n"}},
            "project_pending_call": {
                "name": "apply_patch",
                "arguments": {
                    "path": "app/main.py",
                    "original": "missing",
                    "replacement": "fixed",
                },
            },
        },
    )

    assert result["project_retry_tool"] == ""
    assert result["project_repair_strategy"]["kind"] == "whole_file"
    assert result["project_repair_strategy"]["path"] == "app/main.py"
    recovery = [
        payload for kind, payload in events if kind == "project.recovery_strategy"
    ]
    assert recovery[0]["to"] == "replace_lines:whole_file"


@pytest.mark.asyncio
async def test_whole_file_request_prefetches_current_bytes_and_pins_the_write() -> None:
    seen: dict[str, Any] = {}

    class Model:
        async def project_step(self, request, *, model_aliases=None):
            seen.update(request)
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="replace_lines",
                    arguments={
                        "path": "app/main.py",
                        "start_line": 1,
                        "end_line": WHOLE_FILE_END_LINE,
                        "replacement": "fixed = True\n",
                    },
                ),
            )

    class Workspace:
        async def context(self, project_id):
            return {"manifest": {"file_tree": ["app/main.py"]}, "metis_md": ""}

        async def execute_staged(self, project_id, call, staged, next_paths=()):
            assert call.name == "read_file"
            return {
                "path": "app/main.py",
                "content": "broken = True\n",
                "start_line": 1,
                "end_line": 1,
                "total_lines": 1,
                "truncated": False,
            }, None

    plane = _step_plane(Model(), Workspace())
    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Fix app/main.py.",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_iterations": 5,
            "project_plan_taken": True,
            "project_planned_files": ["app/main.py"],
            "project_staged": {"app/main.py": {"content": "broken = True\n"}},
            "project_repair_strategy": {"kind": "whole_file", "path": "app/main.py"},
        },
    )

    assert any(
        entry["tool"] == "read_file"
        and entry["result"]["output"]["content"] == "broken = True\n"
        for entry in seen["tool_trace"]
    )
    assert seen["files_still_to_write"] == ["app/main.py"]
    assert seen["repair_strategy"] == {"kind": "whole_file", "path": "app/main.py"}
    assert result["project_pending_call"]["name"] == "replace_lines"


def test_whole_file_recovery_is_structural_for_both_provider_shapes() -> None:
    path = "app/main.py"
    schema = project_whole_file_schema(path)
    arguments = schema["properties"]["arguments"]
    assert schema["properties"]["tool"]["enum"] == ["replace_lines"]
    assert arguments["properties"]["path"]["enum"] == [path]
    assert arguments["properties"]["start_line"]["enum"] == [1]
    assert arguments["properties"]["end_line"]["enum"] == [WHOLE_FILE_END_LINE]

    roster = project_roster(
        {
            "build_turn": True,
            "reads_closed": True,
            "repair_strategy": {"kind": "whole_file", "path": path},
        }
    )
    by_name = {tool["name"]: tool for tool in roster}
    assert set(by_name) == {"replace_lines", "revise_plan"}
    replace = by_name["replace_lines"]["parameters"]["properties"]
    assert replace["path"]["enum"] == [path]
    assert replace["start_line"]["enum"] == [1]
    assert replace["end_line"]["enum"] == [WHOLE_FILE_END_LINE]


def test_whole_file_range_can_rewrite_an_empty_existing_file() -> None:
    assert (
        _splice_lines(
            "",
            {
                "start_line": 1,
                "end_line": WHOLE_FILE_END_LINE,
                "replacement": "fixed = True\n",
            },
        )
        == "fixed = True\n"
    )


def test_whole_file_repair_cannot_pass_by_deleting_the_module() -> None:
    before = """\
class AuditEvent:
    pass

def insert_audit_event() -> int:
    return 1

def get_event() -> AuditEvent | None:
    return None
"""
    after = "def insert_audit_event() -> int:\n    return 1\n"
    with pytest.raises(
        ProjectWorkspaceError, match="incomplete|removed public"
    ) as error:
        _guard_whole_file_repair(
            "app/repository.py",
            before * 4,
            after,
            {"start_line": 1, "end_line": WHOLE_FILE_END_LINE},
        )
    assert error.value.repair_strategy == "whole_file"


def test_whole_file_repair_preserves_public_surface() -> None:
    before = """\
class AuditEvent:
    pass

def insert_audit_event(value: int | None) -> int:
    return value

def get_event() -> AuditEvent | None:
    return None
"""
    after = before.replace(
        "    return value\n", "    assert value is not None\n    return value\n"
    )
    _guard_whole_file_repair(
        "app/repository.py",
        before,
        after,
        {"start_line": 1, "end_line": WHOLE_FILE_END_LINE},
    )


def test_whole_file_html_can_extract_a_large_inline_script() -> None:
    markup = """\
<!doctype html>
<html><head><title>Meridian Evidence Desk</title></head>
<body><main><h1>Meridian Evidence Desk</h1><section id="upload">Upload</section>
<section id="review">Review</section><section id="audit">Audit history</section></main>
"""
    before = (
        markup
        + "<script>\n"
        + ("const value = 'interaction';\n" * 500)
        + "</script></body></html>\n"
    )
    after = markup + '<script src="/static/app.js"></script></body></html>\n'

    _guard_whole_file_repair(
        "app/static/index.html",
        before,
        after,
        {"start_line": 1, "end_line": WHOLE_FILE_END_LINE},
    )


def test_whole_file_html_cannot_replace_the_page_with_only_a_script_tag() -> None:
    before = (
        "<!doctype html><html><body>"
        + "".join(
            f'<section id="panel-{index}">Workflow {index}</section>'
            for index in range(40)
        )
        + ("<script>const value = 1;</script>" * 100)
        + "</body></html>"
    )
    after = '<script src="/static/app.js"></script>\n'

    with pytest.raises(ProjectWorkspaceError, match="incomplete"):
        _guard_whole_file_repair(
            "app/static/index.html",
            before,
            after,
            {"start_line": 1, "end_line": WHOLE_FILE_END_LINE},
        )


def test_actual_last_line_is_also_guarded_as_a_whole_file_html_rewrite() -> None:
    before = (
        "<!doctype html><html><body>"
        + "".join(
            f'<section id="panel-{index}">Workflow {index}</section>'
            for index in range(40)
        )
        + "</body></html>"
    )
    after = '<script src="/static/app.js"></script>\n'

    # Minified HTML is one physical line, so 1..1 is the complete file even
    # though it does not use the host's recovery sentinel.
    with pytest.raises(ProjectWorkspaceError, match="incomplete"):
        _guard_whole_file_repair(
            "app/static/index.html",
            before,
            after,
            {"start_line": 1, "end_line": 1},
        )
