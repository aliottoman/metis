"""Every project-loop exit uses the approval gate's definition of blocking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.contracts import ProjectAgentStepV1
from waqil_api.control_plane import ControlPlane


PLAN = [
    "app/a.py",
    "app/b.py",
    "app/c.py",
    "app/d.py",
    "app/e.py",
    "app/f.py",
    "app/g.py",
]

ADVISORY = {
    "path": "app/a.py",
    "error": ("import app.main failed: ConfigError: OCI_RESPONSES_BASE_URL is not set"),
    "severity": "error",
    "kind": "import",
    "rung": "runtime",
}

BLOCKING = {
    "path": "app/a.py",
    "error": "SyntaxError: invalid syntax",
    "severity": "error",
    "kind": "syntax",
    "rung": "syntax",
}

ACCEPTANCE_BLOCKING = {
    "path": "app/a.py",
    "error": (
        "acceptance: persisted review failed: sqlite3.ProgrammingError: "
        "Cannot operate on a closed database"
    ),
    "severity": "error",
    "kind": "acceptance",
    "rung": "runtime",
}


async def _noop(*args: object, **kwargs: object) -> None:
    return None


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self,
        run_id: str,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.items.append((event_type, payload))


def _staged(paths: list[str]) -> dict[str, dict[str, Any]]:
    return {
        path: {
            "content": f"# {path}\n",
            "origin": "create",
            "base_sha256": "",
            "bytes": len(path) + 3,
        }
        for path in paths
    }


def _state(**extra: Any) -> dict[str, Any]:
    return {
        "prompt": "Build the project",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_iterations": 2,
        "project_staged": _staged([PLAN[0]]),
        "project_planned_files": [],
        "project_planned_scenarios": [],
        "project_direction": {},
        "project_focus_path": "",
        "project_repair_strategy": {},
        "project_syntax_retries": 0,
        "project_verified_prefix": 0,
        "project_slice_verifications": 0,
        "project_repair_context": {},
        "project_consecutive_reads": 0,
        "project_verify_bonus_steps": 0,
        "project_trace": [],
        **extra,
    }


def _plane(finding: dict[str, str]) -> tuple[ControlPlane, _Events]:
    plane = object.__new__(ControlPlane)
    events = _Events()
    plane.events = events
    plane.settings = SimpleNamespace(project_agent_max_steps=8)
    plane._guard = _noop

    async def verify(
        project_id: str, staged: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        assert project_id == "asset_x"
        return {
            "errors": [finding],
            "warnings": [],
            "notes": [],
            "checks": [{"kind": "import", "ok": False}],
        }

    async def context(project_id: str) -> dict[str, Any]:
        return {"manifest": {"file_tree": PLAN}}

    plane.projects = SimpleNamespace(
        context=context,
        record_plan=_noop,
        record_learnings=_noop,
    )
    plane._verify_staged_changeset = verify
    return plane, events


async def _exercise(
    branch: str, finding: dict[str, str]
) -> tuple[dict[str, Any], _Events]:
    plane, events = _plane(finding)
    if branch == "host_completion":
        result = await ControlPlane._project_step(
            plane,
            _state(
                project_planned_files=[PLAN[0]],
                project_direction={},
                project_focus_path="",
            ),
        )
    elif branch == "step_cap":
        result = await ControlPlane._project_step(
            plane,
            _state(project_iterations=8),
        )
    elif branch == "exploration_cap":
        result = await ControlPlane._project_step(
            plane,
            _state(project_consecutive_reads=12),
        )
    elif branch == "dependency_slice":
        result = await ControlPlane._project_step(
            plane,
            _state(
                project_iterations=6,
                project_staged=_staged(PLAN[:6]),
                project_planned_files=PLAN,
                project_planned_scenarios=[{"name": "complete app"}],
            ),
        )
    elif branch == "model_completion":
        result = await ControlPlane._project_step_result(
            plane,
            _state(project_planned_files=[PLAN[0]]),
            ProjectAgentStepV1(status="complete", response="Finished."),
            2,
            {"manifest": {"file_tree": PLAN}},
            planned=[PLAN[0]],
        )
    else:  # pragma: no cover - guarded by the parametrization below
        raise AssertionError(branch)
    return result, events


BRANCHES = [
    "host_completion",
    "step_cap",
    "exploration_cap",
    "dependency_slice",
    "model_completion",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", BRANCHES)
async def test_runtime_environment_advisories_do_not_consume_repairs(
    branch: str,
) -> None:
    result, events = await _exercise(branch, ADVISORY)

    assert result.get("project_syntax_retries", 0) == 0
    assert not result.get("project_direction")
    assert not result.get("project_focus_path")
    assert not result.get("project_write_pin")

    verified = [
        payload for kind, payload in events.items if kind == "project.staged_verified"
    ]
    assert len(verified) == 1
    # Advisory classification changes control flow, not observability: the raw
    # verifier total and exact finding still reach the durable timeline.
    assert verified[0]["errors"] == 1
    assert verified[0]["findings"][0]["error"] == ADVISORY["error"]

    if branch == "host_completion":
        assert result["project_verified_prefix"] == 1
        assert "All 1 planned file change(s)" in result["response_text"]
    elif branch == "step_cap":
        assert "step limit" in result["response_text"]
    elif branch == "exploration_cap":
        assert "stopped after inspecting 12 files" in result["response_text"]
    elif branch == "dependency_slice":
        assert result["project_verified_prefix"] == 6
        checked = [
            payload for kind, payload in events.items if kind == "project.slice_checked"
        ]
        assert checked[0]["errors"] == 1
    else:
        assert result["response_text"] == "Finished."


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", BRANCHES)
async def test_host_proven_findings_still_enter_the_repair_loop(branch: str) -> None:
    result, events = await _exercise(branch, BLOCKING)

    assert result["project_syntax_retries"] == 1
    assert result["project_direction"]["path"] == "app/a.py"
    assert result["project_focus_path"] == "app/a.py"
    assert result["project_write_pin"] == ["app/a.py"]
    assert result["project_repair_context"]["findings"] == [BLOCKING]

    verified = [
        payload for kind, payload in events.items if kind == "project.staged_verified"
    ]
    assert len(verified) == 1
    assert verified[0]["errors"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["host_completion", "dependency_slice"])
async def test_acceptance_errors_enter_repairs_from_completion_and_slices(
    branch: str,
) -> None:
    result, events = await _exercise(branch, ACCEPTANCE_BLOCKING)

    assert result["project_syntax_retries"] == 1
    assert result["project_direction"]["path"] == "app/a.py"
    assert result["project_write_pin"] == ["app/a.py"]
    assert result["project_repair_context"]["findings"] == [ACCEPTANCE_BLOCKING]
    assert any(kind == "project.staged_verified" for kind, _ in events.items)
