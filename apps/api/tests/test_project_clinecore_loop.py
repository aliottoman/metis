from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.coding_contracts import (
    CodingFinishReason,
    CodingSessionState,
    ControlledStopReason,
    SliceResultV1,
)
from waqil_api.config import Settings
from waqil_api.contracts import RunStatus
from waqil_api.control_plane import ControlPlane, _verifier_finding_signature
from waqil_api.project_coding_engine import ProjectCodingError, ProjectCodingRound


async def _noop(*args: Any, **kwargs: Any) -> None:
    return None


class Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self,
        run_id: str,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        del run_id, conversation_id
        self.items.append((event_type, payload))


class CodingCoordinator:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.continue_calls: list[dict[str, Any]] = []
        self.recover_calls: list[dict[str, Any]] = []
        self.completed: list[str] = []
        self.released: list[tuple[str, CodingSessionState]] = []
        self.release_cleanup_ids: list[tuple[str, ...]] = []
        self.release_error: Exception | None = None
        self.next_rounds: list[ProjectCodingRound] = []
        self.recovery: ProjectCodingRound | None = None
        self.sessions = SimpleNamespace(for_run=self._for_run)

    async def _for_run(self, run_id: str) -> list[Any]:
        del run_id
        return []

    async def start(self, **kwargs: Any) -> ProjectCodingRound:
        self.start_calls.append(kwargs)
        return self.next_rounds.pop(0)

    async def continue_session(
        self, session_id: str, **kwargs: Any
    ) -> ProjectCodingRound:
        self.continue_calls.append({"session_id": session_id, **kwargs})
        return self.next_rounds.pop(0)

    async def recover_for_run(
        self, run_id: str, **kwargs: Any
    ) -> ProjectCodingRound | None:
        self.recover_calls.append({"run_id": run_id, **kwargs})
        return self.recovery

    async def complete(self, session_id: str) -> None:
        self.completed.append(session_id)

    async def release(
        self,
        session_id: str,
        terminal_state: CodingSessionState,
        *,
        additional_sidecar_ids: tuple[str, ...] = (),
    ) -> None:
        self.released.append((session_id, terminal_state))
        self.release_cleanup_ids.append(tuple(additional_sidecar_ids))
        if self.release_error is not None:
            raise self.release_error


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=tmp_path,
        asset_roots=[tmp_path / "Projects"],
        model_backend="deterministic",
        allow_test_backends=True,
        cline_api_key="test-cline-key",
        coder_model="deepseek-v4-pro:cloud",
        cline_coder_model="cline-pass/deepseek-v4-pro",
        cline_sidecar_max_rounds=4,
        cline_sidecar_unchanged_findings_limit=2,
        **overrides,
    )


def _round(
    session_id: str,
    staged: dict[str, dict[str, Any]],
    *,
    engine_error: str = "",
    recovered: bool = False,
    settled_recovery: bool = False,
    cleanup_sidecar_ids: tuple[str, ...] = (),
    result: SliceResultV1 | None = None,
    finish_reason: CodingFinishReason | None = None,
    controlled_stop_reason: ControlledStopReason | None = None,
    rejection_reason: str = "",
    rejection_path: str = "",
    rejection_repairable: bool = False,
) -> ProjectCodingRound:
    return ProjectCodingRound(
        session=SimpleNamespace(id=session_id),
        staged=staged,
        changes={"changes": []},
        result=result,
        engine_error=engine_error,
        recovered=recovered,
        settled_recovery=settled_recovery,
        cleanup_sidecar_ids=cleanup_sidecar_ids,
        finish_reason=(
            finish_reason
            if finish_reason is not None
            else result.finish_reason
            if result is not None
            else None
        ),
        controlled_stop_reason=(
            controlled_stop_reason
            if controlled_stop_reason is not None
            else result.controlled_stop_reason
            if result is not None
            else None
        ),
        rejection_reason=rejection_reason,
        rejection_path=rejection_path,
        rejection_repairable=rejection_repairable,
    )


def _state(**overrides: Any) -> dict[str, Any]:
    aliases = {
        "_project_id": "asset_x",
        "_coding_engine": "clinecore",
        "_provider": "local",
        "coder": "deepseek-v4-pro:cloud",
        "_chain_coder": json.dumps(
            [
                {"provider": "local", "model": "deepseek-v4-pro:cloud"},
                {"provider": "local", "model": "kimi-k3:cloud"},
            ]
        ),
    }
    return {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "prompt": "Build the document review application.",
        "model_aliases": aliases,
        "project_plan_taken": True,
        "project_planned_files": ["app/main.py"],
        "project_planned_scenarios": [],
        "project_build_intent": "build",
        "project_spec": {},
        "project_staged": {},
        "project_coding_session_id": "",
        "project_coding_cleanup_ids": [],
        "project_coding_rounds": 0,
        "project_coding_findings": [],
        "project_coding_finding_signature": "",
        "project_coding_unchanged_findings": 0,
        "project_chain_index": 0,
        "project_iterations": 0,
        **overrides,
    }


def _plane(
    tmp_path: Path,
    coding: CodingCoordinator,
    verifications: list[dict[str, Any]],
) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.settings = _settings(tmp_path)
    plane.project_coding = coding
    plane.events = Events()
    plane._guard = _noop
    plane._emit_staged_verification = _noop

    async def verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return verifications.pop(0)

    plane._verify_staged_changeset = verify
    return plane


def _clean() -> dict[str, Any]:
    return {"errors": [], "warnings": [], "checks": ["host"]}


@pytest.mark.asyncio
async def test_clinecore_owns_the_full_planned_slice_and_host_verifies_it(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            {"app/main.py": {"content": "APP = True\n", "bytes": 11}},
        )
    )
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane,
        _state(
            project_planned_scenarios=[
                {
                    "name": "status changes",
                    "method": "PATCH",
                    "path": "/api/projects/atlas-2/status",
                    "body_kind": "json",
                    "body": {"status": "healthy"},
                    "expect_status": "2xx",
                }
            ]
        ),
        prompt_context={"repo_map": "app/main.py: create_app()"},
    )

    [started] = coding.start_calls
    assert started["allowed_paths"] == ["app/main.py"]
    assert "FILE STATUS FOR THIS SLICE" in started["prompt"]
    assert '"method": "PATCH"' in started["prompt"]
    assert "create_app()" in started["prompt"]
    assert coding.continue_calls == []
    # The runtime remains available through review/approval; only the durable
    # approval verdict owns terminal cleanup.
    assert coding.completed == []
    assert "passed 1 independent verification" in result["response_text"]
    assert result["project_coding_session_id"] == "coding_one"


@pytest.mark.asyncio
async def test_clinecore_checkpoints_and_verifies_each_vertical_slice(
    tmp_path: Path,
) -> None:
    planned = [f"app/feature_{index}.py" for index in range(7)]
    first_files = planned[:6]
    first_staged = {
        path: {"content": f"VALUE = {index}\n", "bytes": 10}
        for index, path in enumerate(first_files)
    }
    final_staged = {
        **first_staged,
        planned[-1]: {"content": "READY = True\n", "bytes": 13},
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round("coding_slice_one", first_staged),
            _round("coding_slice_two", final_staged),
        ]
    )
    plane = _plane(tmp_path, coding, [_clean(), _clean()])
    initial = _state(project_planned_files=planned)

    first = await ControlPlane._project_clinecore_round(plane, initial)

    assert "response_text" not in first
    assert first["project_verified_prefix"] == 6
    assert first["project_coding_slice_complete"] is True
    assert coding.start_calls[0]["allowed_paths"] == first_files
    assert coding.released == []

    final = await ControlPlane._project_clinecore_round(plane, {**initial, **first})

    assert coding.released == [("coding_slice_one", CodingSessionState.COMPLETED)]
    assert coding.start_calls[1]["allowed_paths"] == [planned[-1]]
    assert final["project_verified_prefix"] == len(planned)
    assert "across the vertical slices" in final["response_text"]


def _integration_plan() -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """A genuine outcome-based plan: three independent bootstrap outcomes,
    then a fourth slice that must reopen an earlier, already-verified file
    (app/models.py) to wire it into new routes -- a real vertical-slice
    integration point, not a horizontal layer."""

    planned = [
        "app/config.py",
        "app/models.py",
        "app/routes.py",
        "app/main.py",
        "app/extra.py",
    ]
    declared_slices = [
        {
            "name": "Config bootstrap",
            "outcome": "Configuration loads.",
            "files": ["app/config.py"],
            "owned_files": ["app/config.py"],
            "integration_files": [],
            "scenario_names": ["config loads"],
        },
        {
            "name": "Models",
            "outcome": "Models import cleanly.",
            "files": ["app/models.py"],
            "owned_files": ["app/models.py"],
            "integration_files": [],
            "scenario_names": ["models import"],
        },
        {
            "name": "Routes integration",
            "outcome": "Routes wire into the app and the existing models.",
            "files": ["app/models.py", "app/routes.py", "app/main.py"],
            "owned_files": ["app/routes.py", "app/main.py"],
            "integration_files": ["app/models.py"],
            "scenario_names": ["routes respond"],
        },
        {
            "name": "Extra",
            "outcome": "The extra endpoint works.",
            "files": ["app/extra.py"],
            "owned_files": ["app/extra.py"],
            "integration_files": [],
            "scenario_names": ["extra works"],
        },
    ]
    scenarios = [
        {
            "name": name,
            "method": "GET",
            "path": "/",
            "body_kind": "none",
            "body": {},
            "expect_status": "2xx",
        }
        for name in ("config loads", "models import", "routes respond", "extra works")
    ]
    return planned, declared_slices, scenarios


@pytest.mark.asyncio
async def test_a_later_slice_may_extend_a_declared_integration_file_but_nothing_else_earlier(
    tmp_path: Path,
) -> None:
    planned, declared_slices, scenarios = _integration_plan()
    coding = CodingCoordinator()
    staged1 = {"app/config.py": {"content": "DEBUG = False\n", "bytes": 15}}
    staged2 = {
        **staged1,
        "app/models.py": {"content": "class Model: ...\n", "bytes": 18},
    }
    staged3 = {
        **staged2,
        "app/routes.py": {"content": "ROUTES = []\n", "bytes": 12},
        "app/main.py": {"content": "APP = True\n", "bytes": 11},
    }
    coding.next_rounds.extend(
        [
            _round("coding_config", staged1),
            _round("coding_models", staged2),
            _round("coding_routes", staged3),
        ]
    )
    plane = _plane(tmp_path, coding, [_clean(), _clean(), _clean()])
    state = _state(
        project_planned_files=planned,
        project_planned_slices=declared_slices,
        project_planned_scenarios=scenarios,
    )

    first = await ControlPlane._project_clinecore_round(plane, state)
    second = await ControlPlane._project_clinecore_round(plane, {**state, **first})
    third = await ControlPlane._project_clinecore_round(plane, {**state, **second})

    assert coding.start_calls[0]["allowed_paths"] == ["app/config.py"]
    assert coding.start_calls[1]["allowed_paths"] == ["app/models.py"]
    # The routes/main slice may edit its own new files AND the earlier,
    # already-verified models.py it explicitly declared as an integration
    # point -- but not the equally earlier config.py, which it never named.
    assert coding.start_calls[2]["allowed_paths"] == [
        "app/models.py",
        "app/routes.py",
        "app/main.py",
    ]
    assert "app/config.py" not in coding.start_calls[2]["allowed_paths"]
    assert third["project_verified_prefix"] == 4
    assert third["project_coding_slice_complete"] is True


@pytest.mark.asyncio
async def test_earlier_scenarios_rerun_when_a_slice_touches_their_integration_file(
    tmp_path: Path,
) -> None:
    """The routes slice reopens app/models.py, which the config and models
    slices never touched. Only the scenarios belonging to slices whose files
    are already fully verified -- config and models, not the not-yet-staged
    extra slice -- are re-checked alongside the routes slice's own scenario."""

    planned, declared_slices, scenarios = _integration_plan()
    coding = CodingCoordinator()
    staged1 = {"app/config.py": {"content": "DEBUG = False\n", "bytes": 15}}
    staged2 = {
        **staged1,
        "app/models.py": {"content": "class Model: ...\n", "bytes": 18},
    }
    staged3 = {
        **staged2,
        "app/routes.py": {"content": "ROUTES = []\n", "bytes": 12},
        "app/main.py": {"content": "APP = True\n", "bytes": 11},
    }
    coding.next_rounds.extend(
        [
            _round("coding_config", staged1),
            _round("coding_models", staged2),
            _round("coding_routes", staged3),
        ]
    )
    plane = _plane(tmp_path, coding, [_clean(), _clean(), _clean()])
    state = _state(
        project_planned_files=planned,
        project_planned_slices=declared_slices,
        project_planned_scenarios=scenarios,
    )

    first = await ControlPlane._project_clinecore_round(plane, state)
    second = await ControlPlane._project_clinecore_round(plane, {**state, **first})
    await ControlPlane._project_clinecore_round(plane, {**state, **second})

    routes_prompt = coding.start_calls[2]["prompt"]
    assert '"name": "config loads"' in routes_prompt
    assert '"name": "models import"' in routes_prompt
    assert '"name": "routes respond"' in routes_prompt
    assert '"name": "extra works"' not in routes_prompt


@pytest.mark.asyncio
async def test_first_verifier_repair_keeps_session_and_routes_deepseek_to_kimi(
    tmp_path: Path,
) -> None:
    finding = {
        "path": "app/main.py",
        "error": "NameError: repository is undefined",
        "severity": "error",
        "kind": "static",
        "rung": "mypy",
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round(
                "coding_one",
                {"app/main.py": {"content": "BROKEN = True\n", "bytes": 14}},
            ),
            _round(
                "coding_one",
                {"app/main.py": {"content": "FIXED = True\n", "bytes": 13}},
            ),
        ]
    )
    plane = _plane(
        tmp_path,
        coding,
        [
            {"errors": [finding], "warnings": [], "checks": ["mypy"]},
            _clean(),
        ],
    )
    first = await ControlPlane._project_clinecore_round(
        plane, _state(), prompt_context={"repo_map": "map"}
    )
    second_state = {**_state(), **first}

    second = await ControlPlane._project_clinecore_round(plane, second_state)

    [continued] = coding.continue_calls
    assert continued["session_id"] == "coding_one"
    assert continued["provider"].model_id == "kimi-k3:cloud"
    assert continued["operation_id"] == "run_x:round:2"
    assert continued["allowed_paths"] == ["app/main.py"]
    assert "repository is undefined" in continued["prompt"]
    assert "AUTHORIZED REPAIR PATHS" in continued["prompt"]
    assert "- app/main.py" in continued["prompt"]
    assert "authorize edits outside this exact list" in continued["prompt"]
    assert "passed 1 independent verification" in second["response_text"]


@pytest.mark.asyncio
async def test_staged_pytest_failure_enters_the_same_cline_repair_session(
    tmp_path: Path,
) -> None:
    finding = {
        "path": "app/main.py",
        "error": (
            "pytest tests/test_main.py::test_status failed against the staged "
            "project: assert 'pending' == 'ready'"
        ),
        "severity": "error",
        "kind": "test",
        "rung": "runtime",
    }
    broken = {"app/main.py": {"content": "STATUS = 'pending'\n", "bytes": 19}}
    fixed = {"app/main.py": {"content": "STATUS = 'ready'\n", "bytes": 17}}
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [_round("coding_tests", broken), _round("coding_tests", fixed)]
    )
    plane = _plane(
        tmp_path,
        coding,
        [
            {"errors": [finding], "warnings": [], "checks": ["pytest"]},
            _clean(),
        ],
    )

    first = await ControlPlane._project_clinecore_round(
        plane, _state(), prompt_context={"repo_map": "app/main.py: STATUS"}
    )
    second = await ControlPlane._project_clinecore_round(plane, {**_state(), **first})

    [continued] = coding.continue_calls
    assert continued["session_id"] == "coding_tests"
    assert continued["staged"] == broken
    assert continued["allowed_paths"] == ["app/main.py"]
    assert "test_status" in continued["prompt"]
    assert "assert 'pending' == 'ready'" in continued["prompt"]
    assert second["project_staged"] == fixed
    assert "passed 1 independent verification" in second["response_text"]


@pytest.mark.asyncio
async def test_ui_405_repair_does_not_authorize_the_protected_backend(
    tmp_path: Path,
) -> None:
    """The measured Atlas failure remains inside its five-file edit contract."""
    planned = [
        "app/static/index.html",
        "app/static/styles.css",
        "app/static/app.js",
        "tests/test_ui_contract.py",
        "README.md",
    ]
    staged = {
        path: {"content": f"updated {path}\n", "bytes": len(path) + 9}
        for path in planned
    }
    finding = {
        "path": "tests/test_ui_contract.py",
        "error": (
            "acceptance: patch_status_action_works failed: "
            "POST /api/projects/atlas-2/status returned HTTP 405, expected 2xx"
        ),
        "severity": "error",
        "kind": "acceptance",
        "rung": "runtime",
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [_round("coding_atlas", staged), _round("coding_atlas", staged)]
    )
    plane = _plane(
        tmp_path,
        coding,
        [
            {"errors": [finding], "warnings": [], "checks": ["acceptance"]},
            _clean(),
        ],
    )
    initial = _state(project_planned_files=planned)

    blocked = await ControlPlane._project_clinecore_round(
        plane, initial, prompt_context={"repo_map": "app/api.py: update_status()"}
    )
    repaired = await ControlPlane._project_clinecore_round(
        plane, {**initial, **blocked}
    )

    [continued] = coding.continue_calls
    assert continued["session_id"] == "coding_atlas"
    assert continued["allowed_paths"] == planned
    assert "- tests/test_ui_contract.py" in continued["prompt"]
    assert "- app/api.py" not in continued["prompt"]
    assert "HTTP 405" in continued["prompt"]
    assert "passed 1 independent verification" in repaired["response_text"]


@pytest.mark.asyncio
async def test_empty_interrupted_recovery_checkpoints_before_resuming_model(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.recovery = _round("coding_one", {}, recovered=True)
    plane = _plane(tmp_path, coding, [])

    recovered = await ControlPlane._project_clinecore_round(plane, _state())

    assert "response_text" not in recovered
    assert recovered["project_coding_session_id"] == "coding_one"
    assert recovered["project_coding_rounds"] == 1
    assert coding.start_calls == []
    assert coding.continue_calls == []

    coding.recovery = None
    coding.next_rounds.append(_round("coding_one", {}))
    resumed = await ControlPlane._project_clinecore_round(
        plane, {**_state(), **recovered}
    )
    assert "interrupted" in coding.continue_calls[0]["prompt"]
    assert "without producing an application file change" in resumed["response_text"]
    assert resumed["project_coding_session_id"] == "coding_one"
    assert coding.released == []


@pytest.mark.asyncio
async def test_settled_empty_recovery_finishes_without_resuming_model_or_releasing(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.recovery = _round(
        "coding_one",
        {},
        recovered=True,
        settled_recovery=True,
        cleanup_sidecar_ids=("sidecar_parent", "sidecar_expected_child"),
    )
    plane = _plane(tmp_path, coding, [])

    result = await ControlPlane._project_clinecore_round(plane, _state())

    assert "without producing an application file change" in result["response_text"]
    assert result["project_coding_session_id"] == "coding_one"
    assert result["project_coding_cleanup_ids"] == [
        "sidecar_parent",
        "sidecar_expected_child",
    ]
    assert coding.start_calls == []
    assert coding.continue_calls == []
    # The graph node is replayable. _drive releases only after it commits the
    # terminal RunStatus, so a crash here can still recover this exact verdict.
    assert coding.released == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_state", "finish_reason", "terminal_state", "staged"),
    [
        ("aborted", None, CodingSessionState.ABORTED, {}),
        (
            "failed",
            None,
            CodingSessionState.FAILED,
            {"app/main.py": {"content": "PRIVATE = True\n", "bytes": 15}},
        ),
        (
            "failed",
            "mistake_limit",
            CodingSessionState.FAILED,
            {"app/main.py": {"content": "PRIVATE = True\n", "bytes": 15}},
        ),
    ],
)
async def test_terminal_cline_round_is_a_visible_run_failure_before_no_write_or_verify(
    tmp_path: Path,
    result_state: str,
    finish_reason: CodingFinishReason | None,
    terminal_state: CodingSessionState,
    staged: dict[str, dict[str, Any]],
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            staged,
            cleanup_sidecar_ids=("sidecar_parent", "sidecar_child"),
            result=SliceResultV1(
                sessionId="sidecar_child",
                state=result_state,  # type: ignore[arg-type]
                finishReason=finish_reason,
                summary="The requested tool was denied.",
            ),
        )
    )
    verifications = [_clean()]
    plane = _plane(tmp_path, coding, verifications)

    with pytest.raises(ProjectCodingError, match=rf"ended {result_state} before"):
        await ControlPlane._project_clinecore_round(plane, _state())

    assert coding.released == [("coding_one", terminal_state)]
    assert coding.release_cleanup_ids == [("sidecar_parent", "sidecar_child")]
    [failure] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.coding_failed"
    ]
    assert failure["state"] == result_state
    assert failure["finish_reason"] == (finish_reason or "")
    assert failure["session_id"] == "coding_one"
    assert not any(kind == "run.model_fallback" for kind, _ in plane.events.items)
    # The clean verifier fixture remains unused: terminal SDK state wins even
    # if a failed/aborted round happened to leave private mirror bytes behind.
    assert verifications == [_clean()]


@pytest.mark.asyncio
async def test_host_does_not_string_match_raw_max_iteration_error(
    tmp_path: Path,
) -> None:
    staged = {"app/main.py": {"content": "READY = True\n", "bytes": 13}}
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            staged,
            result=SliceResultV1(
                sessionId="sidecar_one",
                state="failed",
                finishReason="error",
                iterations=24,
                summary="Agent runtime exceeded maxIterations (24)",
            ),
        )
    )
    verifications = [_clean()]
    plane = _plane(tmp_path, coding, verifications)

    with pytest.raises(ProjectCodingError, match="ended failed before"):
        await ControlPlane._project_clinecore_round(plane, _state())

    assert coding.released == [("coding_one", CodingSessionState.FAILED)]
    assert verifications == [_clean()]


@pytest.mark.asyncio
async def test_max_iterations_with_safe_bytes_runs_exact_verifier_and_can_approve(
    tmp_path: Path,
) -> None:
    content = "def extract_document() -> str:\n    return 'ready'\n"
    staged = {
        "app/main.py": {
            "content": content,
            "bytes": len(content.encode("utf-8")),
        }
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            staged,
            result=SliceResultV1(
                sessionId="sidecar_one",
                state="failed",
                finishReason="error",
                controlledStopReason="max_iterations",
                iterations=24,
                toolCallCount=9,
                summary="The agent stopped at its configured boundary.",
            ),
        )
    )
    plane = _plane(tmp_path, coding, [_clean()])
    verified: list[dict[str, dict[str, Any]]] = []

    async def verify(
        project_id: str,
        candidate: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del project_id, kwargs
        verified.append(candidate)
        return _clean()

    plane._verify_staged_changeset = verify  # type: ignore[method-assign]

    result = await ControlPlane._project_clinecore_round(plane, _state())

    assert verified == [staged]
    assert result["project_staged"] == staged
    assert "controlled iteration limit" in result["response_text"]
    assert "passed 1 independent verification" in result["response_text"]
    assert coding.released == []
    assert not any(kind == "project.coding_failed" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_max_iterations_blocker_preserves_bytes_and_routes_bounded_repair(
    tmp_path: Path,
) -> None:
    staged = {
        "app/main.py": {"content": "def answer() -> str:\n    return 3\n", "bytes": 34}
    }
    finding = {
        "path": "app/main.py",
        "error": "return-value: incompatible return value type",
        "severity": "error",
        "kind": "static",
        "rung": "typecheck",
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round(
                "coding_one",
                staged,
                result=SliceResultV1(
                    sessionId="sidecar_one",
                    state="failed",
                    finishReason="error",
                    controlledStopReason="max_iterations",
                    summary="Stopped at the configured boundary.",
                ),
            ),
            _round(
                "coding_one",
                staged,
                result=SliceResultV1(
                    sessionId="sidecar_two",
                    state="completed",
                ),
            ),
        ]
    )
    plane = _plane(
        tmp_path,
        coding,
        [
            {"errors": [finding], "warnings": [], "checks": ["mypy"]},
            _clean(),
        ],
    )

    first = await ControlPlane._project_clinecore_round(plane, _state())

    assert "response_text" not in first
    assert first["project_staged"] == staged
    assert first["project_coding_findings"] == [finding]
    assert coding.released == []

    repaired = await ControlPlane._project_clinecore_round(plane, {**_state(), **first})

    assert coding.continue_calls[0]["provider"].model_id == "kimi-k3:cloud"
    assert coding.continue_calls[0]["staged"] == staged
    assert "passed 1 independent verification" in repaired["response_text"]


@pytest.mark.asyncio
async def test_real_sdk_budget_stop_preserves_four_files_and_routes_missing_readme_to_k2_7(
    tmp_path: Path,
) -> None:
    planned = [
        "README.md",
        "app/static/app.js",
        "app/static/index.html",
        "app/static/styles.css",
        "tests/test_ui_contract.py",
    ]

    def entry(content: str) -> dict[str, Any]:
        return {"content": content, "bytes": len(content.encode("utf-8"))}

    staged_four = {
        "app/static/app.js": entry("document.body.dataset.ready = 'true';\n"),
        "app/static/index.html": entry("<main>Document extraction</main>\n"),
        "app/static/styles.css": entry("main { display: grid; }\n"),
        "tests/test_ui_contract.py": entry("def test_ui():\n    assert True\n"),
    }
    staged_five = {**staged_four, "README.md": entry("# Document extraction\n")}
    missing_readme = {
        "path": "README.md",
        "error": "planned artifact is missing",
        "severity": "error",
        "kind": "static",
        "rung": "conformance",
    }
    usage_error = (
        "this model uses extra usage only (not included plan usage) and your "
        "extra usage balance is empty, add extra usage or turn on auto reload"
    )
    aliases = {
        "_project_id": "asset_x",
        "_coding_engine": "clinecore",
        "_provider": "local",
        "_chain_coder": json.dumps(
            [
                {"provider": "local", "model": "deepseek-v4-pro:cloud"},
                {"provider": "local", "model": "kimi-k3:cloud"},
                {"provider": "local", "model": "kimi-k2.7-code:cloud"},
            ]
        ),
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round(
                "coding_deepseek",
                staged_four,
                result=SliceResultV1(
                    sessionId="sidecar_deepseek",
                    state="failed",
                    finishReason="error",
                    controlledStopReason="max_iterations",
                    iterations=24,
                    summary="Agent runtime exceeded maxIterations (24)",
                ),
            ),
            _round(
                "coding_k3",
                staged_four,
                result=SliceResultV1(
                    sessionId="sidecar_k3",
                    state="failed",
                    finishReason="error",
                    summary=usage_error,
                ),
            ),
            _round(
                "coding_k2_7",
                staged_five,
                result=SliceResultV1(
                    sessionId="sidecar_k2_7",
                    state="completed",
                    finishReason="completed",
                ),
            ),
        ]
    )
    plane = _plane(
        tmp_path,
        coding,
        [
            {"errors": [missing_readme], "warnings": [], "checks": ["manifest"]},
            _clean(),
        ],
    )
    initial = _state(
        model_aliases=aliases,
        project_planned_files=planned,
    )

    blocked = await ControlPlane._project_clinecore_round(plane, initial)

    assert "response_text" not in blocked
    assert blocked["project_staged"] == staged_four
    assert blocked["project_coding_findings"] == [missing_readme]
    assert coding.released == []

    unavailable = await ControlPlane._project_clinecore_round(
        plane, {**initial, **blocked}
    )

    assert coding.continue_calls[0]["provider"].model_id == "kimi-k3:cloud"
    assert "README.md" in coding.continue_calls[0]["prompt"]
    assert unavailable["project_chain_index"] == 2
    assert unavailable["project_coding_session_id"] == ""
    assert unavailable["project_staged"] == staged_four
    assert unavailable["project_coding_findings"] == [missing_readme]

    completed = await ControlPlane._project_clinecore_round(
        plane, {**initial, **blocked, **unavailable}
    )

    assert coding.start_calls[-1]["provider"].model_id == "kimi-k2.7-code:cloud"
    assert "README.md" in coding.start_calls[-1]["prompt"]
    assert completed["project_staged"] == staged_five
    assert "passed 1 independent verification" in completed["response_text"]


@pytest.mark.asyncio
async def test_in_plan_syntax_rejection_keeps_private_session_for_kimi_repair(
    tmp_path: Path,
) -> None:
    syntax_reason = (
        "generated Python is not valid: app/main.py: "
        "invalid syntax at line 1, column 12"
    )
    valid = {
        "app/main.py": {
            "content": "def answer() -> int:\n    return 42\n",
            "bytes": 35,
        }
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round(
                "coding_one",
                {},
                result=SliceResultV1(
                    sessionId="sidecar_deepseek",
                    state="failed",
                    finishReason="error",
                    controlledStopReason="max_iterations",
                    summary="Agent runtime exceeded maxIterations (32)",
                ),
                rejection_reason=syntax_reason,
                rejection_path="app/main.py",
                rejection_repairable=True,
            ),
            _round(
                "coding_one",
                valid,
                result=SliceResultV1(
                    sessionId="sidecar_kimi",
                    state="completed",
                    finishReason="completed",
                ),
            ),
        ]
    )
    plane = _plane(tmp_path, coding, [_clean()])

    rejected = await ControlPlane._project_clinecore_round(plane, _state())

    assert rejected["project_staged"] == {}
    assert rejected["project_coding_session_id"] == "coding_one"
    assert rejected["project_chain_index"] == 1
    assert rejected["project_coding_findings"] == [
        {
            "path": "app/main.py",
            "error": syntax_reason,
            "severity": "error",
            "kind": "import_rejection",
            "rung": "syntax",
        }
    ]
    assert coding.released == []
    [fallback] = [
        payload for kind, payload in plane.events.items if kind == "run.model_fallback"
    ]
    assert fallback["reason"] == "repairable_import_rejection"
    assert fallback["path"] == "app/main.py"

    repaired = await ControlPlane._project_clinecore_round(
        plane, {**_state(), **rejected}
    )

    assert len(coding.recover_calls) == 1
    assert coding.continue_calls[0]["session_id"] == "coding_one"
    assert coding.continue_calls[0]["provider"].model_id == "kimi-k3:cloud"
    assert "app/main.py" in coding.continue_calls[0]["prompt"]
    assert repaired["project_staged"] == valid
    assert "passed 1 independent verification" in repaired["response_text"]


@pytest.mark.asyncio
async def test_max_iterations_without_model_writes_switches_coder_for_the_retry(
    tmp_path: Path,
) -> None:
    """A round that spent its whole budget and wrote nothing is not proof the
    slice is impossible -- it is proof THIS coder, on THIS attempt, could
    not act. With another rung configured, the retry switches coder rather
    than silently repeating the identical attempt or giving up after one try."""
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            {},
            cleanup_sidecar_ids=("sidecar_one",),
            result=SliceResultV1(
                sessionId="sidecar_one",
                state="failed",
                finishReason="error",
                controlledStopReason="max_iterations",
                summary="Stopped at the configured boundary.",
            ),
        )
    )
    verifications = [_clean()]
    plane = _plane(tmp_path, coding, verifications)

    result = await ControlPlane._project_clinecore_round(plane, _state())

    assert coding.released == [("coding_one", CodingSessionState.FAILED)]
    assert result["project_chain_index"] == 1
    assert result["project_coding_session_id"] == ""
    assert "response_text" not in result
    [failure] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.coding_failed"
    ]
    assert failure["finish_reason"] == "error"
    assert failure["controlled_stop_reason"] == "max_iterations"
    [fallback] = [
        payload for kind, payload in plane.events.items if kind == "run.model_fallback"
    ]
    assert fallback["reason"] == "controlled_budget_stop_no_write"
    assert fallback["from"] == "deepseek-v4-pro:cloud"
    assert fallback["to"] == "kimi-k3:cloud"


@pytest.mark.asyncio
async def test_max_iterations_without_model_writes_fails_honestly_when_ladder_exhausted(
    tmp_path: Path,
) -> None:
    """The genuine stop-honestly case: no other coder is configured to try."""
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            {},
            cleanup_sidecar_ids=("sidecar_one",),
            result=SliceResultV1(
                sessionId="sidecar_one",
                state="failed",
                finishReason="error",
                controlledStopReason="max_iterations",
                summary="Stopped at the configured boundary.",
            ),
        )
    )
    verifications = [_clean()]
    plane = _plane(tmp_path, coding, verifications)
    state = _state(
        model_aliases={
            **_state()["model_aliases"],
            "_chain_coder": json.dumps(
                [{"provider": "local", "model": "deepseek-v4-pro:cloud"}]
            ),
        }
    )

    with pytest.raises(ProjectCodingError, match="controlled iteration limit"):
        await ControlPlane._project_clinecore_round(plane, state)

    assert coding.released == [("coding_one", CodingSessionState.FAILED)]
    assert verifications == [_clean()]
    assert not any(kind == "run.model_fallback" for kind, _ in plane.events.items)
    [failure] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.coding_failed"
    ]
    assert failure["finish_reason"] == "error"
    assert failure["controlled_stop_reason"] == "max_iterations"


@pytest.mark.asyncio
async def test_recovered_max_iterations_bytes_are_verified_without_another_call(
    tmp_path: Path,
) -> None:
    content = "READY = True\n"
    staged = {
        "app/main.py": {
            "content": content,
            "bytes": len(content.encode("utf-8")),
        }
    }
    coding = CodingCoordinator()
    coding.recovery = _round(
        "coding_one",
        staged,
        recovered=True,
        settled_recovery=True,
        finish_reason="error",
        controlled_stop_reason="max_iterations",
    )
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(plane, _state())

    assert "controlled iteration limit" in result["response_text"]
    assert result["project_staged"] == staged
    assert coding.start_calls == []
    assert coding.continue_calls == []


@pytest.mark.asyncio
async def test_completed_cline_round_may_finish_as_an_honest_no_write(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            {},
            result=SliceResultV1(sessionId="sidecar_one", state="completed"),
        )
    )
    plane = _plane(tmp_path, coding, [])

    result = await ControlPlane._project_clinecore_round(plane, _state())

    assert "without producing an application file change" in result["response_text"]
    assert result["project_coding_session_id"] == "coding_one"
    assert coding.released == []
    assert not any(kind == "project.coding_failed" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_drive_commits_failed_not_completed_for_an_aborted_cline_round(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_one",
            {},
            result=SliceResultV1(
                sessionId="sidecar_one",
                state="aborted",
                summary="A read tool was denied.",
            ),
        )
    )
    plane = _plane(tmp_path, coding, [])
    plane._shutting_down = False
    statuses: list[tuple[RunStatus, dict[str, Any]]] = []

    class Database:
        async def set_run_status(
            self,
            run_id: str,
            status: RunStatus,
            **kwargs: Any,
        ) -> None:
            del run_id
            statuses.append((status, kwargs))

    class Graph:
        async def ainvoke(
            self, graph_input: dict[str, Any], *, config: Any
        ) -> dict[str, Any]:
            del config
            return await ControlPlane._project_clinecore_round(plane, graph_input)

    plane.database = Database()
    plane.graph = Graph()
    plane._config = lambda conversation_id, run_id: {  # type: ignore[method-assign]
        "conversation_id": conversation_id,
        "run_id": run_id,
    }

    await ControlPlane._drive(
        plane,
        "run_x",
        "conv_x",
        _state(),  # type: ignore[arg-type]
    )

    assert [status for status, _ in statuses] == [RunStatus.RUNNING, RunStatus.FAILED]
    assert "ended aborted before" in statuses[-1][1]["error"]
    assert [kind for kind, _ in plane.events.items][-1] == "run.failed"
    assert not any(kind == "run.completed" for kind, _ in plane.events.items)


@pytest.mark.asyncio
async def test_pre_model_terminal_failure_advances_kimi_k3_to_k2_7(
    tmp_path: Path,
) -> None:
    usage_error = (
        "this model uses extra usage only (not included plan usage) and your "
        "extra usage balance is empty, add extra usage or turn on auto reload"
    )
    staged = {"app/main.py": {"content": "FIXED = True\n", "bytes": 13}}
    finding = {
        "path": "app/main.py",
        "error": "return-value: incompatible return value type",
        "severity": "error",
        "kind": "static",
        "rung": "mypy",
    }
    aliases = {
        "_project_id": "asset_x",
        "_coding_engine": "clinecore",
        "_provider": "local",
        "_chain_coder": json.dumps(
            [
                {"provider": "local", "model": "deepseek-v4-pro:cloud"},
                {"provider": "local", "model": "kimi-k3:cloud"},
                {"provider": "local", "model": "kimi-k2.7-code:cloud"},
            ]
        ),
    }
    coding = CodingCoordinator()
    coding.next_rounds.extend(
        [
            _round(
                "coding_k3",
                staged,
                cleanup_sidecar_ids=("sidecar_k3",),
                result=SliceResultV1(
                    sessionId="sidecar_k3",
                    state="failed",
                    summary=usage_error,
                ),
            ),
            _round(
                "coding_k2_7",
                staged,
                result=SliceResultV1(
                    sessionId="sidecar_k2_7",
                    state="completed",
                ),
            ),
        ]
    )
    plane = _plane(tmp_path, coding, [_clean()])
    repair_state = _state(
        model_aliases=aliases,
        project_staged=staged,
        project_coding_session_id="coding_deepseek",
        project_coding_rounds=1,
        project_coding_findings=[finding],
    )

    fallback_state = await ControlPlane._project_clinecore_round(plane, repair_state)

    assert coding.continue_calls[0]["provider"].model_id == "kimi-k3:cloud"
    assert fallback_state["project_chain_index"] == 2
    assert fallback_state["project_coding_session_id"] == ""
    assert fallback_state["project_staged"] == staged
    assert "response_text" not in fallback_state
    assert coding.released == [("coding_k3", CodingSessionState.FAILED)]
    [fallback] = [
        payload for kind, payload in plane.events.items if kind == "run.model_fallback"
    ]
    assert fallback["from"] == "kimi-k3:cloud"
    assert fallback["to"] == "kimi-k2.7-code:cloud"
    assert fallback["reason"] == "rate_limited"
    assert fallback["terminal_state"] == "failed"
    assert fallback["session_id"] == "coding_k3"
    assert not any(kind == "project.coding_failed" for kind, _ in plane.events.items)

    completed = await ControlPlane._project_clinecore_round(
        plane, {**repair_state, **fallback_state}
    )

    assert coding.start_calls[0]["provider"].model_id == "kimi-k2.7-code:cloud"
    assert "passed 1 independent verification" in completed["response_text"]


@pytest.mark.asyncio
async def test_drive_releases_no_write_session_only_after_terminal_commit(
    tmp_path: Path,
) -> None:
    del tmp_path
    order: list[str] = []
    scheduled: list[Any] = []
    plane = object.__new__(ControlPlane)
    plane._shutting_down = False

    class Database:
        async def set_run_status(
            self,
            run_id: str,
            status: RunStatus,
            **kwargs: Any,
        ) -> None:
            del run_id, kwargs
            order.append(f"status:{status.value}")

    class Graph:
        async def ainvoke(self, graph_input: Any, *, config: Any) -> dict[str, Any]:
            del graph_input, config
            order.append("graph:return")
            return {
                "response_text": "Nothing was applied.",
                "artifacts": [],
                "proposal": {},
                "project_coding_session_id": "coding_one",
                "project_coding_cleanup_ids": ["sidecar_parent", "sidecar_child"],
            }

    class OrderedEvents:
        async def emit(
            self,
            run_id: str,
            conversation_id: str,
            event_type: str,
            payload: dict[str, Any],
        ) -> None:
            del run_id, conversation_id, payload
            order.append(f"event:{event_type}")

    async def release(
        state: dict[str, Any],
        session_id: str,
        terminal_state: CodingSessionState,
    ) -> None:
        assert state["project_coding_cleanup_ids"] == [
            "sidecar_parent",
            "sidecar_child",
        ]
        assert session_id == "coding_one"
        assert terminal_state is CodingSessionState.COMPLETED
        order.append("release")

    plane.database = Database()
    plane.graph = Graph()
    plane.events = OrderedEvents()
    plane._config = lambda conversation_id, run_id: {  # type: ignore[method-assign]
        "conversation_id": conversation_id,
        "run_id": run_id,
    }
    plane._release_project_coding_session = release  # type: ignore[method-assign]

    async def cleanup(state: dict[str, Any], session_id: str) -> None:
        await release(state, session_id, CodingSessionState.COMPLETED)

    plane._cleanup_completed_no_approval_session = cleanup  # type: ignore[method-assign]

    def spawn_maintenance(work: Any, *, name: str) -> None:
        assert name == "metis-coding-cleanup-coding_one"
        order.append("cleanup:scheduled")
        scheduled.append(work)

    plane._spawn_maintenance = spawn_maintenance  # type: ignore[method-assign]

    await ControlPlane._drive(
        plane,
        "run_x",
        "conv_x",
        {"prompt": "build"},  # type: ignore[arg-type]
    )

    assert order.index("status:completed") < order.index("cleanup:scheduled")
    assert order.index("event:run.completed") < order.index("cleanup:scheduled")
    assert "release" not in order
    await scheduled[0]
    assert order.index("cleanup:scheduled") < order.index("release")


@pytest.mark.asyncio
async def test_provider_wide_cap_skips_same_provider_models_and_restarts_local(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(
        _round(
            "coding_capped",
            {},
            engine_error="INFERENCE_CAP_ERROR: weekly ClinePass limit reached",
            cleanup_sidecar_ids=("sidecar_parent", "sidecar_expected_child"),
        )
    )
    coding.release_error = RuntimeError("parent deletion failed")
    plane = _plane(tmp_path, coding, [])
    aliases = {
        "_project_id": "asset_x",
        "_coding_engine": "clinecore",
        "_provider": "cline",
        "_chain_coder": json.dumps(
            [
                {"provider": "cline", "model": "cline-pass/deepseek-v4-pro"},
                {"provider": "cline", "model": "cline-pass/kimi-k3"},
                {"provider": "local", "model": "deepseek-v4-pro:cloud"},
            ]
        ),
    }

    result = await ControlPlane._project_clinecore_round(
        plane, _state(model_aliases=aliases), prompt_context={"repo_map": "map"}
    )

    assert result["project_chain_index"] == 2
    assert result["project_coding_session_id"] == ""
    assert "response_text" not in result
    assert coding.released == [("coding_capped", CodingSessionState.FAILED)]
    assert coding.release_cleanup_ids == [("sidecar_parent", "sidecar_expected_child")]
    fallback = next(
        payload for kind, payload in plane.events.items if kind == "run.model_fallback"
    )
    assert fallback["reason"] == "provider_exhausted"
    assert fallback["skipped"] == ["Cline (cline-pass/kimi-k3)"]
    cleanup_failure = next(
        payload
        for kind, payload in plane.events.items
        if kind == "project.coding_cleanup_failed"
    )
    assert cleanup_failure["reason"] == "parent deletion failed"


# ── Attributed acceptance repairs ──────────────────────────────────────────
# A final acceptance failure used to be repaired by replaying the whole
# request: the follow-up reset the verified frontier, so every clean slice was
# planned and coded again to fix one file. These prove one defect now reopens
# exactly one slice and leaves every other verified byte alone.


def _repair_plan() -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    planned = [
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
        "app/static/app.js",
        "app/static/index.html",
    ]
    slices = [
        {
            "name": "SQLite-backed persistence",
            "outcome": "GET /api/projects reads from SQLite.",
            "files": planned[:3],
            "owned_files": planned[:3],
            "integration_files": [],
            "scenario_names": ["GET lists persisted projects"],
        },
        {
            "name": "Status workflow and UI",
            "outcome": "PATCH persists a status change and the UI acts on it.",
            "files": ["app/static/app.js", "app/static/index.html"] + planned[1:3],
            "owned_files": ["app/static/app.js", "app/static/index.html"],
            "integration_files": planned[1:3],
            "scenario_names": ["PATCH changes status", "GET reflects patched status"],
        },
    ]
    scenarios = [
        {
            "name": name,
            "method": "GET",
            "path": "/api/projects",
            "body_kind": "none",
            "body": {},
            "expect_status": "2xx",
        }
        for name in (
            "GET lists persisted projects",
            "PATCH changes status",
            "GET reflects patched status",
        )
    ]
    return planned, slices, scenarios


def _verified_staged() -> dict[str, dict[str, Any]]:
    content = {
        "app/db.py": "SEED = 'harbor migration'\n",
        "app/main.py": "APP = True\n",
        "tests/test_workflows.py": "def test_x(): ...\n",
        "app/static/app.js": "fetch('/api/projects');\n",
        "app/static/index.html": "<html></html>\n",
    }
    return {
        path: {"content": value, "bytes": len(value)} for path, value in content.items()
    }


def _repair_state(target: str, slice_index: int, **overrides: Any) -> dict[str, Any]:
    planned, slices, scenarios = _repair_plan()
    staged = _verified_staged()
    item = slices[slice_index]
    authorized = [str(path) for path in item["files"]]
    return _state(
        project_planned_files=planned,
        project_planned_slices=slices,
        project_planned_scenarios=scenarios,
        project_staged=staged,
        # The build really did verify every slice; only final acceptance failed.
        project_verified_prefix=len(planned),
        project_repair_slice={
            "name": item["name"],
            "outcome": item["outcome"],
            "files": authorized,
            "owned_files": list(item["owned_files"]),
            "integration_files": list(item["integration_files"]),
            "scenario_names": [
                "GET lists persisted projects",
                "PATCH changes status",
                "GET reflects patched status",
            ],
            "target": target,
            "slice_index": slice_index,
            "attribution": (
                "integration_file"
                if target in item["integration_files"]
                else "owned_file"
            ),
            "target_hashes": {
                path: _sha256(str(staged[path]["content"])) for path in authorized
            },
        },
        project_repair_hashes={
            path: _sha256(str(entry["content"]))
            for path, entry in staged.items()
            if path not in set(authorized)
        },
        **overrides,
    )


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_an_owned_file_repair_reopens_one_slice_and_recodes_no_other(
    tmp_path: Path,
) -> None:
    staged = _verified_staged()
    repaired = {
        **staged,
        "app/db.py": {"content": "SEED = 'Harbor Migration'\n", "bytes": 26},
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", repaired))
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane, _repair_state("app/db.py", 0)
    )

    # Exactly one coding session, scoped to slice 1's own declared files --
    # slice 2's UI files are never authorized and never recoded.
    assert len(coding.start_calls) == 1
    assert coding.continue_calls == []
    assert coding.start_calls[0]["allowed_paths"] == [
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
    ]
    assert "app/static/app.js" not in coding.start_calls[0]["allowed_paths"]
    assert "app/static/index.html" not in coding.start_calls[0]["allowed_paths"]
    # The repair converged and cleared its own routing state.
    assert result["project_repair_slice"] == {}
    assert result["project_repair_hashes"] == {}
    assert result["project_verified_prefix"] == 5
    assert "app/db.py" in result["response_text"]
    assert "byte-identical" in result["response_text"]


@pytest.mark.asyncio
async def test_a_shared_integration_file_repair_uses_the_later_slice_scope(
    tmp_path: Path,
) -> None:
    staged = _verified_staged()
    repaired = {**staged, "app/main.py": {"content": "APP = 'patched'\n", "bytes": 17}}
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", repaired))
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane, _repair_state("app/main.py", 1)
    )

    # app/main.py is owned by slice 1 but explicitly reopened by slice 2, so
    # the repair runs under slice 2's scope -- which authorizes its own UI
    # files, not slice 1's exclusive app/db.py.
    assert coding.start_calls[0]["allowed_paths"] == [
        "app/static/app.js",
        "app/static/index.html",
        "app/main.py",
        "tests/test_workflows.py",
    ]
    assert "app/db.py" not in coding.start_calls[0]["allowed_paths"]
    [checked] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.vertical_slice_checked"
    ]
    assert checked["repair"] is True
    assert checked["repair_attribution"] == "integration_file"
    assert checked["repair_target"] == "app/main.py"
    assert "response_text" in result


@pytest.mark.asyncio
async def test_every_downstream_scenario_is_replayed_after_an_earlier_slice_repair(
    tmp_path: Path,
) -> None:
    staged = _verified_staged()
    repaired = {
        **staged,
        "app/db.py": {"content": "SEED = 'Harbor Migration'\n", "bytes": 26},
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", repaired))
    captured: list[dict[str, Any]] = []

    async def verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        captured.append(kwargs)
        return _clean()

    plane = _plane(tmp_path, coding, [])
    plane._verify_staged_changeset = verify

    await ControlPlane._project_clinecore_round(plane, _repair_state("app/db.py", 0))

    [call] = captured
    # A slice-1 repair can regress slice 1's own claim and everything slice 2
    # built on it, so the whole proven set is replayed -- not just slice 1's.
    assert [item["name"] for item in call["scenarios"]] == [
        "GET lists persisted projects",
        "PATCH changes status",
        "GET reflects patched status",
    ]
    # Verification covers the complete staged changeset, not the repair scope.
    assert call["planned"] == [
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
        "app/static/app.js",
        "app/static/index.html",
    ]


@pytest.mark.asyncio
async def test_unaffected_owned_files_stay_byte_identical_across_a_repair(
    tmp_path: Path,
) -> None:
    staged = _verified_staged()
    repaired = {
        **staged,
        "app/db.py": {"content": "SEED = 'Harbor Migration'\n", "bytes": 26},
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", repaired))
    plane = _plane(tmp_path, coding, [_clean()])
    state = _repair_state("app/db.py", 0)
    before = dict(state["project_repair_hashes"])

    result = await ControlPlane._project_clinecore_round(plane, state)

    after = result["project_staged"]
    assert set(before) == {"app/static/app.js", "app/static/index.html"}
    for path, digest in before.items():
        assert _sha256(str(after[path]["content"])) == digest
    assert "response_text" in result


@pytest.mark.asyncio
async def test_a_repair_that_rewrites_an_unauthorized_file_never_approves(
    tmp_path: Path,
) -> None:
    staged = _verified_staged()
    overreached = {
        **staged,
        "app/db.py": {"content": "SEED = 'Harbor Migration'\n", "bytes": 26},
        # Slice 2's file, which this slice-1 repair was never authorized to
        # touch. Clean verification must not turn this into an approval.
        "app/static/app.js": {"content": "// rewritten\n", "bytes": 14},
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", overreached))
    plane = _plane(tmp_path, coding, [_clean()])
    state = _repair_state("app/db.py", 0)

    result = await ControlPlane._project_clinecore_round(plane, state)

    assert "did not converge" in result["response_text"]
    assert "not authorized to write" in result["response_text"]
    # The previously verified overlay survives untouched, and the failed
    # attempt's session is released rather than carried into an approval.
    assert result["project_staged"] == state["project_staged"]
    assert result["project_repair_slice"] == {}
    assert coding.released == [("coding_repair", CodingSessionState.FAILED)]
    [rejected] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.repair_rejected"
    ]
    assert rejected["unaffected_changed"] == ["app/static/app.js"]


@pytest.mark.asyncio
async def test_a_repair_that_changes_no_authorized_byte_earns_no_credit(
    tmp_path: Path,
) -> None:
    # The session settled and verification is clean only because the build was
    # already clean structurally -- the acceptance defect is untouched.
    staged = _verified_staged()
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_repair", dict(staged)))
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane, _repair_state("app/db.py", 0)
    )

    assert "did not converge" in result["response_text"]
    assert "no change to any file it was authorized to write" in result["response_text"]
    assert result["project_repair_slice"] == {}
    [rejected] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.repair_rejected"
    ]
    assert rejected["authorized_changed"] == []


@pytest.mark.asyncio
async def test_a_recovered_repair_round_does_not_repeat_the_model_call(
    tmp_path: Path,
) -> None:
    # The local process died after the model wrote but before Metis
    # checkpointed. Recovery must import those exact bytes, not start a second
    # repair session over the same overlay.
    staged = _verified_staged()
    repaired = {
        **staged,
        "app/db.py": {"content": "SEED = 'Harbor Migration'\n", "bytes": 26},
    }
    coding = CodingCoordinator()
    coding.recovery = _round("coding_repair", repaired, recovered=True)
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane, _repair_state("app/db.py", 0)
    )

    assert coding.start_calls == []
    assert coding.continue_calls == []
    assert [call["run_id"] for call in coding.recover_calls] == ["run_x"]
    # Recovery is still scoped to the routed repair, never the whole plan.
    assert coding.recover_calls[0]["allowed_paths"] == [
        "app/db.py",
        "app/main.py",
        "tests/test_workflows.py",
    ]
    assert result["project_repair_slice"] == {}
    assert "app/db.py" in result["response_text"]


# ── A repair round that writes nothing ─────────────────────────────────────
# The retained Atlas defect: slice 1 staged index.html and styles.css, and
# verification found `legacy-shell` referenced by a file already in the
# project with no rule to render against. The repair round must either fix it
# or be told, in terms, that it changed nothing -- never be credited as
# convergence.

LEGACY_SHELL_FINDING = {
    "path": "app/static/styles.css",
    "error": (
        "1 class(es) still undefined (legacy-shell…) — used by 1 file(s) "
        "already in this project. This change does not yet give them a "
        "stylesheet to render against."
    ),
    "severity": "error",
    "kind": "",
}

ATLAS_SLICE_ONE = ["app/static/index.html", "app/static/styles.css"]
ATLAS_STAGED = {
    "app/static/index.html": {
        "content": '<html><body class="legacy-shell"></body></html>\n',
        "bytes": 46,
    },
    "app/static/styles.css": {"content": "body { color: #111; }\n", "bytes": 22},
    "app/static/app.js": {"content": "// untouched\n", "bytes": 13},
}
ATLAS_REPAIRED_CSS = "body { color: #111; }\n.legacy-shell { display: block; }\n"


def _atlas_state(**overrides: Any) -> dict[str, Any]:
    return _state(
        project_planned_files=ATLAS_SLICE_ONE + ["app/static/app.js"],
        project_planned_slices=[
            {
                "name": "Console shell and styling",
                "outcome": "The console page loads and is styled.",
                "files": ATLAS_SLICE_ONE,
                "owned_files": ATLAS_SLICE_ONE,
                "integration_files": [],
                "scenario_names": ["console page loads"],
            },
            {
                "name": "Data and interaction",
                "outcome": "The table renders.",
                "files": ["app/static/app.js"],
                "owned_files": ["app/static/app.js"],
                "integration_files": [],
                "scenario_names": ["table renders"],
            },
        ],
        project_staged=dict(ATLAS_STAGED),
        project_coding_session_id="coding_atlas",
        project_coding_rounds=1,
        project_coding_slice_rounds=1,
        project_coding_findings=[LEGACY_SHELL_FINDING],
        project_coding_finding_signature=_verifier_finding_signature(
            [LEGACY_SHELL_FINDING]
        ),
        **overrides,
    )


@pytest.mark.asyncio
async def test_a_repair_that_writes_nothing_is_not_convergence_and_switches_rung(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    # The exact live shape: the repair round returns the SAME bytes.
    coding.next_rounds.append(_round("coding_atlas", dict(ATLAS_STAGED)))
    plane = _plane(
        tmp_path,
        coding,
        [{"errors": [LEGACY_SHELL_FINDING], "warnings": [], "checks": ["wiring"]}],
    )

    result = await ControlPlane._project_clinecore_round(plane, _atlas_state())

    [no_change] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.repair_no_change"
    ]
    assert no_change["authorized_paths"] == ATLAS_SLICE_ONE
    assert no_change["finding_unchanged"] is True
    assert no_change["retries_spent"] == 0
    # Not convergence: the slice is not complete and the frontier does not move.
    assert result.get("project_coding_slice_complete") is not True
    assert "project_verified_prefix" not in result
    # One bounded corrective continuation, on the next configured rung, with
    # explicit feedback that no change was detected.
    assert result["project_chain_index"] == 1
    assert result["project_repair_no_change"] == 1
    assert result["project_repair_context"]["no_change_detected"] is True
    assert result["project_coding_findings"] == [LEGACY_SHELL_FINDING]


@pytest.mark.asyncio
async def test_the_corrective_repair_fixes_the_class_and_the_slice_reverifies(
    tmp_path: Path,
) -> None:
    """Second attempt defines the class; the same slice is verified again."""

    repaired = {
        **ATLAS_STAGED,
        "app/static/styles.css": {"content": ATLAS_REPAIRED_CSS, "bytes": 55},
    }
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_atlas", repaired))
    plane = _plane(tmp_path, coding, [_clean()])

    result = await ControlPlane._project_clinecore_round(
        plane,
        _atlas_state(project_repair_no_change=1, project_chain_index=1),
    )

    # Verified immediately after the edit, and the slice now completes.
    assert not any(kind == "project.repair_no_change" for kind, _ in plane.events.items)
    [checked] = [
        payload
        for kind, payload in plane.events.items
        if kind == "project.vertical_slice_checked"
    ]
    assert checked["errors"] == 0
    assert checked["files"] == ATLAS_SLICE_ONE
    assert result["project_coding_slice_complete"] is True
    # Only after clean verification does the frontier reach the next slice.
    assert result["project_verified_prefix"] == 2
    # The repaired byte is the staged byte; the untouched file is identical.
    assert result["project_staged"]["app/static/styles.css"]["content"] == (
        ATLAS_REPAIRED_CSS
    )
    assert (
        result["project_staged"]["app/static/app.js"]
        == ATLAS_STAGED["app/static/app.js"]
    )


@pytest.mark.asyncio
async def test_a_second_no_change_round_stops_honestly_without_looping(
    tmp_path: Path,
) -> None:
    coding = CodingCoordinator()
    coding.next_rounds.append(_round("coding_atlas", dict(ATLAS_STAGED)))
    plane = _plane(
        tmp_path,
        coding,
        [{"errors": [LEGACY_SHELL_FINDING], "warnings": [], "checks": ["wiring"]}],
    )
    # The retry is spent and this is the last rung.
    state = _atlas_state(
        project_repair_no_change=1,
        project_chain_index=1,
        model_aliases={
            **_state()["model_aliases"],
            "_chain_coder": json.dumps(
                [{"provider": "local", "model": "kimi-k2.7-code:cloud"}]
            ),
        },
    )

    result = await ControlPlane._project_clinecore_round(plane, state)

    assert "changed none of the" in result["response_text"]
    assert "stopped rather than repeat" in result["response_text"]
    # The overlay is preserved exactly, and nothing is credited.
    assert result["project_staged"] == ATLAS_STAGED
    assert result.get("project_coding_slice_complete") is not True
    assert "project_verified_prefix" not in result
