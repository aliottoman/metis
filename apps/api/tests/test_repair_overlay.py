"""P1.3: an undecided build changeset follows the conversation into repair.

The failure this pins shut: a build ends at the approval card, the user sends
a follow-up ("fix the duplicate check"), and the new run — a new checkpoint
thread — sees only the disk. The model then repairs files it cannot read,
and the exact bytes verification inspected are unreachable. Carrying the
newest undecided project changeset into the follow-up run's overlay is the
whole feature; everything else here is the guard rails around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.contracts import (
    ApprovalRequestV1,
    ProjectAgentStepV1,
    ProjectToolCallV1,
    RiskLevel,
    RunStatus,
)
from waqil_api.control_plane import ControlPlane
from waqil_api.database import Database

STAGED = {
    "app/main.py": {
        "content": "x = 1\n",
        "origin": "create",
        "base_sha256": "",
        "bytes": 6,
    },
    "appkit/money.py": {
        "content": "CENT = 1\n",
        "origin": "create",
        "base_sha256": "",
        "bytes": 9,
    },
}

SECOND = {
    "path": "app/other.py",
    "error": "NameError: missing_name is not defined",
    "severity": "error",
    "kind": "wiring",
    "rung": "wiring",
}
FIRST = {
    "path": "app/main.py",
    "error": "SyntaxError: invalid syntax",
    "severity": "error",
    "kind": "syntax",
    "rung": "syntax",
}


def _approval(kind: str = "project_apply_build") -> ApprovalRequestV1:
    return ApprovalRequestV1(
        id="appr_1",
        run_id="run_prev",
        action_id="act_1",
        kind=kind,  # type: ignore[arg-type]
        title="Apply staged build",
        summary="2 files",
        risk_level=RiskLevel.R3,
        input_digest="digest",
        blocked_reason="app/main.py: name 'missing' is not defined",
    )


class _Events:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, run_id: str, conversation_id: str, name: str, payload: dict
    ) -> None:
        self.emitted.append((name, payload))


def _plane(
    *,
    prior: tuple[str, str] | None = ("run_prev", "asset_x"),
    approval: ApprovalRequestV1 | None = None,
    staged: dict | None = None,
    checkpoint_values: dict[str, Any] | None = None,
) -> tuple[ControlPlane, _Events]:
    plane = object.__new__(ControlPlane)
    events = _Events()
    plane.events = events
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        cline_coder_model="cline-pass/deepseek-v4-pro",
    )
    plane.database = SimpleNamespace(
        latest_awaiting_project_approval=lambda conversation_id: _async(prior),
        get_pending_approval=lambda run_id: _async(approval),
    )
    plane.checkpointer = SimpleNamespace(
        aget_tuple=lambda config: _async(
            SimpleNamespace(
                checkpoint={
                    "channel_values": checkpoint_values
                    if checkpoint_values is not None
                    else {"project_staged": staged or {}}
                }
            )
        )
    )
    return plane, events


async def _async(value):  # noqa: ANN001, ANN202 - tiny awaitable helper
    return value


def _state() -> dict:
    return {
        "run_id": "run_new",
        "conversation_id": "conv_1",
        "model_aliases": {"_project_id": "asset_x"},
    }


def _repair_checkpoint() -> dict[str, Any]:
    staged = {
        **STAGED,
        "app/other.py": {
            "content": "value = missing_name\n",
            "origin": "create",
            "base_sha256": "",
            "bytes": 21,
        },
    }
    return {
        "project_staged": staged,
        "project_context": {"manifest": {"file_tree": ["app/main.py", "app/other.py"]}},
        "project_planned_files": ["app/main.py", "app/other.py"],
        "project_planned_scenarios": [
            {"name": "health", "method": "GET", "path": "/health"}
        ],
        "project_plan_taken": True,
        "project_build_intent": "edit",
        "project_build_scope": "narrow",
        # This was successful work from the old run and must not be replayed.
        "project_direction": {"path": "app/other.py", "instruction": "stale"},
        "project_repair_context": {
            "files": ["app/main.py", "app/other.py"],
            "findings": [FIRST, SECOND],
        },
    }


@pytest.mark.asyncio
async def test_the_pending_changeset_rides_into_the_follow_up_run() -> None:
    plane, events = _plane(approval=_approval(), staged=STAGED)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())

    assert seeded["project_staged"] == STAGED
    (note,) = seeded["project_trace"]
    assert note["tool"] == "resume_staged"
    assert note["result"]["carried_files"] == sorted(STAGED)
    assert "missing" in note["result"]["blocked_reason"]
    assert note["result"]["verification_summary"] == "2 files"
    assert "complete findings" in note["result"]["note"]
    assert events.emitted == [
        ("project.staged_resumed", {"files": sorted(STAGED), "from_run": "run_prev"})
    ]


@pytest.mark.asyncio
async def test_the_complete_approval_summary_reaches_the_repair_trace() -> None:
    approval = _approval()
    approval.summary = "finding\n" * 1_200
    plane, _ = _plane(approval=approval, staged=STAGED)

    seeded = await ControlPlane._carry_pending_overlay(plane, _state())

    assert (
        seeded["project_trace"][0]["result"]["verification_summary"] == approval.summary
    )


@pytest.mark.asyncio
async def test_nothing_is_carried_across_projects_kinds_or_empty_state() -> None:
    # A changeset staged for another project stays there.
    plane, _ = _plane(
        prior=("run_prev", "asset_OTHER"), approval=_approval(), staged=STAGED
    )
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(
        plane, _state()
    )

    # A pending approval of a different kind is not a build changeset.
    plane, _ = _plane(approval=_approval(kind="project_verify"), staged=STAGED)
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(
        plane, _state()
    )

    # No prior run parked at approval — nothing to carry.
    plane, _ = _plane(prior=None)
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(
        plane, _state()
    )

    # An empty overlay carries nothing.
    plane, _ = _plane(approval=_approval(), staged={})
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(
        plane, _state()
    )


@pytest.mark.asyncio
async def test_runs_without_a_project_never_touch_the_database() -> None:
    plane, _ = _plane()
    plane.database = SimpleNamespace()  # any lookup would raise AttributeError
    state = {"run_id": "r", "conversation_id": "c", "model_aliases": {}}
    assert await ControlPlane._carry_pending_overlay(plane, state) == state


@pytest.mark.asyncio
async def test_database_returns_the_parked_run_with_its_own_project(tmp_path) -> None:
    database = Database(tmp_path / "waqil.db")
    await database.open()
    try:
        conversation = await database.create_conversation("Repair")
        message = await database.add_message(conversation.id, "user", "build it")
        run = await database.create_run(
            conversation.id,
            message.id,
            graph_schema_version="1",
            model_aliases={"_project_id": "asset_x", "_provider": "oci"},
        )
        assert await database.latest_awaiting_project_approval(conversation.id) is None

        await database.set_run_status(run.id, RunStatus.AWAITING_APPROVAL)
        found = await database.latest_awaiting_project_approval(conversation.id)
        assert found == (run.id, "asset_x")

        # Another conversation sees nothing of it.
        other = await database.create_conversation("Other")
        assert await database.latest_awaiting_project_approval(other.id) is None

        # Once decided, the run leaves awaiting_approval and the query moves on.
        await database.set_run_status(run.id, RunStatus.COMPLETED)
        assert await database.latest_awaiting_project_approval(conversation.id) is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_carried_changesets_finding_count_rides_along() -> None:
    """The repair's own card compares against what it inherited, so the count
    has to travel with the overlay — a live turn took a one-finding changeset
    to thirteen and the card had no memory of the one."""
    plane, _ = _plane(approval=_approval(), staged=STAGED)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())
    # _approval() carries a blocked_reason with no leading count, so nothing
    # is claimed; a real card's wording is parsed in test_repair_regression.
    assert "project_prior_blocking" in seeded

    counted = _approval()
    counted.blocked_reason = (
        "4 problem(s) would stop this project working — app/x.py: boom."
    )
    plane, _ = _plane(approval=counted, staged=STAGED)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())
    assert seeded["project_prior_blocking"] == 4


@pytest.mark.asyncio
async def test_structured_continuation_reuses_scope_and_pins_only_first_blocker() -> (
    None
):
    checkpoint = _repair_checkpoint()
    plane, events = _plane(approval=_approval(), checkpoint_values=checkpoint)
    state = _state()
    state["model_aliases"].update(
        {
            "_provider": "cline",
            "_chain_coder": json.dumps(
                [
                    {"provider": "cline", "model": "cline-pass/deepseek-v4-pro"},
                    {"provider": "cline", "model": "cline-pass/kimi-k3"},
                    {"provider": "cline", "model": "cline-pass/kimi-k2.7-code"},
                ]
            ),
        }
    )

    seeded = await ControlPlane._carry_pending_overlay(plane, state)

    assert seeded["project_plan_taken"] is True
    assert seeded["project_planned_files"] == ["app/main.py", "app/other.py"]
    assert seeded["project_planned_scenarios"] == [
        {
            "name": "health",
            "method": "GET",
            "path": "/health",
            "body_kind": "none",
            "body": {},
            "expect_status": "2xx_or_4xx",
            "expect_contains": [],
            # A carried scenario round-trips the strict-assertion fields too;
            # absent means "this scenario makes no exact claim".
            "expect_json_exact": None,
            "json_match": "exact",
        }
    ]
    assert seeded["project_direction"]["path"] == "app/main.py"
    assert seeded["project_focus_path"] == "app/main.py"
    assert seeded["project_write_pin"] == ["app/main.py"]
    assert seeded["project_repair_context"]["findings"] == [FIRST, SECOND]
    # Structured verifier repair bypasses the destructive broad-build primary.
    assert seeded["project_chain_index"] == 1
    resumed = [
        payload for kind, payload in events.emitted if kind == "project.repair_resumed"
    ]
    assert resumed == [
        {
            "from_run": "run_prev",
            "path": "app/main.py",
            "findings": 2,
            "coder_chain_index": 1,
            "coder": "Cline (cline-pass/kimi-k3)",
        }
    ]


@pytest.mark.asyncio
async def test_carried_repair_prefetches_exact_overlay_before_any_planner_call() -> (
    None
):
    checkpoint = _repair_checkpoint()
    plane, _ = _plane(approval=_approval(), checkpoint_values=checkpoint)
    state = _state()
    state.update(
        {
            "prompt": "Fix the verified blockers and keep the existing scope.",
            "project_iterations": 0,
            "model_aliases": {
                "_project_id": "asset_x",
                "_provider": "cline",
                "_chain_coder": json.dumps(
                    [
                        {
                            "provider": "cline",
                            "model": "cline-pass/deepseek-v4-pro",
                        },
                        {"provider": "cline", "model": "cline-pass/kimi-k3"},
                    ]
                ),
            },
        }
    )
    seeded = await ControlPlane._carry_pending_overlay(plane, state)
    calls: list[tuple[str, str]] = []
    requests: list[dict[str, Any]] = []

    class Model:
        async def project_spec(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("repair continuation must not rewrite the spec")

        async def project_plan_files(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("repair continuation must not re-plan")

        async def project_direction(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("the verifier already owns the repair direction")

        async def project_step(self, request, *, model_aliases=None):
            requests.append(request)
            calls.append(("coder", model_aliases["_cline_model"]))
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="apply_patch",
                    arguments={
                        "path": "app/main.py",
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    },
                ),
            )

    class Projects:
        async def context(self, project_id: str) -> dict[str, Any]:
            return {
                "manifest": {"file_tree": ["app/main.py", "app/other.py"]},
                "metis_md": "",
            }

        async def execute_staged(self, project_id, call, staged, owed):
            path = str(call.arguments["path"])
            calls.append(("prefetch", path))
            return {
                "path": path,
                "content": staged[path]["content"],
            }, staged

    async def noop(*args: object, **kwargs: object) -> None:
        return None

    plane.model = Model()
    plane.projects = Projects()
    plane._guard = noop
    plane._stage = noop
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        cline_coder_model="cline-pass/deepseek-v4-pro",
        project_spec_rewrite=True,
        project_spec_rewrite_max_chars=1_800,
        project_reference_enabled=False,
        project_reference_dir=Path("/none"),
        project_reference_max_chars=0,
        project_reference_max_chars_local=0,
        project_repo_map_enabled=False,
        project_orchestrator_enabled=True,
    )

    result = await ControlPlane._project_step(plane, seeded)

    assert calls == [
        ("prefetch", "app/main.py"),
        ("coder", "cline-pass/kimi-k3"),
    ]
    assert result["project_pending_call"]["arguments"]["path"] == "app/main.py"
    (request,) = requests
    prefetched = [
        item
        for item in request["tool_trace"]
        if item["tool"] == "read_file" and item["arguments"]["path"] == "app/main.py"
    ]
    assert prefetched[-1]["result"]["output"]["content"] == "x = 1\n"
    assert request["verification_repair"]["findings"] == [FIRST, SECOND]
    assert "app/main.py" in request["attention"]


@pytest.mark.asyncio
async def test_next_verification_moves_the_queue_to_one_remaining_file() -> None:
    checkpoint = _repair_checkpoint()
    plane, _ = _plane(approval=_approval(), checkpoint_values=checkpoint)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())
    plane.projects = SimpleNamespace(record_plan=_async)

    update = await ControlPlane._staged_verify_retry(
        plane,
        seeded,
        iterations=1,
        retries=0,
        errors=[SECOND],
        repair_files=["app/main.py", "app/other.py"],
    )

    assert update["project_direction"]["path"] == "app/other.py"
    assert update["project_focus_path"] == "app/other.py"
    assert update["project_write_pin"] == ["app/other.py"]
    assert update["project_repair_context"]["findings"] == [SECOND]


@pytest.mark.asyncio
async def test_host_owned_and_advisory_findings_never_seed_a_repair() -> None:
    checkpoint = _repair_checkpoint()
    checkpoint["project_repair_context"]["findings"] = [
        {
            "path": "appkit/config.py",
            "error": "SyntaxError: invalid syntax",
            "severity": "error",
            "kind": "syntax",
            "rung": "syntax",
        },
        {
            "path": "app/main.py",
            "error": "OCI_RESPONSES_BASE_URL is not set",
            "severity": "error",
            "kind": "import",
            "rung": "runtime",
        },
    ]
    plane, events = _plane(approval=_approval(), checkpoint_values=checkpoint)

    seeded = await ControlPlane._carry_pending_overlay(plane, _state())

    assert not seeded.get("project_repair_context")
    assert not seeded.get("project_direction")
    assert not seeded.get("project_focus_path")
    assert not seeded.get("project_write_pin")
    assert seeded.get("project_chain_index", 0) == 0
    assert not [kind for kind, _ in events.emitted if kind == "project.repair_resumed"]


@pytest.mark.asyncio
async def test_final_approval_checkpoints_only_structured_repairable_findings() -> None:
    plane = object.__new__(ControlPlane)
    events = _Events()
    plane.events = events

    class Projects:
        @staticmethod
        def staged_summary(staged):
            return "2 files", "digest", [{"path": path} for path in staged]

    class Database:
        @staticmethod
        async def create_approval(approval):
            return approval

    async def noop(*args: object, **kwargs: object) -> None:
        return None

    async def verify(*args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "errors": [
                FIRST,
                {
                    "path": "appkit/config.py",
                    "error": "SyntaxError: invalid syntax",
                    "severity": "error",
                    "kind": "syntax",
                    "rung": "syntax",
                },
                {
                    "path": "app/main.py",
                    "error": "OCI_RESPONSES_BASE_URL is not set",
                    "severity": "error",
                    "kind": "import",
                    "rung": "runtime",
                },
            ],
            "warnings": [],
            "notes": [],
            "checks": [{"kind": "syntax", "ok": False}],
        }

    async def policy(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(require_approval=lambda: None)

    plane.projects = Projects()
    plane.database = Database()
    plane._guard = noop
    plane._verify_staged_changeset = verify
    plane._policy_gate = policy
    state = {
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_staged": STAGED,
        "project_planned_files": ["app/main.py"],
        "project_planned_scenarios": [],
        "project_context": {
            "manifest": {"file_tree": ["app/main.py", "appkit/config.py"]}
        },
        "project_prior_blocking": 0,
    }

    result = await ControlPlane._project_prepare_build_approval(plane, state)

    assert result["project_repair_context"]["files"] == ["app/main.py"]
    assert result["project_repair_context"]["findings"] == [FIRST]
    assert result["approval_request"]["blocked_reason"].startswith("2 problem(s)")
