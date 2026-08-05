"""P1.3: an undecided build changeset follows the conversation into repair.

The failure this pins shut: a build ends at the approval card, the user sends
a follow-up ("fix the duplicate check"), and the new run — a new checkpoint
thread — sees only the disk. The model then repairs files it cannot read,
and the exact bytes verification inspected are unreachable. Carrying the
newest undecided project changeset into the follow-up run's overlay is the
whole feature; everything else here is the guard rails around it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from waqil_api.contracts import ApprovalRequestV1, RiskLevel, RunStatus
from waqil_api.control_plane import ControlPlane
from waqil_api.database import Database

STAGED = {
    "app/main.py": {"content": "x = 1\n", "origin": "create", "base_sha256": "", "bytes": 6},
    "appkit/money.py": {"content": "CENT = 1\n", "origin": "create", "base_sha256": "", "bytes": 9},
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

    async def emit(self, run_id: str, conversation_id: str, name: str, payload: dict) -> None:
        self.emitted.append((name, payload))


def _plane(
    *,
    prior: tuple[str, str] | None = ("run_prev", "asset_x"),
    approval: ApprovalRequestV1 | None = None,
    staged: dict | None = None,
) -> tuple[ControlPlane, _Events]:
    plane = object.__new__(ControlPlane)
    events = _Events()
    plane.events = events
    plane.settings = SimpleNamespace(project_agent_max_steps=48)
    plane.database = SimpleNamespace(
        latest_awaiting_project_approval=lambda conversation_id: _async(prior),
        get_pending_approval=lambda run_id: _async(approval),
    )
    plane.checkpointer = SimpleNamespace(
        aget_tuple=lambda config: _async(
            SimpleNamespace(
                checkpoint={"channel_values": {"project_staged": staged or {}}}
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


@pytest.mark.asyncio
async def test_the_pending_changeset_rides_into_the_follow_up_run() -> None:
    plane, events = _plane(approval=_approval(), staged=STAGED)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())

    assert seeded["project_staged"] == STAGED
    (note,) = seeded["project_trace"]
    assert note["tool"] == "resume_staged"
    assert note["result"]["carried_files"] == sorted(STAGED)
    assert "missing" in note["result"]["blocked_reason"]
    assert events.emitted == [
        ("project.staged_resumed", {"files": sorted(STAGED), "from_run": "run_prev"})
    ]


@pytest.mark.asyncio
async def test_nothing_is_carried_across_projects_kinds_or_empty_state() -> None:
    # A changeset staged for another project stays there.
    plane, _ = _plane(prior=("run_prev", "asset_OTHER"), approval=_approval(), staged=STAGED)
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(plane, _state())

    # A pending approval of a different kind is not a build changeset.
    plane, _ = _plane(approval=_approval(kind="project_verify"), staged=STAGED)
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(plane, _state())

    # No prior run parked at approval — nothing to carry.
    plane, _ = _plane(prior=None)
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(plane, _state())

    # An empty overlay carries nothing.
    plane, _ = _plane(approval=_approval(), staged={})
    assert "project_staged" not in await ControlPlane._carry_pending_overlay(plane, _state())


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
    counted.blocked_reason = "4 problem(s) would stop this project working — app/x.py: boom."
    plane, _ = _plane(approval=counted, staged=STAGED)
    seeded = await ControlPlane._carry_pending_overlay(plane, _state())
    assert seeded["project_prior_blocking"] == 4
