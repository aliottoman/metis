"""Two ways an evaluator can lie to itself, both taken from live probes.

The first: a repair inference reached ``iteration_start`` and never came back.
No round event was ever emitted, so every accounting path summed nothing and
the evidence read "0 tokens" -- a request that was made, billed by whoever
serves it, and reported as free.

The second: that same probe was killed by its own wall clock, and its sidecar
session and disposable mirror stayed on disk because cleanup only ran after a
successful round. Cleanup belongs in a ``finally``, and it must be scoped so
tightly that a crashing evaluator can never reach the session belonging to the
Metis app the developer is running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from waqil_api.project_capability_eval import (
    combined_observed_tokens,
    release_evaluation_artifacts,
    score_evaluation,
    summarize_coding_attempts,
)


class TimelineEvent:
    """The shape the evaluator reads: a durable event type and payload."""

    def __init__(self, type: str, payload: dict[str, Any]) -> None:
        self.type = type
        self.payload = payload


def _started(session_id: str) -> TimelineEvent:
    return TimelineEvent(
        "project.coding_started",
        {"engine": "clinecore", "session_id": session_id, "model": "glm-5.2:cloud"},
    )


def _round(session_id: str, *, total: int = 4_000) -> TimelineEvent:
    return TimelineEvent(
        "project.coding_round",
        {
            "session_id": session_id,
            "state": "completed",
            "changed_paths": ["app/static/styles.css"],
            "usage": {"totalTokens": total},
        },
    )


# ── An attempt that never settles is unknown, not free ─────────────────────


def test_a_started_round_that_never_returns_is_an_attempted_request() -> None:
    # Exactly the retained probe: session created, sidecar reached
    # iteration_start, the call never returned inside the wall clock.
    attempts = summarize_coding_attempts([_started("coding_1176849bd1d8")])

    assert attempts["started"] == 1
    assert attempts["attempted_requests"] == 1
    assert attempts["settled"] == 0
    assert attempts["unaccounted"] == 1
    assert attempts["unaccounted_sessions"] == ["coding_1176849bd1d8"]


def test_an_unsettled_attempt_makes_the_total_a_floor_not_a_bill() -> None:
    attempts = summarize_coding_attempts([_started("coding_a")])

    usage = combined_observed_tokens(
        {"totalTokens": 0},
        {"usage": {"totalTokens": 0}, "calls_without_usage": 0},
        attempts,
    )

    # Never zero-cost: the number is still 0 because nothing was observed, but
    # it is explicitly flagged as incomplete rather than presented as the bill.
    assert usage["partially_unknown"] is True
    assert usage["coder_attempts_without_usage"] == 1


def test_a_settled_round_leaves_the_accounting_complete() -> None:
    attempts = summarize_coding_attempts([_started("coding_a"), _round("coding_a")])

    assert attempts["unaccounted"] == 0
    assert attempts["convergence_credit"] is True

    usage = combined_observed_tokens(
        {"totalTokens": 4_000},
        {"usage": {"totalTokens": 900}, "calls_without_usage": 0},
        attempts,
    )

    assert usage["partially_unknown"] is False
    assert usage["total_tokens"] == 4_900
    assert usage["coder_attempts_without_usage"] == 0


def test_one_settled_round_does_not_cover_a_second_that_vanished() -> None:
    attempts = summarize_coding_attempts(
        [_started("coding_a"), _round("coding_a"), _started("coding_b")]
    )

    assert attempts["attempted_requests"] == 2
    assert attempts["settled"] == 1
    assert attempts["unaccounted_sessions"] == ["coding_b"]
    assert attempts["repair_credit"] is False


def test_a_rejected_round_settles_because_it_came_back_with_a_decision() -> None:
    trace = [
        _started("coding_a"),
        TimelineEvent(
            "project.coding_rejected",
            {"session_id": "coding_a", "rejection_reason": "path_outside_scope"},
        ),
    ]

    attempts = summarize_coding_attempts(trace)

    assert attempts["unaccounted"] == 0
    assert attempts["convergence_credit"] is True


def _attempt(blocking: int, attempts: dict[str, Any]) -> dict[str, Any]:
    """One scoreable attempt, shaped the way summarize_run reports it."""

    return {
        "plan": {
            "present": True,
            "intent": "build",
            "files": ["app/static/styles.css"],
        },
        "writes": {
            "attempted": 1,
            "planned_successful": 1,
            "path_evidence": "complete",
        },
        "verification": {"attempts": 1, "blocking": blocking},
        "approval": {"offered": True, "blocked": blocking > 0},
        "model_steps": 3,
        "coding_attempts": attempts,
    }


ACCEPTANCE = {"available": True, "passed": False, "checks_total": 0, "checks_passed": 0}

SETTLED = summarize_coding_attempts([_started("coding_a"), _round("coding_a")])
UNSETTLED = summarize_coding_attempts([_started("coding_a")])


def test_an_unsettled_attempt_earns_no_repair_or_convergence_credit() -> None:
    # Same run, same clean verification. The only difference is whether the
    # inference that supposedly produced it ever came back.
    card = score_evaluation([_attempt(0, UNSETTLED)], ACCEPTANCE, max_steps=24)

    assert card["categories"]["repair_convergence"] == 0.0
    assert card["signals"]["blocking_reduction"] == 0.0
    assert card["signals"]["convergence_credit_withheld"] is True


def test_the_same_run_with_a_settled_round_keeps_its_convergence_credit() -> None:
    card = score_evaluation([_attempt(0, SETTLED)], ACCEPTANCE, max_steps=24)

    assert card["categories"]["repair_convergence"] == 15.0
    assert card["signals"]["convergence_credit_withheld"] is False


# ── Cleanup runs on every exit, and only over what the evaluation owns ─────


class _Session:
    def __init__(self, session_id: str, workspace_path: Path) -> None:
        self.id = session_id
        self.workspace_path = workspace_path


class _Sessions:
    def __init__(self, sessions: list[_Session]) -> None:
        self._sessions = sessions

    async def for_run(self, run_id: str) -> list[_Session]:
        return list(self._sessions)


class _Coordinator:
    """Records what a release would actually have deleted."""

    def __init__(
        self, sessions: list[_Session], *, fail: set[str] | None = None
    ) -> None:
        self.sessions = _Sessions(sessions)
        self.aborted: list[str] = []
        self.released: list[str] = []
        self._fail = fail or set()

    async def abort(self, session_id: str, *, discard: bool = False) -> None:
        self.aborted.append(session_id)

    async def release(self, session_id: str, terminal_state: Any) -> None:
        if session_id in self._fail:
            raise RuntimeError("sidecar refused the delete")
        self.released.append(session_id)


class _BrokenSessions:
    async def for_run(self, run_id: str) -> list[_Session]:
        raise RuntimeError("the session store never opened")


@pytest.mark.asyncio
async def test_a_timed_out_probe_still_releases_its_session_and_mirror(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repair-canary"
    mine = workspace / "data" / "coding-sessions" / "mirror-a"
    mine.mkdir(parents=True)
    coordinator = _Coordinator([_Session("coding_mine", mine)])

    # The probe body raises the way a wall-clock abort does; the finally path
    # is what has to clean up.
    result: dict[str, Any] = {}
    with pytest.raises(TimeoutError):
        try:
            raise TimeoutError("120s wall clock")
        finally:
            result = await release_evaluation_artifacts(
                coordinator, run_id="run_1", workspace_root=workspace
            )

    assert result["released"] == ["coding_mine"]
    assert result["clean"] is True
    assert coordinator.aborted == ["coding_mine"]


@pytest.mark.asyncio
async def test_setup_failure_before_any_round_is_cleaned_up_too(
    tmp_path: Path,
) -> None:
    # A session created, then a crash before the first inference: still owned,
    # still released.
    workspace = tmp_path / "repair-canary"
    mine = workspace / "data" / "mirror-a"
    mine.mkdir(parents=True)
    coordinator = _Coordinator([_Session("coding_mine", mine)])

    with pytest.raises(KeyError):
        try:
            raise KeyError("prompt")
        finally:
            await release_evaluation_artifacts(
                coordinator, run_id="run_1", workspace_root=workspace
            )

    assert coordinator.released == ["coding_mine"]


@pytest.mark.asyncio
async def test_the_running_app_session_is_never_touched(tmp_path: Path) -> None:
    """The rule this file exists to keep.

    The app's own sidecar lives under the repository's ``.data`` directory.
    An evaluator that reaches it would take down the developer's running
    Metis while trying to tidy up after itself.
    """

    workspace = tmp_path / "repair-canary"
    mine = workspace / "data" / "mirror-a"
    mine.mkdir(parents=True)
    theirs = tmp_path / "repo" / ".data" / "coding-sessions" / "mirror-app"
    theirs.mkdir(parents=True)
    coordinator = _Coordinator(
        [_Session("coding_mine", mine), _Session("coding_app", theirs)]
    )

    result = await release_evaluation_artifacts(
        coordinator, run_id="run_1", workspace_root=workspace
    )

    assert result["released"] == ["coding_mine"]
    assert result["skipped"] == ["coding_app"]
    assert "coding_app" not in coordinator.aborted
    assert "coding_app" not in coordinator.released


@pytest.mark.asyncio
async def test_a_sibling_path_prefix_is_not_treated_as_ownership(
    tmp_path: Path,
) -> None:
    # "repair-canary-app" starts with "repair-canary". String prefixes are not
    # containment, and this is the difference between tidying up and deleting
    # somebody else's session.
    workspace = tmp_path / "repair-canary"
    workspace.mkdir()
    sibling = tmp_path / "repair-canary-app" / "mirror"
    sibling.mkdir(parents=True)
    coordinator = _Coordinator([_Session("coding_sibling", sibling)])

    result = await release_evaluation_artifacts(
        coordinator, run_id="run_1", workspace_root=workspace
    )

    assert result["released"] == []
    assert result["skipped"] == ["coding_sibling"]


@pytest.mark.asyncio
async def test_a_failed_release_is_reported_rather_than_swallowed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repair-canary"
    mine = workspace / "mirror-a"
    mine.mkdir(parents=True)
    coordinator = _Coordinator([_Session("coding_mine", mine)], fail={"coding_mine"})

    result = await release_evaluation_artifacts(
        coordinator, run_id="run_1", workspace_root=workspace
    )

    assert result["clean"] is False
    assert result["released"] == []
    assert "sidecar refused the delete" in result["failures"][0]


@pytest.mark.asyncio
async def test_cleanup_of_a_store_that_never_opened_does_not_raise(
    tmp_path: Path,
) -> None:
    # The finally path must not replace the real failure with its own.
    coordinator = _Coordinator([])
    coordinator.sessions = _BrokenSessions()  # type: ignore[assignment]

    result = await release_evaluation_artifacts(
        coordinator, run_id="run_1", workspace_root=tmp_path
    )

    assert result["clean"] is False
    assert result["released"] == []
