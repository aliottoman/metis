"""What one `cline_direct` baseline round is allowed to measure.

Two live failures produced this file, and each test below is one of them.

* A round ended after six inspections and no edit, because a fixed count of
  four read/search calls was read as a loop. A session that owns a whole
  project reads many distinct files before its first edit; that is the work,
  not a defect. `broadScope` turns that stop off and keeps only the bounds
  that are about safety or genuine repetition.
* That same round then had nothing to explain it: the event drain used a
  150 ms idle window on its FIRST read, the sidecar's backlog had not been
  parsed off stdout yet, and the entire trace -- every tool call, iteration,
  usage snapshot and failure diagnostic, through the terminal event -- was
  discarded. The drain now reads to the journal's stated end cursor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from waqil_api.coding_contracts import CodingEventV1, StartSliceV1
from waqil_api.coding_engine import SidecarCodingEngine


# A sidecar that answers `subscribe` immediately but delivers its journal
# frames only after a delay far longer than the old 150 ms window, then ends
# with a terminal event. The old drain returned nothing here.
SLOW_BACKLOG_SIDECAR = r"""
import json
import sys
import time

def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

TRACE = [
    {"type": "state", "status": "starting"},
    {"type": "tool", "tool": "read_files", "status": "started", "iteration": 1},
    {"type": "tool", "tool": "read_files", "status": "finished", "iteration": 1},
    {"type": "tool", "tool": "search_codebase", "status": "started", "iteration": 2},
    {"type": "tool", "tool": "editor", "status": "started", "iteration": 3},
    {"type": "tool", "tool": "editor", "status": "failed", "iteration": 3,
     "message": "old_text was not found in app/main.py",
     "failureClass": "EditorMismatch"},
    {"type": "usage", "status": "snapshot", "iteration": 4,
     "usage": {"inputTokens": 40, "outputTokens": 9, "totalTokens": 49, "requests": 4},
     "usageScope": "cumulative"},
    {"type": "completed", "status": "completed", "iteration": 4},
]

for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    method = request["method"]
    params = request["params"]
    if method == "getInfo":
        emit({"version": "1", "id": request_id, "result": {
            "protocolVersion": "1", "engine": "clinecore", "runtime": "fake",
            "sdkVersion": "0.0.72", "policyVersion": "1",
            "allowedTools": ["editor", "read_files", "run_check", "search_codebase"],
        }})
    elif method == "subscribe":
        after = params.get("afterCursor", 0)
        pending = [
            (index + 1, item) for index, item in enumerate(TRACE) if index + 1 > after
        ]
        end = pending[-1][0] if pending else after
        emit({"version": "1", "id": request_id, "result": {
            "sessionId": params["sessionId"], "cursor": end,
            "oldestCursor": 1, "replayed": len(pending), "overflowed": False,
        }})
        # The whole point: the frames arrive well after the response, and far
        # later than any first-event idle window would have waited.
        time.sleep(0.6)
        for cursor, item in pending:
            emit({"version": "1", "method": "event", "params": {
                "sessionId": params["sessionId"], "cursor": cursor,
                "occurredAt": "2026-08-12T00:00:00+00:00", **item,
            }})
    elif method == "shutdown":
        emit({"version": "1", "id": request_id, "result": {"state": "shutting_down"}})
        break
"""


@pytest.mark.asyncio
async def test_the_whole_trace_survives_a_backlog_slower_than_the_old_window() -> None:
    """Terminal-event draining retains tools, iterations, usage and diagnostics."""

    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", SLOW_BACKLOG_SIDECAR],
        request_timeout_seconds=10,
    )
    try:
        drained = await engine.drain_events("session_one", after_cursor=0)
    finally:
        await engine.close()

    # Every journaled event, in order, through the terminal one. Under the old
    # 150 ms first-event timeout this list was empty.
    assert [event.cursor for event in drained] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert drained[-1].type == "completed"

    tools = [(event.tool, event.status) for event in drained if event.tool]
    assert tools == [
        ("read_files", "started"),
        ("read_files", "finished"),
        ("search_codebase", "started"),
        ("editor", "started"),
        ("editor", "failed"),
    ]

    # Iterations are preserved, so "is the prompt growing per iteration" stays
    # answerable from the record alone.
    assert [event.iteration for event in drained] == [None, 1, 1, 2, 3, 3, 4, 4]

    # An editor failure keeps its own diagnostic text and coarse class -- the
    # two fields that separate "the model was refused" from "the edit did not
    # apply".
    failure = next(
        event
        for event in drained
        if event.tool == "editor" and event.status == "failed"
    )
    assert failure.message == "old_text was not found in app/main.py"
    assert failure.failure_class == "EditorMismatch"

    usage = next(event for event in drained if event.usage is not None)
    assert usage.usage_scope == "cumulative"
    assert usage.usage is not None
    assert usage.usage.input_tokens == 40
    assert usage.usage.requests == 4


@pytest.mark.asyncio
async def test_a_drain_that_is_already_current_returns_without_waiting() -> None:
    """No target past the caller's cursor means no wait, not a stall."""

    engine = SidecarCodingEngine(
        [sys.executable, "-u", "-c", SLOW_BACKLOG_SIDECAR],
        request_timeout_seconds=10,
    )
    try:
        assert await engine.drain_events("session_one", after_cursor=8) == []
    finally:
        await engine.close()


class _CollectingEvents:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, run_id: str, thread_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        del run_id, thread_id
        self.items.append((event_type, payload))


@pytest.mark.asyncio
async def test_replay_persists_every_drained_event_and_advances_the_cursor(
    tmp_path: Path,
) -> None:
    """The coordinator writes the drained trace out and remembers where it got to."""

    from waqil_api.coding_contracts import CodingSessionState, CodingSessionV1
    from waqil_api.project_coding_engine import ProjectCodingCoordinator

    trace = [
        CodingEventV1(
            sessionId="sidecar_one",
            cursor=index,
            type="tool",
            tool="read_files",
            status="finished",
            iteration=index,
            occurredAt="2026-08-12T00:00:00+00:00",
        )
        for index in range(1, 6)
    ]
    trace.append(
        CodingEventV1(
            sessionId="sidecar_one",
            cursor=6,
            type="completed",
            status="completed",
            occurredAt="2026-08-12T00:00:00+00:00",
        )
    )

    class _Engine:
        def __init__(self) -> None:
            self.requests: list[tuple[str, int]] = []

        async def drain_events(
            self, session_id: str, *, after_cursor: int = 0, limit: int = 256
        ) -> list[CodingEventV1]:
            del limit
            self.requests.append((session_id, after_cursor))
            return [event for event in trace if event.cursor > after_cursor]

    class _Sessions:
        def __init__(self) -> None:
            self.cursors: list[int] = []

        async def update(self, session_id: str, update: Any) -> None:
            del session_id
            if update.event_cursor is not None:
                self.cursors.append(update.event_cursor)

    engine = _Engine()
    sessions = _Sessions()
    events = _CollectingEvents()
    coordinator = object.__new__(ProjectCodingCoordinator)
    coordinator.engine = engine  # type: ignore[attr-defined]
    coordinator.sessions = sessions  # type: ignore[attr-defined]
    coordinator.events = events  # type: ignore[attr-defined]
    coordinator._usage_marks = {}  # type: ignore[attr-defined]

    # Only the identity and the cursor matter to the drain; the rest of the
    # durable record is irrelevant here and is deliberately not invented.
    session = CodingSessionV1.model_construct(
        id="coding_one",
        run_id="run_1",
        conversation_id="conv_1",
        project_id="asset_1",
        sidecar_session_id="sidecar_one",
        event_cursor=0,
        state=CodingSessionState.IDLE,
        workspace_path=tmp_path,
    )

    await ProjectCodingCoordinator._replay_events(coordinator, session, "sidecar_one")

    emitted = [item for item in events.items if item[0] == "project.coding_event"]
    assert len(emitted) == len(trace)
    assert emitted[-1][1]["type"] == "completed"
    # The cursor lands on the terminal event, so the next round starts after it
    # rather than re-emitting the whole turn.
    assert sessions.cursors == [6]
    assert engine.requests == [("sidecar_one", 0)]


def test_a_direct_round_states_broad_scope_on_the_wire() -> None:
    """`broadScope` is sent, not inferred from an absent allowlist."""

    direct = StartSliceV1(
        sessionId="sidecar_one",
        prompt="add invoice-status filtering",
        workspaceRoot=Path("/tmp/mirror"),
        provider={  # type: ignore[arg-type]
            "providerId": "ollama",
            "modelId": "deepseek-v4-pro:cloud",
        },
        broadScope=True,
    )
    wire = direct.wire_dict()
    assert wire["broadScope"] is True
    # A whole-project session sends no allowlist; the deny-list bounds it.
    assert "allowedPaths" not in wire

    sliced = StartSliceV1(
        sessionId="sidecar_two",
        prompt="one slice",
        workspaceRoot=Path("/tmp/mirror"),
        provider={  # type: ignore[arg-type]
            "providerId": "ollama",
            "modelId": "deepseek-v4-pro:cloud",
        },
        allowedPaths=["app/main.py"],
        broadScope=False,
    )
    sliced_wire = sliced.wire_dict()
    assert sliced_wire["broadScope"] is False
    assert sliced_wire["allowedPaths"] == ["app/main.py"]
