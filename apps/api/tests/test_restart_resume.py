from __future__ import annotations

import asyncio
import threading
import time

from fastapi.testclient import TestClient

from waqil_api.main import create_app
from waqil_api.model_provider import DeterministicModelProvider


def _wait(client: TestClient, run_id: str, statuses: set[str]) -> dict:
    deadline = time.monotonic() + 5
    latest = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/v1/runs/{run_id}").json()
        if latest["status"] in statuses:
            return latest
        time.sleep(0.02)
    raise AssertionError(latest)


def test_approval_interrupt_resumes_after_process_restart(settings) -> None:
    with TestClient(create_app(settings)) as first:
        conversation = first.post("/api/v1/conversations", json={}).json()
        accepted = first.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={"content": "Build an architecture diagram", "attachment_ids": []},
        ).json()
        waiting = _wait(first, accepted["run_id"], {"awaiting_approval", "failed"})
        assert waiting["status"] == "awaiting_approval"

    with TestClient(create_app(settings)) as restarted:
        pending = restarted.get("/api/v1/tool-proposals?status=pending").json()
        assert len(pending) == 1
        response = restarted.post(
            f"/api/v1/tool-proposals/{pending[0]['id']}/approve", json={}
        )
        assert response.status_code == 200
        completed = _wait(restarted, accepted["run_id"], {"completed", "failed"})
        assert completed["status"] == "completed", completed
        events = restarted.get(
            f"/api/v1/runs/{accepted['run_id']}/events?after=0"
        ).text
        assert "event: run.resumed" in events
        assert events.count("event: approval.applied") == 1


def test_decision_persisted_before_crash_is_resumed_on_startup(settings) -> None:
    with TestClient(create_app(settings)) as first:
        conversation = first.post("/api/v1/conversations", json={}).json()
        accepted = first.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={"content": "Build an architecture diagram", "attachment_ids": []},
        ).json()
        waiting = _wait(first, accepted["run_id"], {"awaiting_approval", "failed"})
        assert waiting["status"] == "awaiting_approval"
        recovery = first.get("/api/v1/runs?status=awaiting_approval").json()
        approval = next(
            item["approval"] for item in recovery if item["run"]["id"] == accepted["run_id"]
        )
        changed = first.portal.call(
            first.app.state.runtime.database.record_approval_decision,
            approval["id"],
            "approve",
            "Persisted immediately before simulated process loss",
        )
        assert changed is True

    with TestClient(create_app(settings)) as restarted:
        completed = _wait(restarted, accepted["run_id"], {"completed", "failed"})
        assert completed["status"] == "completed", completed
        events = restarted.get(
            f"/api/v1/runs/{accepted['run_id']}/events?after=0"
        ).text
        assert "event: run.resumed" in events
        assert events.count("event: approval.applied") == 1


class SlowDeterministicProvider(DeterministicModelProvider):
    def __init__(self, started: threading.Event) -> None:
        self.started = started

    async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
        self.started.set()
        await asyncio.Event().wait()


def test_graceful_shutdown_suspends_and_recovers_inflight_graph(settings) -> None:
    started = threading.Event()
    with TestClient(create_app(settings)) as first:
        slow = SlowDeterministicProvider(started)
        first.app.state.runtime.model = slow
        first.app.state.runtime.control_plane.model = slow
        conversation = first.post("/api/v1/conversations", json={}).json()
        accepted = first.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={"content": "Explain the local agent", "attachment_ids": []},
        ).json()
        assert started.wait(timeout=5)
        checkpoint = first.portal.call(
            first.app.state.runtime.checkpointer.aget_tuple,
            first.app.state.runtime.control_plane._config(
                conversation["id"], accepted["run_id"]
            ),
        )
        assert checkpoint is not None

    with TestClient(create_app(settings)) as restarted:
        completed = _wait(restarted, accepted["run_id"], {"completed", "failed", "cancelled"})
        assert completed["status"] == "completed", completed
        events = restarted.get(
            f"/api/v1/runs/{accepted['run_id']}/events?after=0"
        ).text
        assert "event: run.suspended" in events
        assert "event: run.recovered" in events


def test_runs_in_one_conversation_have_isolated_checkpoint_keys(settings) -> None:
    with TestClient(create_app(settings)) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()
        runs = []
        for content in ("First local question", "Second local question"):
            accepted = client.post(
                f"/api/v1/conversations/{conversation['id']}/messages",
                json={"content": content, "attachment_ids": []},
            ).json()
            runs.append(accepted["run_id"])
        for run_id in runs:
            assert _wait(client, run_id, {"completed", "failed"})["status"] == "completed"
        rows = client.portal.call(
            client.app.state.runtime.checkpointer.conn.execute_fetchall,
            "SELECT DISTINCT thread_id FROM checkpoints",
        )
        stored = {row[0] for row in rows}
        assert {f"{conversation['id']}:{run_id}" for run_id in runs} <= stored
