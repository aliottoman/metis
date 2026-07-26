#!/usr/bin/env python3
"""Prove an interrupted approval survives restart and applies exactly once.

This uses deterministic model responses but the real configured Podman runner,
so the smoke test is fast enough for release verification while retaining the
same LangGraph checkpoints, domain database, approval, activation, and sandbox
boundaries as production.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api" / "src"))

from waqil_api.config import Settings  # noqa: E402
from waqil_api.main import create_app  # noqa: E402


def _wait(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        response.raise_for_status()
        latest = response.json()
        if latest["status"] in statuses:
            return latest
        if latest["status"] in {"completed", "failed", "cancelled"}:
            raise RuntimeError(
                f"run reached unexpected {latest['status']}: {latest.get('last_error')}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"run did not reach {sorted(statuses)}: {latest}")


def _run(data_dir: Path, image: str, timeout_seconds: float) -> dict[str, Any]:
    settings = Settings(
        data_dir=data_dir,
        repo_root=PROJECT_ROOT,
        model_backend="deterministic",
        reference_runner_mode="podman",
        reference_runner_image=image,
        reference_runner_timeout_seconds=min(max(int(timeout_seconds), 135), 600),
        allow_test_backends=True,
    )

    with TestClient(create_app(settings)) as first:
        conversation_response = first.post(
            "/api/v1/conversations", json={"title": "Restart smoke"}
        )
        conversation_response.raise_for_status()
        conversation = conversation_response.json()
        accepted_response = first.post(
            f"/api/v1/conversations/{conversation['id']}/messages",
            json={
                "content": "Build a validated local reference architecture diagram.",
                "attachment_ids": [],
            },
        )
        accepted_response.raise_for_status()
        run_id = accepted_response.json()["run_id"]
        waiting = _wait(
            first, run_id, {"awaiting_approval", "failed"}, timeout_seconds
        )
        if waiting["status"] != "awaiting_approval":
            raise RuntimeError(waiting.get("last_error") or "candidate failed")

        recoverable = first.get("/api/v1/runs?status=awaiting_approval")
        recoverable.raise_for_status()
        pending = next(
            item for item in recoverable.json() if item["run"]["id"] == run_id
        )
        approval = pending["approval"]

    # A new application/runtime instance simulates the process boundary. It
    # reads the existing domain DB and LangGraph checkpoint DB from data_dir.
    with TestClient(create_app(settings)) as restarted:
        recovery_response = restarted.get("/api/v1/runs?status=awaiting_approval")
        recovery_response.raise_for_status()
        recovered = next(
            item
            for item in recovery_response.json()
            if item["run"]["id"] == run_id
        )
        if recovered["approval"]["id"] != approval["id"]:
            raise AssertionError("restart changed the pinned approval identity")

        decision = restarted.post(
            f"/api/v1/runs/{run_id}/decisions",
            json={
                "approval_id": approval["id"],
                "decision": "approve",
                "reason": "Disposable restart acceptance smoke",
            },
        )
        decision.raise_for_status()
        completed = _wait(restarted, run_id, {"completed", "failed"}, timeout_seconds)
        if completed["status"] != "completed":
            raise RuntimeError(completed.get("last_error") or "resumed run failed")

        events_response = restarted.get(f"/api/v1/runs/{run_id}/events?after=0")
        events_response.raise_for_status()
        events = events_response.text
        if "event: run.resumed" not in events:
            raise AssertionError("resumed event was not persisted")
        if events.count("event: approval.applied") != 1:
            raise AssertionError("approved side effect was not applied exactly once")

        result = completed.get("result") or {}
        artifacts = result.get("artifacts") or []
        tools_response = restarted.get("/api/v1/tools")
        tools_response.raise_for_status()
        tool = next(
            item
            for item in tools_response.json()
            if item["slug"] == "reference-architecture-generator"
        )
        if not tool.get("active_version_id"):
            raise AssertionError("resumed approval did not activate a version")

    with sqlite3.connect(settings.database_path) as connection:
        action_rows = connection.execute(
            "SELECT COUNT(*) FROM idempotency_actions WHERE action_id = ?",
            (approval["action_id"],),
        ).fetchone()[0]
    if action_rows != 1:
        raise AssertionError("approval action receipt is missing or duplicated")

    return {
        "status": "passed",
        "run_id": run_id,
        "approval_id": approval["id"],
        "approval_action_receipts": action_rows,
        "approval_applied_events": 1,
        "artifact_count": len(artifacts),
        "active_version_id": tool["active_version_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="localhost/metis/reference-architecture-tool:0.3.0",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--data-dir", type=Path)
    arguments = parser.parse_args()

    if arguments.data_dir:
        arguments.data_dir.mkdir(parents=True, exist_ok=True)
        result = _run(arguments.data_dir, arguments.image, arguments.timeout)
    else:
        with tempfile.TemporaryDirectory(prefix="metis-restart-smoke-") as directory:
            result = _run(Path(directory), arguments.image, arguments.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
