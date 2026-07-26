#!/usr/bin/env python3
"""Exercise Metis's release-gating README-to-architecture vertical slice.

Run this against a disposable Metis API configured with the deterministic model
provider and the real Podman sandbox. The script intentionally creates and
activates a tool, then submits governed corrective feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import httpx


TERMINAL = {"completed", "failed", "cancelled"}
EXPECTED_ARTIFACTS = {
    "architecture-spec.json",
    "diagram.py",
    "architecture.svg",
    "architecture.png",
    "validation-report.json",
}


def _request(
    client: httpx.Client, method: str, path: str, **kwargs: Any
) -> Any:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    if response.status_code == 204:
        return None
    return response.json()


def _wait_for_status(
    client: httpx.Client,
    run_id: str,
    wanted: set[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run = _request(client, "GET", f"/api/v1/runs/{run_id}")
        status = run["status"]
        if status in wanted:
            return run
        if status in TERMINAL:
            raise RuntimeError(
                f"run {run_id} reached unexpected {status}: {run.get('last_error')}"
            )
        time.sleep(0.5)
    raise TimeoutError(f"run {run_id} did not reach {sorted(wanted)}")


def _create_architecture_run(
    client: httpx.Client, upload_id: str, title: str
) -> tuple[str, str]:
    conversation = _request(
        client, "POST", "/api/v1/conversations", json={"title": title}
    )
    accepted = _request(
        client,
        "POST",
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "content": (
                "Given this README, build a validated reference architecture using "
                "Python's diagrams library. Return JSON, Python, SVG, and PNG artifacts."
            ),
            "attachment_ids": [upload_id],
        },
    )
    return conversation["id"], accepted["run_id"]


def _verify_artifacts(client: httpx.Client, run: dict[str, Any]) -> None:
    result = run.get("result") or {}
    artifacts = result.get("artifacts") or []
    names = {item["filename"] for item in artifacts}
    if names != EXPECTED_ARTIFACTS:
        raise AssertionError(
            f"artifact contract mismatch: expected {sorted(EXPECTED_ARTIFACTS)}, "
            f"received {sorted(names)}"
        )
    for artifact in artifacts:
        response = client.get(artifact["download_url"])
        response.raise_for_status()
        if len(response.content) != artifact["size"]:
            raise AssertionError(f"artifact size mismatch: {artifact['filename']}")
        if hashlib.sha256(response.content).hexdigest() != artifact["sha256"]:
            raise AssertionError(f"artifact digest mismatch: {artifact['filename']}")
        disposition = response.headers.get("content-disposition", "").lower()
        if "attachment" not in disposition:
            raise AssertionError(f"artifact is not download-only: {artifact['filename']}")


def _verify_event_replay(client: httpx.Client, run_id: str) -> int:
    response = client.get(f"/api/v1/runs/{run_id}/events?after=0")
    response.raise_for_status()
    sequences = [
        int(line[3:])
        for line in response.text.splitlines()
        if line.startswith("id: ")
    ]
    if not sequences or sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
        raise AssertionError("event replay sequence is missing, duplicate, or unordered")
    if "event: run.completed" not in response.text:
        raise AssertionError("event replay omitted run.completed")
    replay = client.get(
        f"/api/v1/runs/{run_id}/events?after={sequences[-1]}"
    )
    replay.raise_for_status()
    if any(line.startswith("id: ") for line in replay.text.splitlines()):
        raise AssertionError("event replay returned events after the terminal cursor")
    return len(sequences)


def run_acceptance(base_url: str, readme: Path, timeout_seconds: float) -> dict[str, Any]:
    with httpx.Client(
        base_url=base_url.rstrip("/"), timeout=max(30.0, timeout_seconds)
    ) as client:
        health = _request(client, "GET", "/api/v1/health")
        if health["reference_runner"] != "podman":
            raise AssertionError("acceptance requires the real Podman runner")

        with readme.open("rb") as handle:
            upload = _request(
                client,
                "POST",
                "/api/v1/uploads",
                files={"file": (readme.name, handle, "text/markdown")},
            )

        _, first_run_id = _create_architecture_run(
            client, upload["id"], "Architecture candidate acceptance"
        )
        _wait_for_status(
            client, first_run_id, {"awaiting_approval"}, timeout_seconds
        )
        recoverable = _request(
            client, "GET", "/api/v1/runs?status=awaiting_approval"
        )
        pending = next(
            item for item in recoverable if item["run"]["id"] == first_run_id
        )
        approval = pending.get("approval")
        if not approval or approval.get("risk_level") != "R3":
            raise AssertionError("candidate activation did not produce an R3 approval")
        _request(
            client,
            "POST",
            f"/api/v1/runs/{first_run_id}/decisions",
            json={
                "approval_id": approval["id"],
                "decision": "approve",
                "reason": "Automated disposable acceptance run",
            },
        )
        first = _wait_for_status(
            client, first_run_id, {"completed"}, timeout_seconds
        )
        _verify_artifacts(client, first)
        first_events = _verify_event_replay(client, first_run_id)

        tools = _request(client, "GET", "/api/v1/tools")
        tool = next(
            item
            for item in tools
            if item["slug"] == "reference-architecture-generator"
        )
        active_version = tool.get("active_version_id")
        if not active_version:
            raise AssertionError("approved candidate did not become active")

        _, second_run_id = _create_architecture_run(
            client, upload["id"], "Approved capability reuse acceptance"
        )
        second = _wait_for_status(
            client, second_run_id, {"completed"}, timeout_seconds
        )
        _verify_artifacts(client, second)
        if (second.get("result") or {}).get("proposal") is not None:
            raise AssertionError("approved tool reuse unexpectedly rebuilt a candidate")
        second_events = _verify_event_replay(client, second_run_id)

        feedback = _request(
            client,
            "POST",
            f"/api/v1/runs/{second_run_id}/feedback",
            json={
                "run_id": second_run_id,
                "rating": "negative",
                "correction": (
                    "Always label the client-to-service edge with its application protocol."
                ),
            },
        )
        improvement_ids = feedback.get("tool_improvement_proposal_ids") or []
        if not feedback.get("memory_proposal_id") or not improvement_ids:
            raise AssertionError("correction did not create governed learning proposals")
        improvements = _request(
            client, "GET", "/api/v1/tool-improvement-proposals?status=pending"
        )
        if not set(improvement_ids).issubset({item["id"] for item in improvements}):
            raise AssertionError("pending regression proposal is not retrievable")
        unchanged = next(
            item
            for item in _request(client, "GET", "/api/v1/tools")
            if item["id"] == tool["id"]
        )
        if unchanged.get("active_version_id") != active_version:
            raise AssertionError("corrective feedback silently changed the active version")

        return {
            "status": "passed",
            "first_run_id": first_run_id,
            "second_run_id": second_run_id,
            "active_version_id": active_version,
            "first_event_count": first_events,
            "second_event_count": second_events,
            "artifact_count": len(EXPECTED_ARTIFACTS),
            "improvement_proposal_ids": improvement_ids,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--timeout", type=float, default=240.0)
    arguments = parser.parse_args()
    result = run_acceptance(arguments.base_url, arguments.readme, arguments.timeout)
    import json

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
