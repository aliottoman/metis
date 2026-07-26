"""P11 — the self-extending loop: a task with no matching tool → the model AUTHORS
the tool's code → it is AST-gated, capability-approved, evaluated by execution,
trusted-auto-activated, then run (calling the local model at runtime), through
the real HTTP surface.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.main import create_app

SAMPLE = b"the cat sat on the mat and the cat ran fast the dog sat too"


def _settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
    )
    base.update(overrides)
    return Settings(**base)


def _wait(client: TestClient, run_id: str, statuses: set[str]) -> dict:
    for _ in range(400):
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        time.sleep(0.02)
    raise TimeoutError(run)


def _upload(client: TestClient) -> str:
    return client.post(
        "/api/v1/uploads", files={"file": ("notes.txt", SAMPLE, "text/plain")}
    ).json()["id"]


def _send(client: TestClient, conversation: str, content: str, attachments: list[str]) -> str:
    r = client.post(
        f"/api/v1/conversations/{conversation}/messages",
        json={"content": content, "attachment_ids": attachments},
    )
    assert r.status_code in (200, 201, 202), r.text
    return r.json()["run_id"]


def _pending(client: TestClient, run_id: str) -> dict:
    runs = client.get("/api/v1/runs?status=awaiting_approval").json()
    return next(r for r in runs if r["run"]["id"] == run_id)["approval"]


def _approve(client: TestClient, run_id: str) -> None:
    approval = _pending(client, run_id)
    r = client.post(
        f"/api/v1/runs/{run_id}/decisions",
        json={"approval_id": approval["id"], "decision": "approve"},
    )
    assert r.status_code == 200, r.text


def _authored_record(client: TestClient) -> dict:
    return next(
        item for item in client.get("/api/v1/tool-definitions").json()
        if item["definition"]["archetype"] == "code-authoring"
    )


def test_authors_builds_and_runs_a_tool_for_a_new_task(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)

        # An explicit task no specific archetype covers takes the trusted fast path.
        run1 = _send(client, conversation, "turn this into a tool: count word frequency in the attached text", [upload])
        assert _wait(client, run1, {"completed", "failed", "awaiting_approval"})["status"] == "completed"

        record = _authored_record(client)
        slug = record["definition"]["slug"]
        assert record["definition"]["capability_profile"]["code_allowlist"] == "pure-python-authored-v1"
        assert record["runnable"] is True

        # The model authored code is AST-gated, evaluated, and pinned in that run.
        builds = client.get("/api/v1/tool-definition-builds").json()
        built = next(b for b in builds if b["slug"] == slug)
        assert built["eval_report"]["passed"] is True
        assert built["status"] == "active"
        assert "def run(inputs, model)" in built["implementation"]  # real authored code pinned

        # Run it — the authored code executes AND calls the local model at runtime.
        run2 = _send(client, conversation, f"{slug.replace('-', ' ')} of this", [upload])
        assert _wait(client, run2, {"completed", "failed", "awaiting_approval"})["status"] == "completed"
        answer = [
            m for m in client.get(f"/api/v1/conversations/{conversation}/messages").json()
            if m["role"] == "assistant"
        ][-1]["content"]
        assert "word_count" in answer  # the authored tool's typed output surfaced
        # The tool's runtime model() call ran: its output's `topic` field is filled
        # from the brokered reply (it would be "unknown" only if the call had raised).
        assert "topic" in answer and "unknown" not in answer
