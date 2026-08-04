"""P9 — the tool-definition flow, end to end, through the real HTTP surface.

Drives the deterministic backend through the trusted lifecycle for a non-diagram
tool: explicit "toolify this" → approve the immutable capability definition →
factory build + hermetic eval → automatic activation inside the no-network,
run-IO boundary → typed output. Broader deployments can still retain Gate-2,
Gate-1 rejection tombstones the draft, and kill-switches pause the factory.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.main import create_app

README = (
    b"# Acme Queue\n"
    b"Acme Queue is a lightweight distributed task queue for Python.\n\n"
    b"## Components\n- api\n- worker\n- scheduler\n\n"
    b"## Built with\nPython, Redis, FastAPI\n"
)


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
    for _ in range(300):
        latest = client.get(f"/api/v1/runs/{run_id}").json()
        if latest["status"] in statuses:
            return latest
        time.sleep(0.02)
    raise TimeoutError(latest)


def _upload(client: TestClient) -> str:
    return client.post(
        "/api/v1/uploads", files={"file": ("README.md", README, "text/markdown")}
    ).json()["id"]


def _send(client: TestClient, conversation: str, content: str, attachments: list[str]) -> str:
    response = client.post(
        f"/api/v1/conversations/{conversation}/messages",
        json={"content": content, "attachment_ids": attachments},
    )
    assert response.status_code in (200, 201, 202), response.text
    return response.json()["run_id"]


def _pending_approval(client: TestClient, run_id: str) -> dict:
    runs = client.get("/api/v1/runs?status=awaiting_approval").json()
    record = next(item for item in runs if item["run"]["id"] == run_id)
    return record["approval"]


def _decide(client: TestClient, run_id: str, decision: str) -> None:
    approval = _pending_approval(client, run_id)
    response = client.post(
        f"/api/v1/runs/{run_id}/decisions",
        json={"approval_id": approval["id"], "decision": decision},
    )
    assert response.status_code == 200, response.text


def test_explicit_toolify_uses_trusted_fast_path_end_to_end(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)

        # The explicit build action is authorization for the host-hardened,
        # network-free profile, so draft → eval → activation → use is one run.
        run1 = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        assert _wait(client, run1, {"completed", "failed", "awaiting_approval"})["status"] == "completed"

        record = next(
            item for item in client.get("/api/v1/tool-definitions").json()
            if item["definition"]["slug"] == "readme-summary"
        )
        assert record["buildable"] is False and record["runnable"] is True
        assert record["definition"]["status"] == "defined"
        assert record["definition"]["capability_profile"]["code_allowlist"] == "declarative-host-v1"
        builds = client.get("/api/v1/tool-definition-builds").json()
        assert builds and builds[0]["eval_report"]["passed"] is True
        assert builds[0]["status"] == "active"

        # Runnable now.
        record = next(
            item for item in client.get("/api/v1/tool-definitions").json()
            if item["definition"]["slug"] == "readme-summary"
        )
        assert record["runnable"] is True

        messages = client.get(f"/api/v1/conversations/{conversation}/messages").json()
        answer = [m for m in messages if m["role"] == "assistant"][-1]["content"]
        assert "Acme Queue" in answer
        assert "api" in answer  # a component surfaced

        # The brokered model call during the run was audited on the timeline.
        events = client.get(f"/api/v1/runs/{run1}/events?after=0").text
        assert "tool.definition_auto_approved" in events
        assert "tool.build_auto_activated" in events
        assert "run.broker_call" in events
        assert "tool.output" in events


def test_trusted_auto_activation_can_be_disabled(tmp_path) -> None:
    with TestClient(
        create_app(_settings(tmp_path, tool_trusted_auto_activation=False))
    ) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)
        run1 = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        assert _wait(client, run1, {"awaiting_approval", "failed"})["status"] == "awaiting_approval"
        _decide(client, run1, "approve")
        assert _wait(client, run1, {"completed", "failed"})["status"] == "completed"

        run2 = _send(client, conversation, "give me a readme summary of this project", [upload])
        assert _wait(client, run2, {"awaiting_approval", "failed"})["status"] == "awaiting_approval"
        assert _pending_approval(client, run2)["kind"] == "activate_definition"


def test_startup_promotes_legacy_pending_explicit_tool_request(tmp_path) -> None:
    manual = _settings(tmp_path, tool_trusted_auto_activation=False)
    with TestClient(create_app(manual)) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)
        run = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        assert _wait(client, run, {"awaiting_approval", "failed"})["status"] == "awaiting_approval"
        assert client.get("/api/v1/tool-definition-proposals?status=pending").json()

    # The next version recognizes that the human already issued an explicit build
    # instruction. It promotes only the registered safe definition; a future
    # matching request will build and run it through the trusted fast path.
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert not client.get("/api/v1/tool-definition-proposals?status=pending").json()
        proposals = client.get("/api/v1/tool-definition-proposals").json()
        assert proposals[0]["status"] == "approved"
        record = _slug_record(client, "readme-summary")
        assert record["definition"]["status"] == "defined"
        assert record["buildable"] is True


def test_gate1_rejection_tombstones_the_definition(tmp_path) -> None:
    with TestClient(
        create_app(_settings(tmp_path, tool_trusted_auto_activation=False))
    ) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)
        run1 = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        assert _wait(client, run1, {"awaiting_approval", "failed"})["status"] == "awaiting_approval"
        _decide(client, run1, "reject")
        assert _wait(client, run1, {"completed", "failed"})["status"] == "completed"
        # The proposal is rejected and the definition is not active/buildable.
        proposals = client.get("/api/v1/tool-definition-proposals").json()
        assert proposals and proposals[0]["status"] == "rejected"
        assert client.get("/api/v1/tool-definitions").json() == [
            item for item in client.get("/api/v1/tool-definitions").json()
            if item["definition"]["slug"] != "readme-summary"
        ]
        # Re-drafting the identical (rejected) definition is refused.
        run2 = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        assert _wait(client, run2, {"completed", "failed", "awaiting_approval"})["status"] == "completed"
        messages = client.get(f"/api/v1/conversations/{conversation}/messages").json()
        answer = [m for m in messages if m["role"] == "assistant"][-1]["content"].lower()
        assert "previously rejected" in answer


def test_factory_kill_switch_pauses_toolify(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path, tool_factory_enabled=False))) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)
        run1 = _send(client, conversation, "turn this into a tool: summarize readmes", [upload])
        # No gate — the factory is paused, so this is a direct answer, not a draft.
        assert _wait(client, run1, {"completed", "failed", "awaiting_approval"})["status"] == "completed"
        assert not client.get("/api/v1/tool-definition-proposals").json()


def _slug_record(client: TestClient, slug: str) -> dict:
    return next(
        item for item in client.get("/api/v1/tool-definitions").json()
        if item["definition"]["slug"] == slug
    )


def _lifecycle_to_active(client: TestClient, conversation: str, upload: str, toolify_prompt: str) -> None:
    run1 = _send(client, conversation, toolify_prompt, [upload])
    assert _wait(client, run1, {"completed", "failed", "awaiting_approval"})["status"] == "completed"


def test_revision_creates_new_version_and_supersedes(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)

        _lifecycle_to_active(client, conversation, upload, "turn this into a tool: summarize readmes")
        v1 = _slug_record(client, "readme-summary")["definition"]["version"]
        assert _slug_record(client, "readme-summary")["runnable"] is True

        # A changed explicit toolify request replaces the evaluated trusted version.
        run3 = _send(
            client, conversation, "turn this into a tool: summarize readmes and include the license", [upload]
        )
        assert _wait(client, run3, {"completed", "failed", "awaiting_approval"})["status"] == "completed"
        record = _slug_record(client, "readme-summary")
        assert record["runnable"] is True and record["buildable"] is False

        v2 = _slug_record(client, "readme-summary")["definition"]["version"]
        assert v2 != v1  # a genuinely new immutable version is live
        # Exactly one active build; the prior one was superseded.
        builds = client.get("/api/v1/tool-definition-builds").json()
        assert [b["status"] for b in builds].count("active") == 1
        assert "superseded" in [b["status"] for b in builds]


def test_disabling_architecture_tool_routes_direct(tmp_path) -> None:
    settings = _settings(tmp_path, tool_disabled_slugs=["reference-architecture-generator"])
    with TestClient(create_app(settings)) as client:
        conversation = client.post("/api/v1/conversations", json={}).json()["id"]
        upload = _upload(client)
        run = _send(client, conversation, "make an architecture diagram from this README", [upload])
        # The one built-in tool is disabled → no tool runs; a direct answer instead.
        assert _wait(client, run, {"completed", "failed", "awaiting_approval"})["status"] == "completed"
        events = client.get(f"/api/v1/runs/{run}/events?after=0").text
        assert "architecture.spec_created" not in events
