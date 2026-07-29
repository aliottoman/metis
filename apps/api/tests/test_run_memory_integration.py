"""End-to-end checks that a finished run actually feeds the next one.

The unit tests prove each piece works in isolation. These prove the wiring: that
a real run through the API leaves behind a retrievable document and a memory
proposal the user still has to approve.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from test_api import new_conversation, wait_for_status


def _wait_for(predicate, timeout: float = 5.0):
    """Background upkeep is deliberately off the run's critical path."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def test_a_finished_run_is_written_into_history(client: TestClient, settings) -> None:
    conversation_id = new_conversation(client)
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "Why does the nightly import drop rows?", "attachment_ids": []},
    )
    run_id = accepted.json()["run_id"]
    run = wait_for_status(client, run_id, {"completed", "failed"})
    assert run["status"] == "completed"

    root: Path = settings.data_dir / "corpus" / "runs"
    documents = _wait_for(lambda: list(root.rglob("*.md")))
    assert documents, "a completed run left no history document"
    text = documents[0].read_text(encoding="utf-8")
    assert "Why does the nightly import drop rows?" in text
    assert run_id in text


def test_history_appears_in_knowledge_as_an_unconsented_source(
    client: TestClient,
) -> None:
    conversation_id = new_conversation(client)
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "A question worth keeping.", "attachment_ids": []},
    )
    wait_for_status(client, accepted.json()["run_id"], {"completed", "failed"})

    sources = _wait_for(
        lambda: [
            item
            for item in client.get("/api/v1/corpus/sources").json()
            if item["label"] == "Run history"
        ]
    )
    assert sources, "run history never registered itself as a corpus source"
    # Indexing past work is a visible, revocable choice, not a default.
    assert sources[0]["consent"] is False


def test_harvested_memories_are_proposed_and_never_activated(
    client: TestClient,
) -> None:
    conversation_id = new_conversation(client)
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "content": "[memory-harvest-test] Standardize on uv for Python envs.",
            "attachment_ids": [],
        },
    )
    wait_for_status(client, accepted.json()["run_id"], {"completed", "failed"})

    pending = _wait_for(
        lambda: client.get("/api/v1/memory/proposals?status=pending").json()
    )
    assert pending, "a harvestable run produced no memory proposal"
    assert any("durable fact" in item["content"] for item in pending)

    # The invariant the whole memory model rests on: nothing becomes active
    # without a human decision, however confident the harvester was.
    status = client.get("/api/v1/memory/index").json()
    assert status["active"] == 0


def test_ordinary_runs_do_not_manufacture_memories(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    accepted = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "What time is the standup?", "attachment_ids": []},
    )
    wait_for_status(client, accepted.json()["run_id"], {"completed", "failed"})
    time.sleep(0.2)

    # Proposing something from every run would train the user to dismiss the
    # queue, which costs more than the occasional missed fact.
    assert client.get("/api/v1/memory/proposals?status=pending").json() == []


def test_memory_index_reports_keyword_mode_until_consent(client: TestClient) -> None:
    status = client.get("/api/v1/memory/index").json()
    assert status == {
        "consent": False,
        "consent_reason": None,
        "cloud_available": False,
        "semantic": False,
        "active": 0,
        "embedded": 0,
    }

    granted = client.post(
        "/api/v1/memory/index/consent", json={"consent": True, "reason": "enabled"}
    )
    assert granted.status_code == 200
    # Consent is recorded, but the honest answer is still "not semantic": the
    # cloud path is unavailable in the test environment.
    assert granted.json()["consent"] is True
    assert granted.json()["semantic"] is False
