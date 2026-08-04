"""Hand edits, direct notes, and cross-account search.

An extraction proposes; a person decides. These tests hold that line: every
account-scoped record can be written, corrected, and removed without a model in
the loop, and what the user writes is findable again from anywhere.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _account(client: TestClient, name: str, **fields: object) -> str:
    created = client.post("/api/v1/customers", json={"name": name, **fields})
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


def test_account_profile_is_editable(client: TestClient) -> None:
    account_id = _account(client, "Northwind")
    updated = client.put(
        f"/api/v1/customers/{account_id}",
        json={
            "name": "Northwind Authority",
            "aliases": ["NWA", "Northwind"],
            "industry": "Government",
            "region": "UAE",
            "status": "paused",
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Northwind Authority"
    assert updated.json()["aliases"] == ["NWA", "Northwind"]
    assert updated.json()["status"] == "paused"

    detail = client.get(f"/api/v1/customers/{account_id}").json()
    assert detail["account"]["industry"] == "Government"

    missing = client.put(
        "/api/v1/customers/cust_00000000000000000000",
        json={"name": "Ghost"},
    )
    assert missing.status_code == 404


def test_direct_notes_are_saved_edited_pinned_and_deleted(client: TestClient) -> None:
    account_id = _account(client, "Harbor Health")

    created = client.post(
        f"/api/v1/customers/{account_id}/notes",
        json={"title": "Standing context", "body": "Runs OCI Ashburn.", "pinned": True},
    )
    assert created.status_code == 201, created.text
    note = created.json()
    assert note["pinned"] is True
    assert note["origin"] == "manual"

    # A note is knowledge, not raw material: it never enters the review queue.
    detail = client.get(f"/api/v1/customers/{account_id}").json()
    assert [item["id"] for item in detail["notes"]] == [note["id"]]
    assert detail["sources"] == []
    assert detail["account"]["pending_notes"] == 0

    edited = client.put(
        f"/api/v1/customers/notes/{note['id']}",
        json={"title": "Standing context", "body": "Runs OCI Ashburn and Dubai.", "pinned": False},
    )
    assert edited.status_code == 200
    assert edited.json()["body"] == "Runs OCI Ashburn and Dubai."
    assert edited.json()["pinned"] is False

    assert client.delete(f"/api/v1/customers/notes/{note['id']}").status_code == 204
    assert client.delete(f"/api/v1/customers/notes/{note['id']}").status_code == 404
    assert client.get(f"/api/v1/customers/{account_id}").json()["notes"] == []


def test_a_note_saved_from_chat_keeps_its_provenance(client: TestClient) -> None:
    account_id = _account(client, "Simvia")
    created = client.post(
        f"/api/v1/customers/{account_id}/notes",
        json={
            "body": "Metis drafted the migration plan in this conversation.",
            "origin": "chat",
            "origin_ref": "conv_1234",
        },
    )
    assert created.status_code == 201
    assert created.json()["origin"] == "chat"
    assert created.json()["origin_ref"] == "conv_1234"

    orphan = client.post(
        "/api/v1/customers/cust_00000000000000000000/notes",
        json={"body": "No account to hold this."},
    )
    assert orphan.status_code == 404


def test_pinned_notes_reach_the_scoped_chat_context(client: TestClient) -> None:
    """A pin is the user's decision that a note is standing account context."""
    account_id = _account(client, "GlassHub")
    client.post(
        f"/api/v1/customers/{account_id}/notes",
        json={"title": "Procurement", "body": "Only buys through the reseller.", "pinned": True},
    )
    client.post(
        f"/api/v1/customers/{account_id}/notes",
        json={"title": "Aside", "body": "Their office moved floors.", "pinned": False},
    )
    service = client.app.state.runtime.customers  # type: ignore[attr-defined]
    context = client.portal.call(service.context, account_id)  # type: ignore[attr-defined]
    assert "Only buys through the reseller." in context
    assert "moved floors" not in context


def test_facts_actions_and_people_can_be_written_by_hand(client: TestClient) -> None:
    account_id = _account(client, "EHS")

    fact = client.post(
        f"/api/v1/customers/{account_id}/facts",
        json={"kind": "requirement", "content": "Data must stay in-region."},
    )
    assert fact.status_code == 201, fact.text
    # A person asserted it, so it carries full confidence and no evidence quote.
    assert fact.json()["confidence"] == 1.0
    assert fact.json()["evidence"]["quote"] == ""

    edited_fact = client.put(
        f"/api/v1/customers/facts/{fact.json()['id']}",
        json={
            "kind": "constraint",
            "content": "Data must stay in the UAE region.",
            "status": "disputed",
        },
    )
    assert edited_fact.status_code == 200
    assert edited_fact.json()["kind"] == "constraint"
    assert edited_fact.json()["status"] == "disputed"

    action = client.post(
        f"/api/v1/customers/{account_id}/actions",
        json={"description": "Send the DAC shape", "owner": "Ali", "due_at": "2026-08-10T00:00:00Z"},
    )
    assert action.status_code == 201, action.text
    assert action.json()["status"] == "open"
    action_id = action.json()["id"]

    edited_action = client.put(
        f"/api/v1/customers/actions/{action_id}",
        json={
            "description": "Send the DAC shape and the rate card",
            "owner": "Ali Ottoman",
            "due_at": "2026-08-12T00:00:00+04:00",
            "status": "open",
        },
    )
    assert edited_action.status_code == 200
    assert edited_action.json()["owner"] == "Ali Ottoman"
    # Due dates normalize to UTC like every other stored timestamp.
    assert edited_action.json()["due_at"] == "2026-08-11T20:00:00Z"

    person = client.post(
        f"/api/v1/customers/{account_id}/people",
        json={"name": "Dana", "role": "Platform lead", "organization": "EHS"},
    )
    assert person.status_code == 201, person.text
    person_id = person.json()["id"]

    # Adding the same name again corrects that contact rather than duplicating.
    again = client.post(
        f"/api/v1/customers/{account_id}/people",
        json={"name": "Dana", "role": "Head of platform", "organization": "EHS"},
    )
    assert again.status_code == 201
    assert again.json()["id"] == person_id
    assert again.json()["role"] == "Head of platform"

    renamed = client.put(
        f"/api/v1/customers/people/{person_id}",
        json={"name": "Dana Q.", "role": "Head of platform", "organization": "EHS"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Dana Q."

    detail = client.get(f"/api/v1/customers/{account_id}").json()
    assert len(detail["facts"]) == 1 and len(detail["actions"]) == 1
    assert len(detail["people"]) == 1

    assert client.delete(f"/api/v1/customers/facts/{fact.json()['id']}").status_code == 204
    assert client.delete(f"/api/v1/customers/actions/{action_id}").status_code == 204
    assert client.delete(f"/api/v1/customers/people/{person_id}").status_code == 204
    emptied = client.get(f"/api/v1/customers/{account_id}").json()
    assert emptied["facts"] == [] and emptied["actions"] == [] and emptied["people"] == []


def test_renaming_a_contact_onto_an_existing_name_is_refused(client: TestClient) -> None:
    account_id = _account(client, "Two contacts")
    first = client.post(
        f"/api/v1/customers/{account_id}/people", json={"name": "Sam"}
    ).json()
    client.post(f"/api/v1/customers/{account_id}/people", json={"name": "Alex"})
    collision = client.put(
        f"/api/v1/customers/people/{first['id']}",
        json={"name": "Alex", "role": "", "organization": ""},
    )
    assert collision.status_code == 409


def test_a_captured_note_can_be_corrected_and_removed(client: TestClient) -> None:
    account_id = _account(client, "Lancashire")
    captured = client.post(
        "/api/v1/customers/sources",
        json={"account_id": account_id, "title": "Call", "content": "Typo'd note"},
    )
    assert captured.status_code == 201
    source_id = captured.json()["id"]
    other = client.post(
        "/api/v1/customers/sources",
        json={"account_id": account_id, "title": "Second", "content": "Another note"},
    )
    assert other.status_code == 201

    fixed = client.put(
        f"/api/v1/customers/sources/{source_id}",
        json={"title": "Discovery call", "content": "Corrected note", "source_kind": "meeting"},
    )
    assert fixed.status_code == 200, fixed.text
    assert fixed.json()["title"] == "Discovery call"
    assert fixed.json()["source_kind"] == "meeting"
    assert "content_hash" not in fixed.json()

    # The de-duplication guarantee survives the edit.
    collision = client.put(
        f"/api/v1/customers/sources/{source_id}",
        json={"title": "Discovery call", "content": "Another note"},
    )
    assert collision.status_code == 409

    assert client.delete(f"/api/v1/customers/sources/{source_id}").status_code == 204
    assert client.delete(f"/api/v1/customers/sources/{source_id}").status_code == 404


def test_search_spans_every_account_and_record_kind(client: TestClient) -> None:
    first = _account(client, "Cohere Bank", industry="Financial services")
    second = _account(client, "Delta Logistics")
    client.post(
        f"/api/v1/customers/{second}/notes",
        json={"title": "Reranking", "body": "They asked about Cohere rerank latency."},
    )
    client.post(
        f"/api/v1/customers/{second}/facts",
        json={"kind": "model", "content": "Prefers Cohere Command A for summaries."},
    )
    client.post(
        f"/api/v1/customers/{first}/wins",
        json={"title": "Command A DAC live", "brief": "Cohere models on a dedicated cluster."},
    )
    client.post(
        "/api/v1/customers/sources",
        json={"account_id": second, "title": "Notes", "content": "Cohere embed v4 evaluated."},
    )

    found = client.get("/api/v1/customers/search", params={"q": "cohere"})
    assert found.status_code == 200, found.text
    hits = found.json()["hits"]
    kinds = {item["kind"] for item in hits}
    assert {"account", "note", "fact", "win", "source"} <= kinds
    # Accounts lead, because "take me to that customer" is the common intent.
    assert hits[0]["kind"] == "account"
    assert all(item["account_name"] for item in hits)
    assert any("rerank latency" in item["snippet"] for item in hits)

    scoped = client.get("/api/v1/customers/search", params={"q": "Delta"}).json()["hits"]
    assert [item["account_name"] for item in scoped] == ["Delta Logistics"]

    assert client.get("/api/v1/customers/search", params={"q": "  "}).json()["hits"] == []


def test_search_treats_wildcards_as_literal_text(client: TestClient) -> None:
    """A stray % in the query must not match every row in the store."""
    _account(client, "Percentage Co")
    _account(client, "Ordinary Co")
    hits = client.get("/api/v1/customers/search", params={"q": "%"}).json()["hits"]
    assert hits == []


def test_search_reports_truncation_rather_than_implying_completeness(
    client: TestClient,
) -> None:
    for index in range(6):
        _account(client, f"Widget {index} Holdings")
    result = client.get(
        "/api/v1/customers/search", params={"q": "widget", "limit": 3}
    ).json()
    assert len(result["hits"]) == 3
    assert result["truncated"] is True


def test_the_attention_queue_names_the_account_each_action_belongs_to(
    client: TestClient,
) -> None:
    overdue_account = _account(client, "Overdue Co")
    later_account = _account(client, "Later Co")
    client.post(
        f"/api/v1/customers/{later_account}/actions",
        json={"description": "Follow up next year", "due_at": "2099-01-01T00:00:00Z"},
    )
    client.post(
        f"/api/v1/customers/{overdue_account}/actions",
        json={"description": "Chase the signature", "due_at": "2020-01-01T00:00:00Z"},
    )
    client.post(
        f"/api/v1/customers/{later_account}/actions",
        json={"description": "Undated backlog item"},
    )

    dashboard = client.get("/api/v1/customers/dashboard").json()
    queue = dashboard["priority_actions"]
    assert [item["description"] for item in queue] == [
        "Chase the signature",
        "Follow up next year",
        "Undated backlog item",
    ]
    assert queue[0]["account_name"] == "Overdue Co"
    assert dashboard["overdue_actions"] == 1
