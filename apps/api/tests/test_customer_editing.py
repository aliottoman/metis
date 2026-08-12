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
        json={
            "title": "Standing context",
            "body": "Runs OCI Ashburn and Dubai.",
            "pinned": False,
        },
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
        json={
            "title": "Procurement",
            "body": "Only buys through the reseller.",
            "pinned": True,
        },
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
        json={
            "description": "Send the DAC shape",
            "owner": "Ali",
            "due_at": "2026-08-10T00:00:00Z",
        },
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

    assert (
        client.delete(f"/api/v1/customers/facts/{fact.json()['id']}").status_code == 204
    )
    assert client.delete(f"/api/v1/customers/actions/{action_id}").status_code == 204
    assert client.delete(f"/api/v1/customers/people/{person_id}").status_code == 204
    emptied = client.get(f"/api/v1/customers/{account_id}").json()
    assert (
        emptied["facts"] == [] and emptied["actions"] == [] and emptied["people"] == []
    )


def test_renaming_a_contact_onto_an_existing_name_is_refused(
    client: TestClient,
) -> None:
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
        json={
            "title": "Discovery call",
            "content": "Corrected note",
            "source_kind": "meeting",
        },
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
        json={
            "title": "Command A DAC live",
            "brief": "Cohere models on a dedicated cluster.",
        },
    )
    client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": second,
            "title": "Notes",
            "content": "Cohere embed v4 evaluated.",
        },
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

    scoped = client.get("/api/v1/customers/search", params={"q": "Delta"}).json()[
        "hits"
    ]
    assert [item["account_name"] for item in scoped] == ["Delta Logistics"]

    assert (
        client.get("/api/v1/customers/search", params={"q": "  "}).json()["hits"] == []
    )


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


def test_a_self_authored_note_files_without_an_approval(client: TestClient) -> None:
    """A note you wrote — a purely additive change, no closures — lands straight
    away with no approval card. The approval is kept only for a change the model
    inferred (a closure), not one you stated outright.
    """
    from waqil_api.contracts import CapturedNoteV1, QueueUpdateV1

    account_id = _account(client, "Northwind Logistics")
    control_plane = client.app.state.runtime.control_plane  # type: ignore[attr-defined]

    async def only_a_note(_schema: object, **_kwargs: object) -> QueueUpdateV1:
        return QueueUpdateV1(
            note=CapturedNoteV1(account_id=account_id, title="DAC sizing verified"),
        )

    # Advisory stage/queue events would FK against a run row this unit test
    # never creates; the record write under test emits none of its own.
    async def _swallow(*_args: object, **_kwargs: object) -> None:
        return None

    control_plane.model._structured = only_a_note  # type: ignore[attr-defined]
    control_plane.events.emit = _swallow  # type: ignore[attr-defined]

    result = client.portal.call(  # type: ignore[attr-defined]
        control_plane._queue_update,
        {
            "run_id": "run_additive_note",
            "conversation_id": "conv_additive_note",
            "prompt": "Add a note to Northwind Logistics: DAC sizing verified for 2xH100.",
            "model_aliases": {},
        },
    )

    # Filed immediately: a confirmation, and nothing left to grant.
    assert "approval_request" not in result
    assert result["response_text"].startswith("Done")

    # And it is on the record as a source waiting for analysis.
    sources = client.get(f"/api/v1/customers/{account_id}").json()["sources"]
    assert any(source["source_kind"] == "note" for source in sources)


def _capture(client: TestClient, account_id: str, content: str) -> str:
    created = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "source_kind": "note",
            "title": "Note",
            "content": content,
        },
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


def test_auto_analyze_extracts_a_fresh_note_on_a_cloud_pin(client: TestClient) -> None:
    """On a cloud pin, a captured note is analyzed automatically into a review
    proposal (waiting → review) — no manual Analyze click, no facts written."""
    account_id = _account(client, "Aurora Systems")
    source_id = _capture(
        client, account_id, "Agreed a 2xH100 DAC. Action: send pricing by Friday."
    )

    service = client.app.state.runtime.customers  # type: ignore[attr-defined]
    service._cloud_pinned = lambda: True  # pretend Command A+ is the pinned model

    client.portal.call(service.auto_analyze, source_id)  # type: ignore[attr-defined]

    source = next(
        s
        for s in client.get(f"/api/v1/customers/{account_id}").json()["sources"]
        if s["id"] == source_id
    )
    assert source["status"] == "review"  # analyzed, now awaiting your approval


def test_auto_analyze_is_a_noop_without_a_cloud_pin(client: TestClient) -> None:
    """A local pin leaves the note 'waiting' — auto-analysis must never force a
    local weight load on capture; the manual Analyze button still applies."""
    account_id = _account(client, "Borealis Freight")
    source_id = _capture(client, account_id, "Some captured content.")

    service = client.app.state.runtime.customers  # type: ignore[attr-defined]
    service._cloud_pinned = lambda: False

    client.portal.call(service.auto_analyze, source_id)  # type: ignore[attr-defined]

    source = next(
        s
        for s in client.get(f"/api/v1/customers/{account_id}").json()["sources"]
        if s["id"] == source_id
    )
    assert source["status"] == "waiting"


def test_note_capture_reads_update_with_this_note_phrasing() -> None:
    """The phrasing that slipped through: a filing verb the detector did not know
    ('update …'), with the note handed over after 'with this note'. It must read
    as a note to file, while plain questions that merely mention notes must not —
    or a scoped question would be misfiled instead of answered."""
    from waqil_api import queue_update

    assert queue_update.is_note_capture_request(
        'Update customer information with this note "7.08.2026 - throttling matches the GPU shape".'
    )
    assert queue_update.is_note_capture_request("file it as the following note")
    assert queue_update.is_note_capture_request("note: bring back the 1xH100 shape")
    assert not queue_update.is_note_capture_request(
        "what did we do about the throttling?"
    )
    assert not queue_update.is_note_capture_request("summarize the last meeting notes")


def test_an_account_is_matched_by_its_acronym_not_only_its_full_name() -> None:
    """The MCIT failure: people write 'MCIT', not 'Ministry of Communications and
    Information Technology (MCIT)'. The matcher must find the account by acronym
    and rank its OWN acronym above a name that merely mentions it, so the note
    files to the ministry rather than nowhere."""
    from waqil_api import queue_update

    accounts = [
        {
            "id": "mcit",
            "name": "Ministry of Communications and Information Technology (MCIT)",
        },
        {"id": "shura", "name": "Shura Council (via MCIT)"},
        {"id": "tasmu", "name": "TASMU (MCIT)"},
        {"id": "qatar", "name": "Qatar Ministry of Communications"},
    ]
    got = queue_update.candidate_accounts(
        "add a note to MCIT: kickoff done", accounts, ""
    )
    assert got and got[0]["id"] == "mcit", "the ministry's own acronym must rank first"
    assert {a["id"] for a in got} == {"mcit", "shura", "tasmu"}
    # A bare question names no account, and a scoped chat needs no naming.
    assert (
        queue_update.candidate_accounts("what is the DAC status?", accounts, "") == []
    )
    assert (
        queue_update.candidate_accounts("kickoff done", accounts, "tasmu")[0]["id"]
        == "tasmu"
    )


def test_resolve_account_is_confident_on_a_clear_acronym_and_defers_when_vague() -> (
    None
):
    """Routing an unscoped message needs one answer, not a list: a clear acronym
    resolves to a single account (file it, auto-scope); a vague reference resolves
    to nothing (the planner asks); a scoped id always wins."""
    from waqil_api import queue_update

    accounts = [
        {
            "id": "mcit",
            "name": "Ministry of Communications and Information Technology (MCIT)",
        },
        {"id": "shura", "name": "Shura Council (via MCIT)"},
        {"id": "defense", "name": "Ministry of Defense (Saudi)"},
    ]
    resolved, tied = queue_update.resolve_account("add a note to MCIT", accounts, "")
    assert resolved and resolved["id"] == "mcit" and tied == []

    resolved, tied = queue_update.resolve_account("add a note somewhere", accounts, "")
    assert resolved is None and tied == []

    resolved, _ = queue_update.resolve_account("kickoff", accounts, "defense")
    assert resolved and resolved["id"] == "defense"


def test_a_scoped_note_files_instead_of_being_answered(client: TestClient) -> None:
    """The bug this fixes: scoping a chat to the account you want to note against
    used to black-hole the note into the evidence-gated answer path, which then
    fact-checked the user's own words and refused them. Now the customer agent
    routes the message — it picks file_note — and the note is filed with the
    user's verbatim words: no fact-check, no refusal, no card."""
    from waqil_api.contracts import CustomerAgentStepV1, CustomerToolCallV1

    account_id = _account(client, "Wayfarer Freight")
    control_plane = client.app.state.runtime.control_plane  # type: ignore[attr-defined]

    async def _swallow(*_args: object, **_kwargs: object) -> None:
        return None

    control_plane.events.emit = _swallow  # type: ignore[attr-defined]

    # The agent reads the message and routes it to file_note.
    async def route_to_file_note(
        _schema: object, **_kwargs: object
    ) -> CustomerAgentStepV1:
        return CustomerAgentStepV1(
            calls=[CustomerToolCallV1(name="file_note", title="Throttling finding")]
        )

    control_plane.model._structured = route_to_file_note  # type: ignore[attr-defined]

    scoped_state = {
        "run_id": "run_scoped_note",
        "conversation_id": "conv_scoped_note",
        "prompt": 'Update customer information with this note "throttling matches the GPU shape".',
        "model_aliases": {"_customer_id": account_id},
    }

    # Routing: a scoped message goes through the customer agent, which picks
    # file_note, so it reaches the execute node — never the answer path.
    plan_result = client.portal.call(control_plane._plan, scoped_state)  # type: ignore[attr-defined]
    assert plan_result["route_kind"] == "customer_execute"
    assert [call["name"] for call in plan_result["customer_calls"]] == ["file_note"]

    # Filing: the note lands with the user's verbatim words, no card for the
    # note itself (its analysis surfaces its own apply card separately).
    file_result = client.portal.call(  # type: ignore[attr-defined]
        control_plane._customer_execute,
        {**scoped_state, "customer_calls": plan_result["customer_calls"]},
    )
    assert "approval_request" not in file_result
    assert "Filed" in file_result["response_text"]

    sources = client.get(f"/api/v1/customers/{account_id}").json()["sources"]
    note_source = next((s for s in sources if s["source_kind"] == "note"), None)
    assert note_source is not None
    # Stored as the user's own words, not the model's title or a rewrite.
    assert "throttling matches the GPU shape" in note_source["content"]


def test_notion_pages_map_to_records_and_dedupe(client: TestClient) -> None:
    """A Notion page whose title names one account is filed as a source and
    auto-analyzed into a review proposal; re-syncing the unchanged page files
    nothing new (content-hash dedup)."""
    from types import SimpleNamespace

    account_id = _account(client, "Helios Renewables")
    service = client.app.state.runtime.customers  # type: ignore[attr-defined]
    service._cloud_pinned = lambda: True

    page = SimpleNamespace(
        title="Helios Renewables — Q3 review",
        markdown="Kickoff done. Action: send the DAC sizing by Friday.",
        url="https://notion.so/helios",
        page_id="page-1",
    )
    filed = client.portal.call(service.ingest_notion_documents, [page])  # type: ignore[attr-defined]
    assert filed == 1
    notion_sources = [
        s
        for s in client.get(f"/api/v1/customers/{account_id}").json()["sources"]
        if s["source_kind"] == "notion"
    ]
    assert len(notion_sources) == 1 and notion_sources[0]["status"] == "review"

    # The ~12h re-sync of an unchanged page must add nothing.
    assert client.portal.call(service.ingest_notion_documents, [page]) == 0  # type: ignore[attr-defined]
    notion_after = [
        s
        for s in client.get(f"/api/v1/customers/{account_id}").json()["sources"]
        if s["source_kind"] == "notion"
    ]
    assert len(notion_after) == 1


def test_notion_pages_without_a_clear_account_are_left_alone(
    client: TestClient,
) -> None:
    """A page that names no account (or several) stays knowledge-only."""
    from types import SimpleNamespace

    _account(client, "Helios Renewables")
    service = client.app.state.runtime.customers  # type: ignore[attr-defined]
    service._cloud_pinned = lambda: True

    generic = SimpleNamespace(
        title="Weekly planning notes", markdown="Some notes.", url="", page_id="p2"
    )
    assert client.portal.call(service.ingest_notion_documents, [generic]) == 0  # type: ignore[attr-defined]
