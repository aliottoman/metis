from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from waqil_api.attachment_text import AttachmentExtractionError
from waqil_api.context_links import (
    context_account,
    customer_record_url,
    link_run_history,
    safe_source_url,
)
from waqil_api.contracts import CustomerEvidenceV1, KnowledgeSnippetV1
from waqil_api.control_plane import (
    ControlPlane,
    _append_cited_sources,
    _document_sources,
)

ACCOUNTS = [
    {"id": "cust_north", "name": "Northwind Energy"},
    {"id": "cust_south", "name": "Southwind Energy"},
]


def history(*questions: str) -> list[dict]:
    return [{"role": "user", "content": question} for question in questions]


def test_short_follow_up_recovers_the_current_conversations_account() -> None:
    account, inferred = context_account(
        "What should we do next?",
        history("What is outstanding for Northwind Energy?"),
        ACCOUNTS,
    )
    assert account == ACCOUNTS[0]
    assert inferred


@pytest.mark.parametrize(
    "prompt,previous",
    [
        ("What is their status?", "Compare Northwind Energy and Southwind Energy"),
        ("What about Northwind Energy and Southwind Energy?", "Northwind Energy"),
        ("What is the weather?", "Northwind Energy"),
        ("Can you update their status?", "Northwind Energy"),
        ("What should we do next?", "Explain how rain forms"),
    ],
)
def test_ambiguity_new_topics_and_mutations_do_not_inherit_an_account(
    prompt, previous
) -> None:
    assert context_account(prompt, history("Northwind Energy", previous), ACCOUNTS) == (
        None,
        False,
    )


def test_an_assistants_account_mention_is_not_a_scope_decision() -> None:
    assert context_account(
        "What should we do next?",
        [{"role": "assistant", "content": "Northwind Energy"}],
        ACCOUNTS,
    ) == (None, False)


def test_comparing_a_full_name_with_an_acronym_does_not_imply_one_account() -> None:
    accounts = [
        *ACCOUNTS,
        {
            "id": "cust_mcit",
            "name": "Ministry of Communications and Information Technology (MCIT)",
        },
    ]
    for prompt, previous in [
        ("What is their status?", "Compare Northwind Energy and MCIT"),
        ("What about Northwind Energy and MCIT?", "Northwind Energy"),
    ]:
        assert context_account(prompt, history(previous), accounts) == (None, False)


async def test_conversation_links_use_persisted_runs_and_keep_project_provenance() -> (
    None
):
    database = SimpleNamespace(
        get_run=AsyncMock(
            return_value=SimpleNamespace(id="run_abc", conversation_id="conv_abc")
        )
    )
    snippets = [
        {
            "provider": "local",
            "source_label": "Run history",
            "rel_path": "2026-09/run_abc.md",
            "text": "# Review the deployment\n- Project: Customer portal",
            "symbol": "Review the deployment",
        }
    ]
    await link_run_history(database, snippets)
    assert snippets[0]["source_url"] == "/?conversation=conv_abc&run=run_abc"
    assert snippets[0]["source_label"] == "Project conversation · Customer portal"
    answer, dropped = _append_cited_sources(
        "The deployment was reviewed [1].", snippets
    )
    assert "[Review the deployment](/?conversation=conv_abc&run=run_abc)" in answer
    assert not dropped


async def test_missing_runs_and_other_documents_do_not_get_invented_links() -> None:
    database = SimpleNamespace(get_run=AsyncMock(return_value=None))
    snippets = [
        {
            "provider": "local",
            "source_label": "Private notes",
            "rel_path": "run_abc.md",
        },
        {
            "provider": "local",
            "source_label": "Run history",
            "rel_path": "run_deleted.md",
        },
    ]
    await link_run_history(database, snippets)
    assert all("source_url" not in snippet for snippet in snippets)
    database.get_run.assert_awaited_once_with("run_deleted")


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "//example.com",
        "/\\example.com",
        "/notes\nother",
        "https://user:pass@example.com/a",
        "/notes)injected",
        "http://[",
    ],
)
def test_source_urls_cannot_escape_the_link_context(url) -> None:
    assert safe_source_url(url) is None


def test_attached_document_citation_links_only_to_its_valid_upload_id() -> None:
    upload_id = "upl_" + "a" * 20
    sources = _document_sources(["brief.txt", "other.txt"], [upload_id, "../../other"])
    answer, _ = _append_cited_sources("The document says this [1].", sources)
    assert f"[brief.txt](/api/v1/uploads/{upload_id})" in answer
    assert "source_url" not in sources[1]


def ingestion_plane(settings, tmp_path, files):
    records = {}
    ids = []
    for index, (name, media_type, content) in enumerate(files):
        upload_id = f"upl_{index:020x}"
        ids.append(upload_id)
        path = tmp_path / name
        path.write_bytes(content)
        records[upload_id] = {
            "filename": name,
            "media_type": media_type,
            "blob_path": str(path),
        }
    cp = ControlPlane.__new__(ControlPlane)
    cp.settings = settings
    cp._guard = AsyncMock()
    cp._stage = AsyncMock()
    cp.events = SimpleNamespace(emit=AsyncMock())
    cp.database = SimpleNamespace(
        get_upload_record=AsyncMock(side_effect=records.__getitem__)
    )
    return cp, {
        "run_id": "run_ingest",
        "conversation_id": "conv_ingest",
        "attachment_ids": ids,
    }


async def test_mixed_attachment_sources_keep_the_ingest_order(
    settings, tmp_path
) -> None:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aLXkAAAAASUVORK5CYII="
    )
    cp, state = ingestion_plane(
        settings,
        tmp_path,
        [
            ("z-notes.txt", "text/plain", b"Notes"),
            ("diagram.png", "image/png", png),
            ("a-plan.md", "text/markdown", b"Plan"),
        ],
    )
    result = await cp._ingest(state)
    sources = _document_sources(result["attachment_filenames"], state["attachment_ids"])
    assert result["attachment_filenames"] == ["z-notes.txt", "diagram.png", "a-plan.md"]
    assert [(source["rel_path"], source["source_url"]) for source in sources] == [
        (name, f"/api/v1/uploads/{upload_id}")
        for name, upload_id in zip(
            result["attachment_filenames"], state["attachment_ids"]
        )
    ]
    assert "pixel content is not extracted" in result["attachment_text"]


async def test_unsupported_upload_aborts_ingest_instead_of_shifting_source_ids(
    settings, tmp_path
) -> None:
    cp, state = ingestion_plane(
        settings,
        tmp_path,
        [
            ("first.txt", "text/plain", b"Notes"),
            ("unsupported.bin", "application/octet-stream", b"unsupported"),
            ("third.txt", "text/plain", b"Later document"),
        ],
    )
    with pytest.raises(AttachmentExtractionError):
        await cp._ingest(state)
    assert [
        call.args[0] for call in cp.database.get_upload_record.await_args_list
    ] == state["attachment_ids"][:2]


def test_a_model_cannot_invent_meeting_provenance_in_extraction_evidence() -> None:
    with pytest.raises(ValidationError):
        CustomerEvidenceV1.model_validate(
            {"quote": "Invented", "meeting_id": "mtg_fake"}
        )


def test_direct_customer_record_links_survive_async_account_loading() -> None:
    assert (
        customer_record_url("cust_a", "facts", "fact-cfact_a")
        == "/customers?account=cust_a&tab=facts&fact=cfact_a#fact-cfact_a"
    )
    assert (
        customer_record_url("cust_a", "actions", "action-cact_a")
        == "/customers?account=cust_a&tab=actions&action=cact_a#action-cact_a"
    )


def retrieval_plane(settings, *, scope="auto", scoped=False, corpus_available=True):
    cp = ControlPlane.__new__(ControlPlane)
    cp.settings = settings
    cp._guard = AsyncMock()
    cp._stage = AsyncMock()
    cp._search_memories = AsyncMock(return_value=[])
    cp.profile = None
    cp.events = SimpleNamespace(emit=AsyncMock())
    cp.database = SimpleNamespace(
        list_active_tools=AsyncMock(return_value=[]),
        get_conversation_summary=AsyncMock(return_value=""),
        recent_messages_with_metadata=AsyncMock(
            return_value=(history("What is outstanding for Northwind Energy?"), False)
        ),
        list_customer_accounts=AsyncMock(return_value=ACCOUNTS),
    )
    cp.customers = SimpleNamespace(
        evidence=AsyncMock(
            return_value=[
                KnowledgeSnippetV1(
                    source_label="Reviewed fact",
                    provider="customer",
                    rel_path="Northwind#fact",
                    text="A reviewed requirement",
                    score=1.0,
                )
            ]
        )
    )
    cp.corpus = SimpleNamespace(
        available=lambda: corpus_available,
        retrieve=AsyncMock(
            return_value=[
                KnowledgeSnippetV1(
                    source_label="Notes",
                    provider="notion" if scope == "notion" or scoped else "local",
                    rel_path="approved.md",
                    text="A consented passage",
                    score=0.95,
                )
            ]
        ),
    )
    cp.answers = SimpleNamespace(
        enabled=lambda: True, retrieve=AsyncMock(return_value=[])
    )
    cp.web = None
    aliases = {"_knowledge_scope": scope}
    if scoped:
        aliases.update(
            {"_customer_id": "cust_north", "_customer_name": "Northwind Energy"}
        )
    state = {
        "prompt": "What should we do next?",
        "run_id": "run_test",
        "conversation_id": "conv_test",
        "user_message_id": "msg_test",
        "model_aliases": aliases,
        "evidence_plan": {
            "sources": ["private"],
            "public_queries": [],
            "focus_terms": [],
            "action": "answer",
            "needs_verification": False,
        },
        "evidence_method": "explicit_scope" if scope == "notion" else "semantic",
    }
    return cp, state


async def test_follow_up_enriches_retrieval_with_account_name_without_exporting_history(
    settings,
) -> None:
    cp, state = retrieval_plane(settings)
    result = await cp._retrieve(state)
    assert [item["provider"] for item in result["knowledge_snippets"]] == [
        "customer",
        "local",
    ]
    assert (
        cp.corpus.retrieve.call_args.args[0]
        == "Northwind Energy What should we do next?"
    )
    assert "outstanding" not in cp.corpus.retrieve.call_args.args[0]
    assert "_customer_id" not in state["model_aliases"]


async def test_notion_only_excludes_customer_ledger_and_answer_bank(settings) -> None:
    cp, state = retrieval_plane(settings, scope="notion", scoped=True)
    cp._search_memories.return_value = ["A private global memory"]
    cp.database.get_conversation_summary.return_value = "An unrelated account"
    result = await cp._retrieve(state)
    assert [item["provider"] for item in result["knowledge_snippets"]] == ["notion"]
    assert result["memories"] == []
    assert result["recent_messages"] == []
    assert result["conversation_summary"] == ""
    cp.customers.evidence.assert_not_awaited()
    cp.answers.retrieve.assert_not_awaited()
    assert cp.corpus.retrieve.call_args.kwargs["provider"] == "notion"


async def test_customer_scope_does_not_admit_global_answers_or_unavailable_corpus(
    settings,
) -> None:
    cp, state = retrieval_plane(settings, scoped=True, corpus_available=False)
    result = await cp._retrieve(state)
    assert [item["provider"] for item in result["knowledge_snippets"]] == ["customer"]
    cp.corpus.retrieve.assert_not_awaited()
    cp.answers.retrieve.assert_not_awaited()


async def test_auto_uses_web_for_live_public_question_and_returns_links(settings) -> None:
    cp, state = retrieval_plane(settings)
    state["prompt"] = "Who is the current CEO of Example Corp?"
    state["evidence_plan"]["sources"] = ["web"]
    state["evidence_plan"]["public_queries"] = ["Example Corp current CEO"]
    cp.web = SimpleNamespace(
        available=lambda: True,
        retrieve=AsyncMock(
            return_value=[
                KnowledgeSnippetV1(
                    source_label="Example Corp leadership",
                    provider="web",
                    rel_path="https://example.com/leadership",
                    source_url="https://example.com/leadership",
                    text="The current CEO is Alex Example.",
                    score=0.95,
                )
            ]
        ),
    )
    result = await cp._retrieve(state)
    assert [item["provider"] for item in result["knowledge_snippets"]] == [
        "web"
    ]
    cp.web.retrieve.assert_awaited_once_with(
        state["prompt"],
        queries=["Example Corp current CEO"],
        focus_terms=[],
    )
    cp.corpus.retrieve.assert_not_awaited()
    answer, _ = _append_cited_sources("Alex Example [1].", result["knowledge_snippets"])
    assert "[https://example.com/leadership]" in answer
    assert "](https://example.com/leadership)" in answer


async def test_auto_keeps_private_freshness_question_on_local_retrieval(settings) -> None:
    cp, state = retrieval_plane(settings)
    state["prompt"] = "What did we discuss yesterday in my notes?"
    cp.web = SimpleNamespace(available=lambda: True, retrieve=AsyncMock())
    result = await cp._retrieve(state)
    assert [item["provider"] for item in result["knowledge_snippets"]] == [
        "local"
    ]
    cp.web.retrieve.assert_not_awaited()
    cp.corpus.retrieve.assert_awaited_once()


def test_a_kept_meeting_action_can_be_read_and_cited_from_the_customer(client) -> None:
    runtime = client.app.state.runtime
    account_id = client.post(
        "/api/v1/customers", json={"name": "Meeting account"}
    ).json()["id"]

    async def prepare():
        meeting = await runtime.database.create_meeting(
            title="Planning call",
            audio_sha256="a" * 64,
            audio_filename="call.mp3",
            audio_media_type="audio/mpeg",
            audio_bytes=1,
        )
        await runtime.database.link_meeting_account(meeting["id"], account_id)
        await runtime.database.store_meeting_transcript(
            meeting["id"],
            transcript="Confirm the project timeline.",
            language="en",
            provider_request_id="fixture",
            duration_seconds=8.0,
            turns=[
                {
                    "ordinal": 0,
                    "speaker_id": "speaker_0",
                    "text": "Confirm the project timeline.",
                    "start_seconds": 2.0,
                    "end_seconds": 7.0,
                }
            ],
        )
        await runtime.database.store_meeting_proposals(
            meeting["id"],
            [
                {
                    "kind": "action",
                    "payload": {
                        "description": "Confirm the project timeline",
                        "owner": "Ali",
                    },
                    "turn_ordinal": 0,
                    "start": 2.0,
                    "end": 7.0,
                }
            ],
        )
        detail = await runtime.database.meeting_detail(meeting["id"])
        assert not await runtime.customers.evidence(account_id), (
            "An unreviewed meeting proposal is not customer evidence"
        )
        proposal = detail["proposals"][0]
        await runtime.database.decide_meeting_proposal(
            meeting["id"], proposal["id"], "accepted"
        )
        return meeting["id"], detail["turns"][0]["id"]

    meeting_id, turn_id = client.portal.call(prepare)
    response = client.get(f"/api/v1/customers/{account_id}")
    assert response.status_code == 200, response.text
    assert response.json()["actions"][0]["evidence"]["meeting_id"] == meeting_id
    assert response.json()["actions"][0]["evidence"]["turn_id"] == turn_id
    assert response.json()["actions"][0]["evidence"]["start"] == 2.0
    evidence = client.portal.call(runtime.customers.evidence, account_id)
    assert any(
        item.source_url == f"/meetings?meeting={meeting_id}&turn={turn_id}"
        for item in evidence
    )


def test_uploaded_sources_download_the_stored_blob_only(client, tmp_path) -> None:
    uploaded = client.post(
        "/api/v1/uploads",
        files={"file": ("brief.txt", b"Reviewed document", "text/plain")},
    )
    assert uploaded.status_code == 201, uploaded.text
    download = client.get(f"/api/v1/uploads/{uploaded.json()['id']}")
    assert download.status_code == 200
    assert download.content == b"Reviewed document"
    assert "attachment" in download.headers["content-disposition"]
    assert client.get("/api/v1/uploads/unknown").status_code == 404

    # A stored blob_path is not trusted as a file to read: the endpoint always
    # resolves the content-addressed blob from the stored digest.
    unrelated = tmp_path / "outside.txt"
    unrelated.write_text("Must stay unread")
    runtime = client.app.state.runtime
    record = client.portal.call(
        runtime.database.create_upload,
        "b" * 64,
        "outside.txt",
        "text/plain",
        16,
        str(unrelated),
    )
    assert client.get(f"/api/v1/uploads/{record.id}").status_code == 404


def test_reviewed_excerpts_link_to_their_source_without_injecting_raw_notes(
    client,
) -> None:
    runtime = client.app.state.runtime
    account_id = client.post(
        "/api/v1/customers", json={"name": "Source account"}
    ).json()["id"]
    source = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "title": "Discovery",
            "source_kind": "note",
            "content": "Need private deployment. Unreviewed speculation must stay out.",
        },
    )
    assert source.status_code == 201
    source_id = source.json()["id"]
    assert not client.portal.call(runtime.customers.evidence, account_id)
    extraction = {
        "summary": "Reviewed the deployment requirement.",
        "facts": [
            {
                "kind": "requirement",
                "content": "Requires a private deployment",
                "evidence": {
                    "source_id": source_id,
                    "quote": "Need private deployment.",
                },
            }
        ],
    }

    async def review():
        proposal = await runtime.database.create_customer_proposal(
            source_id=source_id,
            account_id=account_id,
            extraction=extraction,
            model="fixture",
            prompt_version="fixture",
        )
        await runtime.database.save_customer_proposal(proposal["id"], extraction)

    client.portal.call(review)
    snippets = client.portal.call(runtime.customers.evidence, account_id)
    fact = next(item for item in snippets if item.source_label == "Reviewed fact")
    assert "Reviewed source excerpt: Need private deployment." in fact.text
    assert (
        fact.source_url
        == f"/customers?account={account_id}&tab=sources&source={source_id}"
    )
    assert all("Unreviewed speculation" not in item.text for item in snippets)


def test_only_pinned_customer_notes_link_back_to_their_conversation(client) -> None:
    runtime = client.app.state.runtime
    account_id = client.post(
        "/api/v1/customers", json={"name": "Conversation account"}
    ).json()["id"]
    for pinned in [True, False]:
        response = client.post(
            f"/api/v1/customers/{account_id}/notes",
            json={
                "title": "A saved conversation",
                "body": "Reviewed context" if pinned else "Unpinned working text",
                "pinned": pinned,
                "origin": "chat",
                "origin_ref": "conv_origin",
            },
        )
        assert response.status_code == 201, response.text
    snippets = client.portal.call(runtime.customers.evidence, account_id)
    assert len(snippets) == 1
    assert snippets[0].source_url == "/?conversation=conv_origin"
    assert snippets[0].text.endswith("Reviewed context")
