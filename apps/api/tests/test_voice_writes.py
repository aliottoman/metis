"""What voice may add to the customer record, and everything it may not.

The first block is the one that matters most. A microphone in a room hears a
great deal of speech that sounds like intent, and an assistant that files
records from overheard sentences destroys the thing it was keeping — you stop
being able to trust the record. So most of these tests assert that nothing
happened.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from waqil_api.config import Settings
from waqil_api.contracts import (
    CustomerAccountV1,
    VoiceAnswerV1,
    VoiceWriteCandidateV1,
)
from waqil_api.database import Database
from waqil_api.main import create_app
from waqil_api.policy import PolicyPermission
from waqil_api.speech_preference import SpeechPreferenceStore
from waqil_api.voice_graph import VOICE_WRITE_PERMISSIONS, VoiceGraph, VoiceTurn
from waqil_api.voice_writes import (
    VoiceWriteService,
    detect_write_intent,
    parse_due_date,
    validate_payload,
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        repo_root=Path(__file__).resolve().parents[3],
        model_backend="deterministic",
        reference_runner_mode="deterministic",
        allow_test_backends=True,
        **overrides,
    )


class DraftingModel:
    """Shapes a payload, and records every call so silence is provable."""

    def __init__(self, candidate: VoiceWriteCandidateV1 | None = None) -> None:
        self.candidate = candidate or VoiceWriteCandidateV1(
            title="Workshop moved",
            body="The workshop moved to Thursday.",
            content="The workshop moved to Thursday.",
            description="Send the sizing",
            name="Sara Ahmed",
            role="Head of infrastructure",
        )
        self.calls: list[dict] = []

    async def _structured(self, schema, **kwargs):
        self.calls.append({"schema": schema, **kwargs})
        if schema is VoiceWriteCandidateV1:
            return self.candidate
        # The read path asks for a different contract entirely, and a fake that
        # answered both with the same object would hide the fact that these are
        # two separate prompts with two separate ceilings.
        return VoiceAnswerV1(written="Nothing to add.", spoken="Nothing to add.")


class Customers:
    def __init__(self, accounts) -> None:
        self._accounts = accounts

    async def accounts(self):
        return list(self._accounts)

    async def evidence(self, account_id, *, compact=False):
        return []


def _account(account_id: str, name: str, *aliases: str) -> CustomerAccountV1:
    now = datetime.now(UTC)
    return CustomerAccountV1(
        id=account_id, name=name, aliases=list(aliases), created_at=now, updated_at=now
    )


async def _seeded(tmp_path: Path, *names: str) -> tuple[Database, list]:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    await database.open()
    accounts = []
    for name in names or ("Batelco",):
        row = await database.create_customer_account(
            name=name, aliases=[], industry="", region=""
        )
        accounts.append(_account(row["id"], name))
    return database, accounts


def _graph(tmp_path, database, accounts, *, model=None, confirm=False) -> VoiceGraph:
    settings = _settings(tmp_path)
    preference = SpeechPreferenceStore(settings)
    if confirm:
        settings.cohere_api_key = "c"
        preference.save("cohere", spoken_confirmation=True)
    return VoiceGraph(
        settings,
        model=model or DraftingModel(),
        speech_preference=preference,
        customers=Customers(accounts),
        writes=VoiceWriteService(database),
    )


async def _records(database: Database, account_id: str, kind: str) -> list:
    """One account's rows of a kind, straight from the store."""
    data = await database.customer_account_data(account_id)
    return list((data or {}).get(kind) or [])


def _turn(said: str, turn_id: str = "t_1") -> VoiceTurn:
    return VoiceTurn(transcript=said, voice_session_id="vs_1", turn_id=turn_id)


# -- what does not write -----------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "The workshop moved to Thursday",
        "I should send the sizing tomorrow",
        "They seemed happy with the proposal",
        "What notes do we have on Batelco?",
        "Remind me what the sizing said",
        "We should probably add a note about that",
        "It might be worth recording that somewhere",
        "Sara Ahmed runs infrastructure over there",
    ],
)
def test_ordinary_speech_is_never_an_instruction_to_file_something(said) -> None:
    """The sentence that separates a record you can trust from one you cannot."""
    assert detect_write_intent(said) is None


@pytest.mark.asyncio
async def test_a_statement_reaches_the_read_path_and_writes_nothing(
    tmp_path,
) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        rendition = await graph.answer(_turn("The workshop with Batelco moved"))
        assert rendition.intent != "customer_append"
        assert rendition.write is None
        assert await database.list_voice_receipts() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_an_earlier_turn_asking_for_a_note_cannot_write_later(
    tmp_path,
) -> None:
    """Intent lives in the current utterance and nowhere else."""
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        turn = VoiceTurn(
            transcript="And what about the sizing?",
            voice_session_id="vs_1",
            turn_id="t_2",
            history=("They: Add a note to Batelco that the workshop moved",),
        )
        rendition = await graph.answer(turn)
        assert rendition.write is None
        assert await database.list_voice_receipts() == []
    finally:
        await database.close()


# -- what does write ---------------------------------------------------------


@pytest.mark.asyncio
async def test_an_explicit_imperative_creates_exactly_one_record(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        said = "Add a note to Batelco that the workshop moved to Thursday"
        rendition = await graph.answer(_turn(said))

        assert rendition.intent == "customer_append"
        assert rendition.write is not None
        receipt = rendition.write
        assert receipt.record_type == "note"
        assert receipt.account_name == "Batelco"
        assert receipt.source == "voice"
        # Provenance is the words themselves, copied by the host.
        assert receipt.transcript_excerpt == said
        assert receipt.voice_session_id == "vs_1" and receipt.turn_id == "t_1"
        assert receipt.undo_token

        notes = await _records(database, accounts[0].id, "notes")
        assert len(notes) == 1
        assert notes[0]["origin"] == "voice"
        assert len(await database.list_voice_receipts()) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_every_record_type_carries_complete_provenance(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        for index, said in enumerate(
            [
                "Add a note to Batelco that the workshop moved",
                "Record a fact for Batelco about their region",
                "Create an action for Batelco to send the sizing by Friday",
                "Log a contact for Batelco, Sara Ahmed",
                "Record a win for Batelco",
            ]
        ):
            graph = _graph(tmp_path, database, accounts)
            rendition = await graph.answer(_turn(said, turn_id=f"t_{index}"))
            assert rendition.write is not None, said
            receipt = rendition.write
            assert receipt.account_id == accounts[0].id
            assert receipt.transcript_excerpt == said
            assert receipt.turn_id == f"t_{index}"
            assert receipt.created_at is not None
        assert len(await database.list_voice_receipts()) == 5
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_host_parses_the_due_date_rather_than_the_model(tmp_path) -> None:
    """A deadline a model invented is a commitment nobody made."""
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        await graph.answer(
            _turn("Create an action for Batelco to send the sizing by Friday")
        )
        actions = await _records(database, accounts[0].id, "actions")
        assert len(actions) == 1
        assert actions[0]["due_at"] is not None
        assert actions[0]["due_at"] == parse_due_date("by Friday")
    finally:
        await database.close()


# -- accounts ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ambiguous_account_asks_and_commits_nothing(tmp_path) -> None:
    database, accounts = await _seeded(
        tmp_path, "Bahrain Petroleum Company", "Bahrain Telecommunications"
    )
    try:
        graph = _graph(tmp_path, database, accounts)
        rendition = await graph.answer(_turn("Add a note to Bahrain that they called"))
        assert rendition.intent == "clarify"
        assert "Which account" in rendition.spoken
        assert rendition.write is None
        assert await database.list_voice_receipts() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_an_unnamed_account_asks_rather_than_guessing(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        rendition = await graph.answer(_turn("Add a note that the workshop moved"))
        assert rendition.intent == "clarify"
        assert rendition.write is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_naming_two_record_types_asks_which_one(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        rendition = await graph.answer(
            _turn("Add a note and an action for Batelco about the workshop")
        )
        assert rendition.intent == "clarify"
        assert "note or action" in rendition.spoken
        assert await database.list_voice_receipts() == []
    finally:
        await database.close()


# -- create-only -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_person_who_already_exists_is_refused_never_updated(
    tmp_path,
) -> None:
    """The one collision that must not become an edit.

    `upsert_customer_person` exists and would happily turn a duplicate contact
    into a silent rewrite of the existing one's role. Voice cannot reach it.
    """
    database, accounts = await _seeded(tmp_path)
    try:
        await database.upsert_customer_person(
            accounts[0].id, name="Sara Ahmed", role="CTO", organization="Batelco"
        )
        model = DraftingModel(
            VoiceWriteCandidateV1(name="Sara Ahmed", role="Head of infrastructure")
        )
        graph = _graph(tmp_path, database, accounts, model=model)
        rendition = await graph.answer(
            _turn("Log a contact for Batelco, Sara Ahmed, head of infrastructure")
        )

        assert rendition.write is None
        assert "already" in rendition.spoken.lower()
        people = await _records(database, accounts[0].id, "people")
        assert len(people) == 1
        # The existing role survived, which is the whole point.
        assert people[0]["role"] == "CTO"
    finally:
        await database.close()


def test_the_service_interface_exposes_no_delete_update_or_status_change() -> None:
    surface = {name for name in dir(VoiceWriteService) if not name.startswith("_")}
    assert surface == {"commit", "undo", "UNDO_WINDOW_SECONDS"}
    for forbidden in ("delete", "update", "edit", "status", "upsert", "apply"):
        assert not any(forbidden in name for name in surface)


def test_the_write_ceiling_adds_exactly_one_permission() -> None:
    from waqil_api.voice_graph import VOICE_PERMISSIONS

    added = VOICE_WRITE_PERMISSIONS - VOICE_PERMISSIONS
    assert added == {PolicyPermission.CUSTOMER_APPEND}


def test_a_payload_is_bounded_and_required_fields_are_enforced() -> None:
    huge = {"body": "x" * 9_000, "title": "y" * 900}
    bounded = validate_payload("note", huge)
    assert len(bounded["body"]) <= 2_000 and len(bounded["title"]) <= 400
    with pytest.raises(ValueError, match="needs a body"):
        validate_payload("note", {"title": "just a title"})
    with pytest.raises(ValueError, match="cannot write"):
        validate_payload("invoice", {"title": "x"})
    # A date the model invented in the wrong shape is dropped, not stored.
    assert (
        validate_payload("action", {"description": "x", "due_at": "next Friday"})[
            "due_at"
        ]
        is None
    )


# -- idempotency -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_retried_turn_returns_the_first_receipt_and_adds_no_row(
    tmp_path,
) -> None:
    """A flaky network must not become two identical notes."""
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        said = "Add a note to Batelco that the workshop moved"
        first = await graph.answer(_turn(said, turn_id="t_7"))
        second = await graph.answer(_turn(said, turn_id="t_7"))

        assert first.write is not None and second.write is not None
        assert first.write.id == second.write.id
        assert len(await _records(database, accounts[0].id, "notes")) == 1
        assert len(await database.list_voice_receipts()) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_the_record_and_its_receipt_commit_together(tmp_path) -> None:
    """A record with no receipt is a customer record from nowhere."""
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts)
        await graph.answer(_turn("Add a note to Batelco that the workshop moved"))
        notes = await _records(database, accounts[0].id, "notes")
        receipts = await database.list_voice_receipts()
        assert len(notes) == len(receipts) == 1
        assert receipts[0]["record_id"] == notes[0]["id"]
        assert receipts[0]["source"] == "voice"
    finally:
        await database.close()


# -- undo --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_is_single_use_and_leaves_the_receipt_behind(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        writes = VoiceWriteService(database)
        graph = VoiceGraph(
            _settings(tmp_path),
            model=DraftingModel(),
            speech_preference=SpeechPreferenceStore(_settings(tmp_path)),
            customers=Customers(accounts),
            writes=writes,
        )
        rendition = await graph.answer(
            _turn("Add a note to Batelco that the workshop moved")
        )
        receipt = rendition.write
        assert receipt is not None

        undone = await writes.undo(receipt.id, receipt.undo_token)
        assert undone.undone_at is not None
        assert await _records(database, accounts[0].id, "notes") == []
        # The audit row survives: "added and taken back" is not "never happened".
        assert len(await database.list_voice_receipts()) == 1

        with pytest.raises(PermissionError):
            await writes.undo(receipt.id, receipt.undo_token)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_undo_refuses_a_wrong_token_an_expired_one_and_another_receipt(
    tmp_path,
) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        writes = VoiceWriteService(database)
        graph = VoiceGraph(
            _settings(tmp_path),
            model=DraftingModel(),
            speech_preference=SpeechPreferenceStore(_settings(tmp_path)),
            customers=Customers(accounts),
            writes=writes,
        )
        first = (
            await graph.answer(_turn("Add a note to Batelco about A", "t_a"))
        ).write
        second = (
            await graph.answer(_turn("Add a note to Batelco about B", "t_b"))
        ).write
        assert first is not None and second is not None

        with pytest.raises(PermissionError):
            await writes.undo(first.id, "not-a-token")
        # A valid token, pointed at somebody else's receipt.
        with pytest.raises(PermissionError):
            await writes.undo(second.id, first.undo_token)
        # Both records are still there.
        assert len(await _records(database, accounts[0].id, "notes")) == 2

        expired = VoiceWriteService(database)
        expired._tokens["stale"] = (first.id, datetime.now(UTC) - timedelta(seconds=1))
        with pytest.raises(PermissionError):
            await expired.undo(first.id, "stale")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_undo_refuses_a_record_someone_edited_since(tmp_path) -> None:
    """Undo reverses a mistake you just heard, not work done in the meantime."""
    database, accounts = await _seeded(tmp_path)
    try:
        writes = VoiceWriteService(database)
        graph = VoiceGraph(
            _settings(tmp_path),
            model=DraftingModel(),
            speech_preference=SpeechPreferenceStore(_settings(tmp_path)),
            customers=Customers(accounts),
            writes=writes,
        )
        receipt = (
            await graph.answer(_turn("Add a note to Batelco that the workshop moved"))
        ).write
        assert receipt is not None
        await database.update_customer_note(
            receipt.record_id, title="Edited", body="Someone fixed this", pinned=True
        )
        with pytest.raises(ValueError, match="edited"):
            await writes.undo(receipt.id, receipt.undo_token)
        assert len(await _records(database, accounts[0].id, "notes")) == 1
    finally:
        await database.close()


def test_the_undo_route_needs_a_token_and_names_no_record(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/voice/receipts/vwr_nothing/undo", json={"undo_token": "made-up"}
        )
        assert response.status_code == 403
        # The contract has no record type and no record id to supply.
        rejected = client.post(
            "/api/v1/voice/receipts/vwr_nothing/undo",
            json={"record_type": "note", "record_id": "cnote_x"},
        )
        assert rejected.status_code == 422


# -- spoken confirmation -----------------------------------------------------


@pytest.mark.asyncio
async def test_confirmation_reads_back_and_waits_for_an_unambiguous_yes(
    tmp_path,
) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts, confirm=True)
        asked = await graph.answer(
            _turn("Add a note to Batelco that the workshop moved", "t_1")
        )
        assert asked.intent == "clarify"
        assert "Shall I?" in asked.spoken
        assert asked.write is None
        assert await database.list_voice_receipts() == []

        committed = await graph.answer(_turn("yes", "t_2"))
        assert committed.intent == "customer_append"
        assert committed.write is not None
        assert len(await _records(database, accounts[0].id, "notes")) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_anything_other_than_yes_drops_the_pending_write(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        for reply in (
            "actually never mind",
            "yeah but change the date",
            "what's next?",
        ):
            graph = _graph(tmp_path, database, accounts, confirm=True)
            await graph.answer(_turn("Add a note to Batelco about the workshop", "t_1"))
            after = await graph.answer(_turn(reply, "t_2"))
            assert after.write is None, reply
            assert await database.list_voice_receipts() == [], reply
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_yes_that_arrives_too_late_commits_nothing(tmp_path) -> None:
    database, accounts = await _seeded(tmp_path)
    try:
        graph = _graph(tmp_path, database, accounts, confirm=True)
        await graph.answer(_turn("Add a note to Batelco about the workshop", "t_1"))
        pending = graph._pending["vs_1"]
        graph._pending["vs_1"] = type(pending)(
            intent=pending.intent,
            account_id=pending.account_id,
            account_name=pending.account_name,
            payload=pending.payload,
            asked_at=datetime.now(UTC) - timedelta(minutes=5),
        )
        late = await graph.answer(_turn("yes", "t_2"))
        assert late.write is None
        assert await database.list_voice_receipts() == []
    finally:
        await database.close()


def test_the_due_date_scale_is_computed_not_guessed() -> None:
    wednesday = date(2026, 8, 12)
    assert parse_due_date("send it by Friday", today=wednesday) == "2026-08-14"
    assert parse_due_date("due tomorrow", today=wednesday) == "2026-08-13"
    # A weekday always means the next one — "by Monday" said on a Monday is the
    # Monday coming, not the one now ending.
    assert parse_due_date("by Monday", today=date(2026, 8, 10)) == "2026-08-17"
    assert parse_due_date("send the sizing") is None
