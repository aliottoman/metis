"""Recordings into transcripts, and what a transcript is not allowed to become.

Two things carry most of the weight here. A failed stage must be retryable
without orphaning or duplicating the blob — so the tests kill stages
deliberately and check that the audio survives and the job resumes rather than
restarts. And nothing a meeting produces may become a customer record on its
own: a transcript is a machine's best guess at what a room said, and filing
commitments from a guess is how a customer record stops being worth reading.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from waqil_api.config import Settings
from waqil_api.contracts import CustomerAccountV1
from waqil_api.database import Database
from waqil_api.meetings import (
    MeetingError,
    MeetingService,
    monotonic,
    propose_actions,
    propose_decisions,
    turns_from_scribe,
)
from waqil_api.blob_store import BlobStore


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


def _scribe(*, speakers: tuple[str, ...] = ("speaker_0", "speaker_1")) -> dict:
    """A Scribe payload in the shape the service actually receives."""
    return {
        "language_code": "en",
        "text": "Hello Batelco. I'll send the sizing by Friday.",
        "_request_id": "req_123",
        "words": [
            {
                "text": "Hello",
                "start": 0.0,
                "end": 0.4,
                "type": "word",
                "speaker_id": speakers[0],
            },
            {
                "text": " ",
                "start": 0.4,
                "end": 0.5,
                "type": "spacing",
                "speaker_id": speakers[0],
            },
            {
                "text": "Batelco.",
                "start": 0.5,
                "end": 1.1,
                "type": "word",
                "speaker_id": speakers[0],
            },
            {
                "text": "I'll",
                "start": 1.5,
                "end": 1.8,
                "type": "word",
                "speaker_id": speakers[1],
            },
            {
                "text": " ",
                "start": 1.8,
                "end": 1.9,
                "type": "spacing",
                "speaker_id": speakers[1],
            },
            {
                "text": "send",
                "start": 1.9,
                "end": 2.2,
                "type": "word",
                "speaker_id": speakers[1],
            },
            {
                "text": " ",
                "start": 2.2,
                "end": 2.3,
                "type": "spacing",
                "speaker_id": speakers[1],
            },
            {
                "text": "the sizing by Friday.",
                "start": 2.3,
                "end": 3.4,
                "type": "word",
                "speaker_id": speakers[1],
            },
        ],
    }


class FakeSpeech:
    """Counts every provider call, so "resumed" is distinguishable from "restarted"."""

    available = True

    def __init__(self, *, fail_transcription: bool = False) -> None:
        self.fail_transcription = fail_transcription
        self.transcriptions = 0
        self.isolations = 0
        self.alignments = 0

    async def transcribe_meeting(self, audio, filename, media_type):
        self.transcriptions += 1
        if self.fail_transcription:
            from waqil_api.model_provider import ModelProviderError

            raise ModelProviderError("Scribe is having a moment")
        return _scribe()

    async def isolate_audio(self, audio, filename, media_type):
        self.isolations += 1
        return b"ID3isolated", "audio/mpeg"

    async def force_align(self, audio, text, *, filename="segment.wav"):
        self.alignments += 1
        return [
            {"text": "Hello", "start": 0.0, "end": 0.4},
            {"text": "Batelco", "start": 0.45, "end": 1.1},
        ]


class FakeRouter:
    def __init__(self, speech) -> None:
        self.elevenlabs = speech


class Customers:
    def __init__(self, accounts) -> None:
        self._accounts = accounts

    async def accounts(self):
        return list(self._accounts)


def _account(account_id: str, name: str) -> CustomerAccountV1:
    now = datetime.now(UTC)
    return CustomerAccountV1(
        id=account_id, name=name, aliases=[], created_at=now, updated_at=now
    )


async def _service(tmp_path, *, speech=None, account_names=None, **overrides):
    """A service over a real database — accounts included, because the
    meeting's account_id is a foreign key and a fake id would only ever pass
    a test that the schema would refuse."""
    settings = _settings(tmp_path, **overrides)
    database = Database(settings.database_path)
    await database.open()
    blobs = BlobStore(settings.blob_dir)
    accounts = None
    if account_names is not None:
        accounts = []
        for name in account_names:
            row = await database.create_customer_account(
                name=name, aliases=[], industry="", region=""
            )
            accounts.append(_account(row["id"], name))
    service = MeetingService(
        settings,
        database,
        blobs,
        model=FakeRouter(speech or FakeSpeech()),
        customers=Customers(accounts) if accounts is not None else None,
    )
    return service, database, blobs, accounts


async def _uploaded(service, database, blobs, *, title: str = "Batelco sync") -> dict:
    blob = await blobs.put_bytes(b"ID3" + b"audio" * 200, max_bytes=10_000_000)
    return await database.create_meeting(
        title=title,
        audio_sha256=blob.sha256,
        audio_filename="sync.mp3",
        audio_media_type="audio/mpeg",
        audio_bytes=blob.size,
    )


# -- turning a provider payload into turns ----------------------------------


def test_words_become_speaker_turns_with_their_own_bounds() -> None:
    turns = turns_from_scribe(_scribe())
    assert len(turns) == 2
    assert turns[0].speaker_id == "speaker_0"
    assert turns[0].text == "Hello Batelco."
    assert (turns[0].start, turns[0].end) == (0.0, 1.1)
    assert turns[1].speaker_id == "speaker_1"
    assert turns[1].text == "I'll send the sizing by Friday."
    assert turns[1].start == 1.5 and turns[1].end == 3.4
    # Spacing tokens punctuate the text and never become words with timings.
    assert all(word["text"].strip() for word in turns[0].words)


def test_a_pause_does_not_split_a_speaker_mid_sentence() -> None:
    """Spacing belongs to the run it sits inside, or every pause is a new turn."""
    payload = {
        "words": [
            {
                "text": "One",
                "start": 0.0,
                "end": 0.3,
                "type": "word",
                "speaker_id": "s0",
            },
            {
                "text": " ",
                "start": 0.3,
                "end": 1.9,
                "type": "spacing",
                "speaker_id": "s0",
            },
            {
                "text": "two",
                "start": 1.9,
                "end": 2.2,
                "type": "word",
                "speaker_id": "s0",
            },
        ]
    }
    turns = turns_from_scribe(payload)
    assert len(turns) == 1
    assert turns[0].text == "One two"


def test_a_payload_with_no_words_produces_no_turns() -> None:
    """No transcript is an honest answer; one long guess is not."""
    assert turns_from_scribe({}) == []
    assert turns_from_scribe({"words": []}) == []
    assert turns_from_scribe({"words": "not a list"}) == []


# -- the pipeline ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_recording_becomes_a_transcript_with_stages_recorded(
    tmp_path,
) -> None:
    service, database, blobs, accounts = await _service(tmp_path)
    try:
        meeting = await _uploaded(service, database, blobs)
        assert meeting["stage"] == "uploaded"

        result = await service.ingest(meeting["id"])
        assert result["stage"] == "ready"
        assert result["provider_request_id"] == "req_123"
        assert result["language"] == "en"
        assert result["duration_seconds"] == 3.4

        detail = await database.meeting_detail(meeting["id"])
        assert len(detail["turns"]) == 2
        # Every stage it passed through is observable afterwards.
        stages = [event["stage"] for event in detail["events"]]
        assert stages[0] == "uploaded" and stages[-1] == "ready"
        assert "transcribing" in stages and "analyzing" in stages
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_failed_stage_keeps_the_blob_and_resumes_rather_than_restarts(
    tmp_path,
) -> None:
    """The requirement the whole schema is shaped around."""
    speech = FakeSpeech(fail_transcription=True)
    service, database, blobs, accounts = await _service(
        tmp_path, speech=speech, meeting_audio_isolation=True
    )
    try:
        meeting = await _uploaded(service, database, blobs)
        digest = meeting["audio_sha256"]

        failed = await service.ingest(meeting["id"])
        assert failed["stage"] == "failed"
        assert "moment" in failed["error"]
        assert failed["attempts"] == 1
        # The audio is exactly where it was. Nobody re-uploads an hour of it.
        assert blobs.path_for(digest).is_file()

        speech.fail_transcription = False
        recovered = await service.ingest(meeting["id"])
        assert recovered["stage"] == "ready"
        assert recovered["audio_sha256"] == digest
        # One blob, not two: the store is content-addressed and the retry
        # resumed from the stage rather than from the upload.
        assert len(list(blobs.root.rglob("*"))) == len(
            [p for p in blobs.root.rglob("*") if p.is_file() or p.is_dir()]
        )
        assert speech.transcriptions == 2
        assert speech.isolations == 1, "retry resumes at transcription"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_isolation_is_skipped_by_default_and_never_fatal(tmp_path) -> None:
    speech = FakeSpeech()
    service, database, blobs, accounts = await _service(tmp_path, speech=speech)
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        assert speech.isolations == 0
    finally:
        await database.close()

    class BrokenIsolation(FakeSpeech):
        async def isolate_audio(self, audio, filename, media_type):
            self.isolations += 1
            from waqil_api.model_provider import ModelProviderError

            raise ModelProviderError("isolation is down")

    speech = BrokenIsolation()
    service, database, blobs, accounts = await _service(
        tmp_path / "b", speech=speech, meeting_audio_isolation=True
    )
    try:
        meeting = await _uploaded(service, database, blobs)
        result = await service.ingest(meeting["id"])
        # A noisy transcript beats no transcript.
        assert speech.isolations == 1
        assert result["stage"] == "ready"
        assert result["isolated_sha256"] is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_an_isolated_track_never_replaces_the_original(tmp_path) -> None:
    service, database, blobs, accounts = await _service(
        tmp_path, meeting_audio_isolation=True
    )
    try:
        meeting = await _uploaded(service, database, blobs)
        original = meeting["audio_sha256"]
        result = await service.ingest(meeting["id"])
        assert result["isolated_sha256"] not in (None, original)
        # The recording is evidence; the isolated track is a derived artifact.
        assert result["audio_sha256"] == original
        assert blobs.path_for(original).is_file()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_retranscription_replaces_turns_rather_than_interleaving_them(
    tmp_path,
) -> None:
    service, database, blobs, accounts = await _service(tmp_path)
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        await database.advance_meeting(meeting["id"], "transcribing")
        await service.ingest(meeting["id"])
        turns = await database.list_meeting_turns(meeting["id"])
        assert len(turns) == 2
        assert [turn["ordinal"] for turn in turns] == [0, 1]
    finally:
        await database.close()


# -- proposals ---------------------------------------------------------------


def test_commitments_are_proposed_with_the_words_that_produced_them() -> None:
    turns = turns_from_scribe(_scribe())
    proposals = propose_actions(turns)
    assert len(proposals) == 1
    assert proposals[0]["kind"] == "action"
    assert "send the sizing" in proposals[0]["payload"]["description"]
    # The span is what makes accepting it a four-second check.
    assert proposals[0]["turn_ordinal"] == 1
    assert proposals[0]["start"] == 1.5 and proposals[0]["end"] == 3.4


def test_ordinary_conversation_proposes_nothing() -> None:
    payload = {
        "words": [
            {
                "text": "The weather is nice",
                "start": 0.0,
                "end": 1.0,
                "type": "word",
                "speaker_id": "s0",
            },
        ]
    }
    turns = turns_from_scribe(payload)
    assert propose_actions(turns) == []
    assert propose_decisions(turns) == []


@pytest.mark.asyncio
async def test_a_derived_action_stays_a_proposal_and_creates_no_customer_record(
    tmp_path,
) -> None:
    """The line a meeting must never cross on its own."""
    service, database, blobs, accounts = await _service(tmp_path, account_names=[])
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        detail = await database.meeting_detail(meeting["id"])
        actions = [p for p in detail["proposals"] if p["kind"] == "action"]
        assert actions and all(p["status"] == "proposed" for p in actions)
        # Nothing reached the customer record.
        assert await database.list_customer_accounts() == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_clear_account_auto_links_and_an_unclear_one_only_proposes(
    tmp_path,
) -> None:
    service, database, blobs, accounts = await _service(
        tmp_path, account_names=["Batelco"]
    )
    try:
        meeting = await _uploaded(service, database, blobs)
        result = await service.ingest(meeting["id"])
        # Named essentially exactly, with nothing close behind it.
        assert result["account_id"] == accounts[0].id
        assert result["link_score"] >= 0.90
    finally:
        await database.close()

    service, database, blobs, accounts = await _service(
        tmp_path / "c",
        account_names=["Bahrain Petroleum Company", "Bahrain Telecommunications"],
    )
    try:
        meeting = await _uploaded(service, database, blobs, title="Bahrain catch-up")
        result = await service.ingest(meeting["id"])
        # Two plausible accounts: a proposal, never a link.
        assert result["account_id"] is None
        detail = await database.meeting_detail(meeting["id"])
        links = [p for p in detail["proposals"] if p["kind"] == "account_link"]
        assert links and links[0]["status"] == "proposed"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_re_analysis_does_not_resurrect_a_rejected_proposal(
    tmp_path,
) -> None:
    service, database, blobs, accounts = await _service(tmp_path, account_names=[])
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        detail = await database.meeting_detail(meeting["id"])
        first = detail["proposals"][0]
        await database.decide_meeting_proposal(meeting["id"], first["id"], "rejected")

        await database.advance_meeting(meeting["id"], "analyzing")
        await service.ingest(meeting["id"])
        after = await database.meeting_detail(meeting["id"])
        rejected = [p for p in after["proposals"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert not any(
            p["status"] == "proposed" and p["payload_json"] == first["payload_json"]
            for p in after["proposals"]
        )
    finally:
        await database.close()


# -- correction and alignment ------------------------------------------------


def test_timings_that_go_backwards_are_rejected() -> None:
    assert monotonic([{"start": 0.0, "end": 0.4}, {"start": 0.5, "end": 1.0}])
    assert not monotonic([{"start": 0.5, "end": 1.0}, {"start": 0.2, "end": 0.6}])
    assert not monotonic([{"start": 1.0, "end": 0.5}])
    # A word with no timing is skipped rather than treated as zero.
    assert monotonic([{"start": None, "end": None}, {"start": 0.1, "end": 0.4}])


@pytest.mark.asyncio
async def test_a_correction_keeps_the_original_and_records_the_edit(
    tmp_path,
) -> None:
    service, database, blobs, accounts = await _service(tmp_path)
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        turns = await database.list_meeting_turns(meeting["id"])
        first = turns[0]

        corrected = await service.correct_turn(
            meeting["id"], first["id"], "Hello Batelco B.S.C."
        )
        assert corrected["text"] == "Hello Batelco B.S.C."
        # What the provider actually heard survives beside the correction.
        assert corrected["original_text"] == "Hello Batelco."
        assert corrected["corrected_at"] is not None

        detail = await database.meeting_detail(meeting["id"])
        assert detail["turns"][0]["text"] == "Hello Batelco B.S.C."
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_alignment_never_receives_more_than_one_speakers_segment(
    tmp_path,
) -> None:
    """Forced alignment is single-speaker; a diarized span produces plausible lies."""
    sent: list[dict] = []

    class RecordingSpeech(FakeSpeech):
        async def force_align(self, audio, text, *, filename="segment.wav"):
            sent.append({"text": text, "bytes": len(audio)})
            return await super().force_align(audio, text, filename=filename)

    service, database, blobs, accounts = await _service(
        tmp_path, speech=RecordingSpeech()
    )
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        turns = await database.list_meeting_turns(meeting["id"])

        await service.correct_turn(meeting["id"], turns[0]["id"], "Hello Batelco BSC")
        if sent:
            # Only the corrected line's own words — never the second speaker's.
            assert sent[0]["text"] == "Hello Batelco BSC"
            assert "send the sizing" not in sent[0]["text"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_correction_still_lands_when_realignment_is_impossible(
    tmp_path,
) -> None:
    """A correct line with stale timings beats a correct line that will not save."""

    class NoAlignment(FakeSpeech):
        async def force_align(self, audio, text, *, filename="segment.wav"):
            from waqil_api.model_provider import ModelProviderError

            raise ModelProviderError("alignment is unavailable")

    service, database, blobs, accounts = await _service(tmp_path, speech=NoAlignment())
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        turns = await database.list_meeting_turns(meeting["id"])
        before = turns[0]["words_json"]

        corrected = await service.correct_turn(
            meeting["id"], turns[0]["id"], "Hello Batelco BSC"
        )
        assert corrected["text"] == "Hello Batelco BSC"
        # The old timings are kept rather than discarded or invented.
        assert corrected["words_json"] == before
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_an_empty_correction_and_an_unknown_line_are_refused(tmp_path) -> None:
    service, database, blobs, accounts = await _service(tmp_path)
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        with pytest.raises(MeetingError, match="needs words"):
            await service.correct_turn(meeting["id"], "mturn_x", "   ")
        with pytest.raises(MeetingError, match="not part of this meeting"):
            await service.correct_turn(meeting["id"], "mturn_nope", "something")
    finally:
        await database.close()


# -- speakers ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_named_speaker_stays_named(tmp_path) -> None:
    service, database, blobs, accounts = await _service(tmp_path)
    try:
        meeting = await _uploaded(service, database, blobs)
        await service.ingest(meeting["id"])
        await database.name_meeting_speaker(
            meeting["id"], "speaker_0", display_name="Sara Ahmed"
        )
        detail = await database.meeting_detail(meeting["id"])
        named = {row["speaker_id"]: row["display_name"] for row in detail["speakers"]}
        assert named["speaker_0"] == "Sara Ahmed"
        assert named["speaker_1"] == ""

        await database.name_meeting_speaker(
            meeting["id"], "speaker_0", display_name="Sara A."
        )
        detail = await database.meeting_detail(meeting["id"])
        assert len(detail["speakers"]) == 2, "renaming must not add a speaker"
    finally:
        await database.close()
