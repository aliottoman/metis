"""Recordings into transcripts, and transcripts into proposals nobody has agreed to yet.

The pipeline is four stages over one content-addressed blob:

    uploaded → isolating → transcribing → analyzing → ready

Each stage is committed before the next begins, which is the whole reason the
stages exist as data rather than as local variables. A crash during
transcription costs the transcription; the audio is already stored under its
own digest and the retry resumes from `transcribing` rather than asking anyone
to upload an hour of audio again. Nothing here re-uploads, and nothing here
writes a second blob for the same bytes — the store is content-addressed, so
the same recording twice is the same blob once.

What comes out the far end is deliberately not a customer record. A meeting
produces *proposals*: this might be the Batelco account, that sentence might
be an action. They stay proposals until someone clicks, because a transcript
is a machine's best guess at what a room said, and filing commitments from a
guess is how a customer record stops being worth reading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .model_provider import ModelProviderError
from .voice_accounts import AUTO_LINK_MARGIN, AUTO_LINK_SCORE, resolve_account

logger = logging.getLogger("waqil.meetings")

# Stages, in the order they run. Held as an ordered tuple because "resume from
# where it failed" needs to know what comes after what.
STAGES = ("uploaded", "isolating", "transcribing", "analyzing", "ready")

# How many times a stage will be retried before the job is left failed for a
# person to look at. Bounded: a recording that fails four times is failing for
# a reason retrying will not fix.
MAX_ATTEMPTS = 3

# Sentences that sound like a commitment. Deliberately a small deterministic
# net rather than a model pass over the whole transcript — every one of these
# is a proposal a person reads, so recall matters more than precision, and a
# host-side rule cannot invent a commitment that was never spoken.
_ACTION_HINTS = (
    "i'll ",
    "i will ",
    "we'll ",
    "we will ",
    "can you send",
    "could you send",
    "let's ",
    "lets ",
    "action item",
    "follow up",
    "by friday",
    "by monday",
    "next week",
    "send over",
    "send you",
)

_DECISION_HINTS = (
    "we decided",
    "we've decided",
    "we agreed",
    "the decision is",
    "we're going with",
    "we are going with",
    "we settled on",
)


class MeetingError(RuntimeError):
    """The message is user-facing."""


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One contiguous stretch of one speaker, with its words and their times."""

    ordinal: int
    speaker_id: str
    text: str
    start: float
    end: float
    words: list[dict[str, Any]]


def turns_from_scribe(payload: dict[str, Any]) -> list[SpeakerTurn]:
    """The provider's word list, grouped into speaker turns.

    Scribe returns a flat sequence of words, each with a speaker and a time.
    Turning that into readable turns is the host's job — grouping on speaker
    change, keeping spacing tokens out of the word timings, and letting the
    first and last word of a run define the turn's own bounds.

    Nothing is invented here. A word with no timing keeps none, and a payload
    with no words at all produces no turns rather than one long guess.
    """
    words = payload.get("words")
    if not isinstance(words, list):
        return []
    turns: list[SpeakerTurn] = []
    current: list[dict[str, Any]] = []
    current_speaker: str | None = None

    def flush() -> None:
        if not current:
            return
        spoken = [item for item in current if item.get("type") != "spacing"]
        text = "".join(str(item.get("text", "")) for item in current).strip()
        if not text:
            current.clear()
            return
        starts = [_number(item.get("start")) for item in spoken]
        ends = [_number(item.get("end")) for item in spoken]
        turns.append(
            SpeakerTurn(
                ordinal=len(turns),
                speaker_id=current_speaker or "speaker_0",
                text=text,
                start=min(
                    (value for value in starts if value is not None), default=0.0
                ),
                end=max((value for value in ends if value is not None), default=0.0),
                words=[
                    {
                        "text": str(item.get("text", "")),
                        "start": _number(item.get("start")),
                        "end": _number(item.get("end")),
                    }
                    for item in spoken
                ],
            )
        )
        current.clear()

    for word in words:
        if not isinstance(word, dict):
            continue
        speaker = str(word.get("speaker_id") or "speaker_0")
        if word.get("type") == "spacing":
            # Spacing belongs to whichever run it sits inside; it never starts
            # a new turn, which would split a sentence on a pause.
            current.append(word)
            continue
        if current_speaker is not None and speaker != current_speaker:
            flush()
        current_speaker = speaker
        current.append(word)
    flush()
    return turns


def _number(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def monotonic(words: list[dict[str, Any]]) -> bool:
    """Whether these timings move forward and never overlap.

    Checked after every realignment. Spliced timings that go backwards make a
    player jump, and a player that jumps is indistinguishable from a transcript
    that is wrong — so timings that fail this are discarded rather than stored.
    """
    last = -1.0
    for word in words:
        start, end = _number(word.get("start")), _number(word.get("end"))
        if start is None or end is None:
            continue
        if start < last or end < start:
            return False
        last = end
    return True


def propose_actions(turns: list[SpeakerTurn]) -> list[dict[str, Any]]:
    """Sentences that sound like commitments, with the words that produced them.

    Every one carries its turn and its time span, so accepting it is a
    four-second check against the audio rather than an act of faith.
    """
    proposals: list[dict[str, Any]] = []
    for turn in turns:
        lowered = turn.text.lower()
        for hint in _ACTION_HINTS:
            if hint in lowered:
                proposals.append(
                    {
                        "kind": "action",
                        "payload": {"description": turn.text[:400]},
                        "turn_ordinal": turn.ordinal,
                        "start": turn.start,
                        "end": turn.end,
                    }
                )
                break
    return proposals[:20]


def propose_decisions(turns: list[SpeakerTurn]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for turn in turns:
        lowered = turn.text.lower()
        if any(hint in lowered for hint in _DECISION_HINTS):
            proposals.append(
                {
                    "kind": "decision",
                    "payload": {"summary": turn.text[:400]},
                    "turn_ordinal": turn.ordinal,
                    "start": turn.start,
                    "end": turn.end,
                }
            )
    return proposals[:20]


class MeetingService:
    """Ingestion, correction, and the proposals a meeting leaves behind."""

    def __init__(
        self,
        settings: Settings,
        database: Any,
        blobs: Any,
        *,
        model: Any = None,
        customers: Any = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.blobs = blobs
        self.model = model
        self.customers = customers

    # -- ingestion --------------------------------------------------------

    def _speech(self) -> Any:
        speech = getattr(self.model, "elevenlabs", None)
        if speech is None or not speech.available:
            raise MeetingError("Transcribing a meeting needs WAQIL_ELEVENLABS_API_KEY.")
        return speech

    async def ingest(self, meeting_id: str) -> dict[str, Any]:
        """Run the job from wherever it currently is, and stop at the first failure.

        Deliberately resumable rather than restartable. `stage` is read from the
        row, not passed in, so a retry after a crash picks up the stage that
        did not finish and every stage before it stays done.
        """
        meeting = await self._require(meeting_id)
        if meeting["stage"] == "ready":
            return meeting
        if meeting["stage"] == "failed":
            resume = str(meeting.get("failed_stage") or "uploaded")
            await self.database.advance_meeting(
                meeting_id, resume, message=f"retrying {resume}"
            )
            meeting = await self._require(meeting_id)
        try:
            if meeting["stage"] in ("uploaded", "isolating"):
                meeting = await self._isolate(meeting)
            if meeting["stage"] == "transcribing":
                meeting = await self._transcribe(meeting)
            if meeting["stage"] == "analyzing":
                meeting = await self._analyze(meeting)
        except (MeetingError, ModelProviderError) as error:
            await self.database.fail_meeting(meeting_id, str(error)[:400])
            logger.info(
                "meeting %s failed at a stage: %s", meeting_id, str(error)[:200]
            )
            return await self._require(meeting_id)
        return await self._require(meeting_id)

    async def _isolate(self, meeting: dict[str, Any]) -> dict[str, Any]:
        """Optional, and never fatal. A noisy transcript beats no transcript."""
        meeting_id = meeting["id"]
        if not self.settings.meeting_audio_isolation:
            await self.database.advance_meeting(
                meeting_id, "transcribing", message="skipped isolation"
            )
            return await self._require(meeting_id)
        await self.database.advance_meeting(
            meeting_id, "isolating", message="cleaning the recording"
        )
        audio = self._read_blob(meeting["audio_sha256"])
        try:
            isolated, media_type = await self._speech().isolate_audio(
                audio, meeting["audio_filename"], meeting["audio_media_type"]
            )
        except (MeetingError, ModelProviderError) as error:
            # The original is still there and still transcribable, so this is a
            # degradation and not a failure.
            logger.info(
                "meeting %s isolation skipped: %s", meeting_id, str(error)[:200]
            )
            await self.database.advance_meeting(
                meeting_id, "transcribing", message="isolation unavailable"
            )
            return await self._require(meeting_id)
        stored = await self.blobs.put_bytes(
            isolated, max_bytes=self.settings.meeting_max_bytes
        )
        await self.database.set_meeting_isolated(meeting_id, stored.sha256)
        await self.database.advance_meeting(
            meeting_id, "transcribing", message=f"isolated ({media_type})"
        )
        return await self._require(meeting_id)

    async def _transcribe(self, meeting: dict[str, Any]) -> dict[str, Any]:
        meeting_id = meeting["id"]
        digest = meeting.get("isolated_sha256") or meeting["audio_sha256"]
        audio = self._read_blob(digest)
        payload = await self._speech().transcribe_meeting(
            audio, meeting["audio_filename"], meeting["audio_media_type"]
        )
        turns = turns_from_scribe(payload)
        if not turns:
            raise MeetingError("Nothing could be transcribed from that recording.")
        await self.database.store_meeting_transcript(
            meeting_id,
            transcript=str(payload.get("text") or "").strip(),
            language=str(payload.get("language_code") or ""),
            provider_request_id=str(payload.get("_request_id") or ""),
            duration_seconds=max((turn.end for turn in turns), default=0.0),
            turns=[
                {
                    "ordinal": turn.ordinal,
                    "speaker_id": turn.speaker_id,
                    "text": turn.text,
                    "start_seconds": turn.start,
                    "end_seconds": turn.end,
                    "words": turn.words,
                }
                for turn in turns
            ],
        )
        await self.database.advance_meeting(
            meeting_id, "analyzing", message=f"{len(turns)} speaker turns"
        )
        return await self._require(meeting_id)

    async def _analyze(self, meeting: dict[str, Any]) -> dict[str, Any]:
        """Everything a meeting suggests, as proposals and never as records."""
        meeting_id = meeting["id"]
        turns = await self._turns(meeting_id)
        proposals = propose_actions(turns) + propose_decisions(turns)
        link = await self._propose_link(meeting, turns)
        if link is not None:
            proposals.append(link)
        await self.database.store_meeting_proposals(meeting_id, proposals)
        # Auto-linking is the one thing here that can act without a click, and
        # only when the transcript names the account essentially exactly and
        # nothing else comes close — the same deterministic scale a spoken
        # write uses, with the same threshold and the same margin.
        if link is not None and link.get("auto"):
            await self.database.link_meeting_account(
                meeting_id, link["payload"]["account_id"], score=link.get("score")
            )
        await self.database.advance_meeting(
            meeting_id, "ready", message=f"{len(proposals)} proposals"
        )
        return await self._require(meeting_id)

    async def _propose_link(
        self, meeting: dict[str, Any], turns: list[SpeakerTurn]
    ) -> dict[str, Any] | None:
        """Which account this meeting was about, on the A.1 name-match scale.

        Deterministic names and aliases first, exactly as the brief asks: this
        is a string-identity question, and a deterministic answer stays
        auditable and stable where a model's float would not.
        """
        if self.customers is None:
            return None
        try:
            accounts = await self.customers.accounts()
        except Exception as error:  # noqa: BLE001 - a read failure is not a job failure
            logger.info("meeting %s account listing skipped: %s", meeting["id"], error)
            return None
        if not accounts:
            return None
        # Scored against the transcript's opening, where a meeting names who it
        # is with, plus the title someone typed. Scoring the whole hour would
        # let one passing mention of a competitor outweigh the actual host.
        opening = " ".join(turn.text for turn in turns[:12])
        haystack = f"{meeting.get('title') or ''} {opening}"
        resolution = resolve_account(
            [
                (account.id, account.name, tuple(getattr(account, "aliases", ()) or ()))
                for account in accounts
            ],
            haystack,
        )
        if resolution.best is None:
            return None
        evidence = next(
            (
                turn
                for turn in turns[:12]
                if resolution.best.name.split()[0].lower() in turn.text.lower()
            ),
            turns[0] if turns else None,
        )
        return {
            "kind": "account_link",
            "payload": {
                "account_id": resolution.best.account_id,
                "account_name": resolution.best.name,
            },
            "score": resolution.best.score,
            # Auto only at the confirmed threshold AND margin. Everything else
            # is one click, which is cheap; a wrong link is not.
            "auto": bool(
                resolution.decisive
                and resolution.best.score >= AUTO_LINK_SCORE
                and (
                    resolution.best.score
                    - (resolution.runner_up.score if resolution.runner_up else 0.0)
                )
                >= AUTO_LINK_MARGIN
            ),
            "turn_ordinal": evidence.ordinal if evidence else None,
            "start": evidence.start if evidence else None,
            "end": evidence.end if evidence else None,
        }

    # -- correction -------------------------------------------------------

    async def correct_turn(
        self, meeting_id: str, turn_id: str, text: str
    ) -> dict[str, Any]:
        """One corrected line, realigned against its own audio interval only.

        The bounded interval is the point. Forced alignment is a single-speaker
        service, so it is handed one speaker's segment and its own slice of
        audio — never a diarized span, which would produce timings that look
        right and are not. When the realignment cannot be trusted (no audio
        tooling, a provider failure, timings that do not move forward) the text
        is still corrected and the original word timings are kept: a correct
        line with slightly stale timings beats a correct line that makes the
        player jump.
        """
        corrected = " ".join((text or "").split())
        if not corrected:
            raise MeetingError("A corrected line still needs words in it.")
        turn = await self.database.get_meeting_turn(meeting_id, turn_id)
        if turn is None:
            raise MeetingError("That line is not part of this meeting.")

        words: list[dict[str, Any]] | None = None
        try:
            words = await self._realign(meeting_id, turn, corrected)
        except (MeetingError, ModelProviderError) as error:
            logger.info(
                "meeting %s realignment skipped: %s", meeting_id, str(error)[:200]
            )

        return await self.database.correct_meeting_turn(
            meeting_id,
            turn_id,
            text=corrected,
            words=words,
            realigned=words is not None,
        )

    async def _realign(
        self, meeting_id: str, turn: dict[str, Any], corrected: str
    ) -> list[dict[str, Any]] | None:
        """Word timings for the corrected segment, or None to keep the old ones."""
        from .audio_transcode import TranscodeError, slice_wav

        meeting = await self._require(meeting_id)
        digest = meeting.get("isolated_sha256") or meeting["audio_sha256"]
        audio = self._read_blob(digest)
        start = float(turn.get("start_seconds") or 0.0)
        end = float(turn.get("end_seconds") or 0.0)
        if end <= start:
            return None
        try:
            segment = await slice_wav(
                audio,
                meeting["audio_filename"],
                meeting["audio_media_type"],
                start=start,
                end=end,
            )
        except TranscodeError as error:
            raise MeetingError(str(error)) from error
        words = await self._speech().force_align(segment, corrected)
        shifted = [
            {
                "text": str(word.get("text", "")),
                "start": _shift(word.get("start"), start),
                "end": _shift(word.get("end"), start),
            }
            for word in words
        ]
        if not monotonic(shifted):
            # Spliced timings that go backwards make a player jump, which reads
            # as a wrong transcript. Discarded rather than stored.
            logger.info("meeting %s realignment was not monotonic", meeting_id)
            return None
        return shifted

    # -- reads ------------------------------------------------------------

    async def _require(self, meeting_id: str) -> dict[str, Any]:
        meeting = await self.database.get_meeting(meeting_id)
        if meeting is None:
            raise MeetingError("That meeting is not here.")
        return meeting

    async def _turns(self, meeting_id: str) -> list[SpeakerTurn]:
        rows = await self.database.list_meeting_turns(meeting_id)
        return [
            SpeakerTurn(
                ordinal=int(row["ordinal"]),
                speaker_id=str(row["speaker_id"]),
                text=str(row["text"]),
                start=float(row["start_seconds"] or 0.0),
                end=float(row["end_seconds"] or 0.0),
                words=json.loads(row.get("words_json") or "[]"),
            )
            for row in rows
        ]

    def _read_blob(self, digest: str) -> bytes:
        path = self.blobs.path_for(digest)
        if not path.is_file():
            raise MeetingError("The recording is no longer in the blob store.")
        return path.read_bytes()


def _shift(value: Any, offset: float) -> float | None:
    number = _number(value)
    return None if number is None else round(number + offset, 3)
