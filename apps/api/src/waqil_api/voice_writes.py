"""What voice is allowed to add to the customer record, and when.

The rule this file exists to enforce is narrow and worth stating plainly: a
write happens only when the utterance in front of us is an explicit imperative
to create a supported record. Not when someone states a fact in passing, not
when an earlier turn asked for something similar, not when retrieved evidence
implies it, not when the model thinks it would be helpful.

    "Add a note to Batelco that the workshop moved to Thursday."   writes
    "The workshop moved to Thursday."                              does not

The second sentence is the one that matters. A microphone in a room hears a
great deal of speech that sounds like intent, and an assistant that files
records from overheard sentences is worse than one that files none — you stop
being able to trust the record, which was the only reason to keep it.

Detection is deterministic and the model is never asked whether to write. It
is asked, afterwards, only to shape the payload of a write the host already
decided on, and even then the host copies the transcript excerpt verbatim and
validates every field before anything is committed.

Five record types, create-only. There is no update here, no status change, no
delete, and `upsert_customer_person` — which would silently turn a duplicate
contact into an edit of the existing one — is deliberately not reachable.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

logger = logging.getLogger("waqil.voice.writes")

VOICE_RECORD_TYPES = ("note", "fact", "action", "person", "win")


class VoiceWriteRefused(ValueError):
    """A write the host declined, with a reason worth saying out loud.

    Its own type rather than a bare ValueError, because pydantic's
    ValidationError is *also* a ValueError — and a caller catching the base
    class turned a genuine contract bug into a polite "I need a bit more",
    which is exactly how a defect hides for a month.
    """


# The imperative has to lead. "Add a note that..." is an instruction; "we
# should add a note about that" is someone thinking out loud, and the leading
# anchor is most of what separates them.
_IMPERATIVE = re.compile(
    r"^(?:and\s+|also\s+|then\s+|please\s+|can you\s+|could you\s+|"
    r"i want you to\s+|i need you to\s+)*"
    r"(add|create|record|log|note|capture|save|file|jot|put|make|set)\b"
)

# What kind of record, by the noun the speaker used. Each pattern needs its
# own noun present: an imperative with no record type named is not a write, it
# is a sentence we did not understand.
_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("note", re.compile(r"\b(note|comment)\b")),
    (
        "action",
        re.compile(r"\b(action|task|to-?do|follow[- ]?up|reminder|next step)\b"),
    ),
    ("win", re.compile(r"\b(win|closed deal|deal we won)\b")),
    ("person", re.compile(r"\b(contact|person|stakeholder)\b")),
    ("fact", re.compile(r"\b(fact|detail|data ?point)\b")),
)

# There is deliberately no separate question filter. The leading-imperative
# anchor already is one — "what notes do we have on Batelco" has no verb in
# first position and never matches — and a question-word list actively got
# this wrong: "can you add a note that they signed" is a polite instruction,
# not an enquiry, and a gate that saw "can" first refused to file it.

# A due date, in the forms a person actually says. Parsed by the host, never
# by the model: a model that invents "next Friday" as a date is a commitment
# in someone's calendar that nobody made.
_RELATIVE_DAYS = {
    "today": 0,
    "tomorrow": 1,
    "monday": None,
    "tuesday": None,
    "wednesday": None,
    "thursday": None,
    "friday": None,
    "saturday": None,
    "sunday": None,
}
_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_DUE = re.compile(
    r"\b(?:by|before|due|on)\s+(today|tomorrow|" + "|".join(_WEEKDAYS) + r")\b"
)

# Ceilings on what one spoken sentence may commit. A voice note is a sentence
# or two; anything past this is a transcript that ran away, not a note.
MAX_NOTE_CHARS = 2_000
MAX_SHORT_CHARS = 400


@dataclass(frozen=True, slots=True)
class VoiceWriteIntent:
    """One explicit instruction to append, as the host understood it.

    `record_type` is empty when the utterance was plainly an instruction to
    file something but named more than one kind of record. That is a question
    to ask, not a write to attempt and not a request to quietly ignore — the
    speaker asked for something and deserves to hear which half we need.
    """

    record_type: str
    # The utterance, verbatim. This is the provenance, so it is never a
    # summary and never the model's restatement of what was said.
    transcript: str
    ambiguous_types: tuple[str, ...] = ()


def detect_write_intent(transcript: str) -> VoiceWriteIntent | None:
    """The explicit append this utterance asks for, or None.

    None is the overwhelmingly common answer and the safe one. Anything that
    is not unmistakably an instruction to file something is treated as a
    question, and questions are answered rather than recorded.
    """
    said = (transcript or "").strip()
    if not said:
        return None
    lowered = said.lower()
    if not _IMPERATIVE.match(lowered):
        return None
    matched = [name for name, pattern in _TYPE_PATTERNS if pattern.search(lowered)]
    if not matched:
        # An imperative naming no record type. Not understood, so not acted on.
        return None
    if len(matched) > 1:
        # "Add a note and an action" is two writes. One utterance commits at
        # most one record, so that a receipt can name exactly what it was and
        # a single Undo can reverse exactly that.
        logger.info("voice write names %s", " and ".join(matched))
        return VoiceWriteIntent("", said, ambiguous_types=tuple(matched))
    return VoiceWriteIntent(record_type=matched[0], transcript=said)


def parse_due_date(transcript: str, *, today: date | None = None) -> str | None:
    """ "by Friday" as a date, computed here rather than guessed by a model.

    Returns an ISO date, or None when nothing was said. A weekday always means
    the next one — saying "by Monday" on a Monday means the Monday coming, not
    the one now ending.
    """
    match = _DUE.search((transcript or "").lower())
    if not match:
        return None
    spoken = match.group(1)
    base = today or datetime.now(UTC).date()
    if spoken == "today":
        return base.isoformat()
    if spoken == "tomorrow":
        return (base + timedelta(days=1)).isoformat()
    target = _WEEKDAYS.index(spoken)
    ahead = (target - base.weekday()) % 7 or 7
    return (base + timedelta(days=ahead)).isoformat()


def validate_payload(record_type: str, payload: dict[str, object]) -> dict[str, str]:
    """The model's candidate, checked into something committable.

    Every field is coerced to a stripped string and bounded here, on the
    trusted side. A model that returns a dict where a title was asked for, or
    four thousand characters of enthusiasm, produces a short valid record
    rather than an exception or a runaway row.
    """
    if record_type not in VOICE_RECORD_TYPES:
        raise VoiceWriteRefused(f"voice cannot write a {record_type}")

    def text(key: str, limit: int = MAX_SHORT_CHARS, *, required: bool = False) -> str:
        raw = payload.get(key)
        value = " ".join(str(raw or "").split())[:limit]
        if required and not value:
            raise VoiceWriteRefused(f"a {record_type} needs a {key}")
        return value

    if record_type == "note":
        return {
            "title": text("title") or "Voice note",
            "body": text("body", MAX_NOTE_CHARS, required=True),
        }
    if record_type == "fact":
        return {
            "kind": text("kind") or "detail",
            "content": text("content", required=True),
        }
    if record_type == "action":
        due = payload.get("due_at")
        return {
            "description": text("description", required=True),
            "owner": text("owner") or "me",
            # Already parsed by the host; a value the model invented instead is
            # dropped rather than trusted.
            "due_at": str(due)
            if isinstance(due, str) and _ISO_DATE.match(due)
            else None,
        }
    if record_type == "person":
        return {
            "name": text("name", required=True),
            "role": text("role"),
            "organization": text("organization"),
        }
    return {"title": text("title", required=True), "brief": text("brief")}


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class VoiceWriteService:
    """Commits one append, and hands back the only key that can reverse it.

    The undo token lives here, in memory, rather than in the receipt table.
    That is the point of it: it is short-lived, single-use, and exists only for
    the browser session that just heard the record read out. A token that
    survived a restart would be a standing permission to delete a customer
    record, which is exactly what undo must not become.
    """

    # Long enough to notice a wrong record and click; short enough that the
    # ability to delete it does not linger. Undo is for "that's not what I
    # said", heard while you are still listening.
    UNDO_WINDOW_SECONDS = 900

    def __init__(self, database: Any) -> None:
        self.database = database
        self._tokens: dict[str, tuple[str, datetime]] = {}

    async def commit(
        self,
        *,
        record_type: str,
        account_id: str,
        account_name: str,
        payload: dict[str, str],
        transcript: str,
        turn: Any,
    ) -> Any:
        """One record, its receipt and its idempotency entry, in one transaction.

        The key is the session, the turn and the record type. A provider that
        retries the same Custom LLM request gets the original receipt back and
        the customer record gains nothing — which is the difference between a
        flaky network and two identical notes.
        """
        from .contracts import VoiceWriteReceiptV1

        key = f"voice:{turn.voice_session_id}:{turn.turn_id}:{record_type}"
        try:
            row = await self.database.commit_voice_append(
                idempotency_key=key,
                record_type=record_type,
                account_id=account_id,
                account_name=account_name,
                payload=payload,
                transcript_excerpt=transcript,
                voice_session_id=turn.voice_session_id,
                turn_id=turn.turn_id,
                run_id=turn.run_id,
            )
        except KeyError as error:
            raise VoiceWriteRefused("that account is no longer there") from error
        except ValueError as error:
            # The create-only guards (a person who already exists) speak in the
            # user's language; anything else is a bug and must not be dressed
            # up as one of them.
            raise VoiceWriteRefused(str(error)) from error
        receipt = _receipt_from_row(row, VoiceWriteReceiptV1)
        if receipt.undone_at is None:
            token = secrets.token_urlsafe(24)
            self._tokens[token] = (
                receipt.id,
                datetime.now(UTC) + timedelta(seconds=self.UNDO_WINDOW_SECONDS),
            )
            receipt = receipt.model_copy(update={"undo_token": token})
        return receipt

    async def undo(self, receipt_id: str, token: str) -> Any:
        """Reverse one receipt, once, for whoever holds its token.

        The token is looked up first and consumed whether or not the reversal
        succeeds. A token that survived a failed attempt would be a retry loop
        against the record's own guards.
        """
        from .contracts import VoiceWriteReceiptV1

        held = self._tokens.pop(token, None)
        self._expire()
        if held is None:
            raise PermissionError("that undo link is no longer valid")
        bound_receipt, expires = held
        if bound_receipt != receipt_id or expires <= datetime.now(UTC):
            raise PermissionError("that undo link is no longer valid")
        row = await self.database.undo_voice_append(receipt_id)
        return _receipt_from_row(row, VoiceWriteReceiptV1)

    def _expire(self) -> None:
        now = datetime.now(UTC)
        for token, (_, expires) in list(self._tokens.items()):
            if expires <= now:
                self._tokens.pop(token, None)


def _receipt_from_row(row: dict[str, Any], contract: Any) -> Any:
    """One stored row as a receipt, field by named field.

    Deliberately not a spread of the row. The contract forbids extra fields —
    which is what caught this — and a database column that quietly became a
    contract field would be a storage detail leaking into an API response.
    """
    raw = row.get("payload_json")
    payload = json.loads(raw) if isinstance(raw, str) else {}
    return contract.model_validate(
        {
            "id": row["id"],
            "record_type": row["record_type"],
            "record_id": row["record_id"],
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "source": row.get("source", "voice"),
            "voice_session_id": row.get("voice_session_id", ""),
            "provider_conversation_id": row.get("provider_conversation_id", ""),
            "turn_id": row.get("turn_id", ""),
            "run_id": row.get("run_id", ""),
            "transcript_excerpt": row.get("transcript_excerpt", ""),
            "payload": payload,
            "created_at": row["created_at"],
            "undone_at": row.get("undone_at"),
            "undone_by": row.get("undone_by", ""),
            "summary": receipt_sentence(
                str(row.get("record_type", "")),
                str(row.get("account_name", "")),
                payload,
            ),
        }
    )


def receipt_sentence(
    record_type: str, account_name: str, payload: dict[str, str]
) -> str:
    """What the card says, and what the voice acknowledges.

    Deliberately quotes the committed text rather than describing it. "Added a
    note to Batelco" tells you something happened; "Added note to Batelco:
    'Workshop moved to Thursday'" lets you notice it is wrong.
    """
    subject = (
        payload.get("body")
        or payload.get("content")
        or payload.get("description")
        or payload.get("name")
        or payload.get("title")
        or ""
    )
    trimmed = subject if len(subject) <= 120 else f"{subject[:117]}…"
    return f"Added {record_type} to {account_name}: “{trimmed}”"
