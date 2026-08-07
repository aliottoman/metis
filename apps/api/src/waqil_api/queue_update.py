"""Closing out work by saying so.

"I met Bank Pivdenny and walked them through the tenancy; still owe them the
pricing sheet" is how the work actually gets reported — and until now it went
into a chat log while the commitments it settles stayed open on another page.

The flow is deliberately the same posture as every other record change in
Metis: the model **proposes**, the host **validates**, the user **approves**,
and only then is anything written. What makes it safe is not the prompt but
the validation in the middle:

* The model may only resolve actions from a list the host supplied, keyed by
  id. An id that was not offered is dropped — a model cannot close a
  commitment it invented, or one belonging to an account this message never
  mentioned.
* New actions inherit the account of the work being reported; an account id
  the host did not offer is refused rather than guessed.
* Everything the model claimed but the host could not match is reported back
  as `unmatched`, so a silent partial match is impossible to mistake for a
  complete one.

The result is that the worst a bad extraction can do is propose the wrong
checkbox, in a card that names exactly what it will change, before anything
happens.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

from .contracts import ActionResolutionV1, NewActionV1, QueueUpdateV1

# Reporting finished work, or asking for a new commitment to be tracked.
# Conservative in the same spirit as the toolify, web, and document signals:
# a message *about* a meeting must not silently close anything.
_COMPLETION = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|just\s+|already\s+)?"
    r"(?:completed|finished|did|done|held|ran|delivered|sent|shared|closed|"
    r"wrapped\s+up|met\s+with|spoke\s+to|presented|demo(?:ed|ed|d)?)\b"
    r"|\b(?:mark|tick|check)\s+(?:it|that|this|them|these|off)\b"
    r"|\bdone\s+with\b",
    re.IGNORECASE,
)
_NEW_WORK = re.compile(
    r"\b(?:add|create|log|track|note)\b[^.?!\n]{0,40}"
    r"\b(?:to-?do|todo|task|action|follow[-\s]?up|reminder)\b",
    re.IGNORECASE,
)
# Filing a note against an account. "add this to their notes", "note for X",
# "capture/log this on the account", "record this against ...".
_NOTE_CAPTURE = re.compile(
    r"\b(?:add|save|put|append|attach)\b[^.?!\n]{0,40}\bnotes?\b"
    r"|\bnote\s+(?:for|on|about|against)\b"
    r"|\b(?:capture|record|log|file|jot)\s+(?:this|that|the\s+following|it)\b",
    re.IGNORECASE,
)
# Only when the user explicitly asks does a note get tidied before storage.
# Absent this, the note is stored exactly as written — evidence is not
# rewritten by a model on its way into the record.
_CLEANUP = re.compile(
    r"\b(?:clean|tidy|polish|format|neaten|structure|organi[sz]e|make\s+it\s+"
    r"(?:presentable|readable|nicer))\b",
    re.IGNORECASE,
)


def is_note_capture_request(prompt: str) -> bool:
    return bool(_NOTE_CAPTURE.search(prompt))


def wants_cleanup(prompt: str) -> bool:
    return bool(_CLEANUP.search(prompt))


def is_queue_update_request(prompt: str) -> bool:
    """True when a message reports finished work, tracks new work, or files a
    note against an account."""
    return bool(
        _COMPLETION.search(prompt)
        or _NEW_WORK.search(prompt)
        or _NOTE_CAPTURE.search(prompt)
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# An identifier-shaped token: has a digit, and enough length to be a real
# reference rather than a stray number. SR numbers, response ids, branch
# codes, shapes like 2xH200 all match; a bare "3" in prose does not.
_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{3,}")


def identifiers_preserved(tidied: str, original: str) -> bool:
    """Whether every identifier in a tidied note survives verbatim in the
    original message.

    The guard on the opt-in cleanup path: a model may reformat prose, but an
    SR number or a response id it altered by one character is evidence turned
    wrong. Comparison is on normalised tokens, so reformatting "SR 4-000..."
    to "SR4-000..." passes while a changed digit does not."""
    source = _normalize(original)
    for token in _ID_TOKEN.findall(tidied):
        if not any(ch.isdigit() for ch in token):
            continue
        if _normalize(token) not in source:
            return False
    return True


def candidate_accounts(
    prompt: str, accounts: list[dict], scoped_id: str = ""
) -> list[dict]:
    """The accounts a note could be filed against, narrowed to those the
    message actually names.

    108 accounts cannot all go in the prompt, and should not: the model must
    choose from the ones the user referred to, not the whole book. A
    conversation already scoped to an account needs no naming at all."""
    if scoped_id:
        scoped = [a for a in accounts if str(a.get("id")) == scoped_id]
        if scoped:
            return scoped
    haystack = _normalize(prompt)
    named = [
        account
        for account in accounts
        if len(_normalize(str(account.get("name", "")))) >= 3
        and _normalize(str(account.get("name", ""))) in haystack
    ]
    return named[:8]


def accounts_block(accounts: list[dict]) -> str:
    if not accounts:
        return "(no account named in the message)"
    return "\n".join(
        f"- id={account['id']} | {account.get('name', '')}" for account in accounts
    )


def candidates_block(actions: list[dict]) -> str:
    """The open actions, numbered by id, as the model's only menu."""
    if not actions:
        return "(no open actions are recorded)"
    return "\n".join(
        f"- id={action['id']} | account={action.get('account_name', '?')} "
        f"({action.get('account_id', '')}) | {action.get('description', '')}"
        for action in actions
    )


def validate(
    proposal: QueueUpdateV1,
    actions: list[dict],
    *,
    today: datetime | None = None,
    offered_accounts: list[dict] | None = None,
) -> tuple[QueueUpdateV1, list[dict]]:
    """Keep only what the host itself offered.

    Returns the cleaned proposal and the matched action rows, so the caller
    renders its approval card from validated records rather than from anything
    the model wrote."""
    by_id = {str(action["id"]): action for action in actions}
    # Accounts a new action or note may touch: those carrying an open action,
    # plus any the host explicitly offered for a note-capture.
    accounts = {str(action.get("account_id") or "") for action in actions}
    accounts |= {str(a.get("id") or "") for a in (offered_accounts or [])}

    unmatched = list(proposal.unmatched)

    # The note. Its account must be one the host offered — a model cannot file
    # a note against an account this message never named.
    note = proposal.note
    if note is not None and note.account_id not in accounts:
        unmatched.append(f"note for unknown account {note.account_id}")
        note = None
    # A new action with no account of its own inherits the note's account when
    # the note is the only account in play (the "add a note and a todo" shape).
    note_account = note.account_id if note is not None else ""

    matched: list[dict] = []
    completed: list[ActionResolutionV1] = []
    seen: set[str] = set()
    for resolution in proposal.completed:
        action = by_id.get(resolution.action_id)
        if action is None:
            # Named something that is not an open action of ours. Reported,
            # never applied.
            unmatched.append(f"unknown action id {resolution.action_id}")
            continue
        if resolution.action_id in seen:
            continue
        seen.add(resolution.action_id)
        matched.append(action)
        completed.append(resolution)

    new_actions: list[NewActionV1] = []
    for candidate in proposal.new_actions:
        account = candidate.account_id.strip()
        if not account:
            # Inherit the account of the work being reported when exactly one
            # is in play; otherwise fall back to the note's account, so "add a
            # note for X and a todo" attaches the todo to X.
            involved = {str(action.get("account_id") or "") for action in matched}
            if len(involved) == 1:
                account = involved.pop()
            elif note_account:
                account = note_account
        if account and account not in accounts:
            unmatched.append(f"new action for unknown account {account}")
            continue
        if not account:
            unmatched.append(f"no account for new action: {candidate.description[:80]}")
            continue
        # A commitment cannot be late before it is made. A model resolving
        # "next week" without knowing today's date produced 2025-06-23 in a
        # 2026 conversation, which the queue would then rank as overdue — a
        # fabricated urgency, ahead of real ones. The date is dropped rather
        # than guessed at, and the drop is reported.
        due = candidate.due_at
        if due is not None:
            when = due if due.tzinfo else due.replace(tzinfo=UTC)
            if when.date() < (today or datetime.now(UTC)).date():
                unmatched.append(
                    f"dropped a past due date ({when.date().isoformat()}) on: "
                    f"{candidate.description[:60]}"
                )
                candidate = candidate.model_copy(update={"due_at": None})
        new_actions.append(candidate.model_copy(update={"account_id": account}))

    return (
        proposal.model_copy(
            update={
                "note": note,
                "completed": completed,
                "new_actions": new_actions,
                "unmatched": unmatched[:10],
            }
        ),
        matched,
    )


def describe(
    proposal: QueueUpdateV1, matched: list[dict], accounts: list[dict] | None = None
) -> str:
    """The approval card's body: exactly what will change, in plain words."""
    by_id = {str(action["id"]): action for action in matched}
    account_names = {str(a.get("id")): str(a.get("name", "")) for a in (accounts or [])}
    lines: list[str] = []
    if proposal.note is not None:
        where = account_names.get(proposal.note.account_id, proposal.note.account_id)
        lines.append(f'File a note on {where}: "{proposal.note.title}"')
        lines.append("  (saved as you wrote it, then queued for analysis)")
    if proposal.completed:
        lines.append("Close these open actions:")
        for resolution in proposal.completed:
            action = by_id.get(resolution.action_id, {})
            account = action.get("account_name", "")
            note = f" — {resolution.note}" if resolution.note else ""
            lines.append(
                f"  • [{account}] {action.get('description', resolution.action_id)}{note}"
            )
    if proposal.new_actions:
        lines.append("Create these follow-ups:")
        for candidate in proposal.new_actions:
            owner = f" (owner: {candidate.owner})" if candidate.owner else ""
            due = f", due {candidate.due_at.date().isoformat()}" if candidate.due_at else ""
            lines.append(f"  • {candidate.description}{owner}{due}")
    if proposal.unmatched:
        lines.append("Not applied — nothing in the record matched:")
        lines.extend(f"  • {item}" for item in proposal.unmatched)
    return "\n".join(lines) or "Nothing in the record matches this message."


def has_changes(proposal: QueueUpdateV1) -> bool:
    """Whether anything survived validation worth an approval."""
    return bool(proposal.note or proposal.completed or proposal.new_actions)
