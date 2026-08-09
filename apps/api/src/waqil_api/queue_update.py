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
    # "update the account with this note", "file it as the following note" — the
    # note arrives attached to a filing verb rather than after one.
    r"|\b(?:with|as)\s+(?:this|the\s+following|a)\s+notes?\b"
    # "note:" / "note —" introduce a body the user is handing over to store.
    r"|\bnotes?\s*[:—]"
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


# "File this in the answer bank:", "Note for later —", "Remember this:". The
# clause is an instruction about where to put the statement, not part of it.
_FILING_PREFIX = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:file|save|capture|record|log|note|jot|remember|keep|store)\b"
    r"[^:—\n]{0,60}[:—]\s*",
    re.IGNORECASE,
)


def statement_without_filing_verb(prompt: str) -> str:
    """The statement a filing request carries, with the filing clause removed.

    Only a leading "…:" or "…—" clause goes, and only when it opens with a
    filing verb: everything after it is the user's own words, untouched. A
    message with no such clause is returned exactly as written."""
    return _FILING_PREFIX.sub("", prompt, count=1).strip() or prompt.strip()


def reports_work(prompt: str) -> bool:
    """True when a message settles or creates work — as opposed to only asking
    for something to be kept. A note-capture that reports no work has nothing
    to match against the action list, so it needs an account of its own."""
    return bool(_COMPLETION.search(prompt) or _NEW_WORK.search(prompt))


def is_queue_update_request(prompt: str) -> bool:
    """True when a message reports finished work, tracks new work, or files a
    note against an account."""
    return bool(reports_work(prompt) or _NOTE_CAPTURE.search(prompt))


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


_ALIAS_STOPWORDS = {"of", "and", "the", "for", "via", "in", "a", "an"}


def _account_match_keys(name: str) -> tuple[set[str], str]:
    """The acronyms an account can be named by, and the initials of its base
    name. "Ministry of Communications and Information Technology (MCIT)" yields
    acronyms {"mcit"} and initials "mcit" — so its own short form ranks above an
    account that merely mentions "(via MCIT)"."""
    acronyms = {a.lower() for a in re.findall(r"\b[A-Z]{2,}\b", name)}
    base = re.split(r"[(\[]", name, maxsplit=1)[0]
    initials = "".join(
        word[0]
        for word in re.findall(r"[A-Za-z]+", base)
        if word.lower() not in _ALIAS_STOPWORDS
    ).lower()
    return acronyms, initials


def _score_accounts(prompt: str, accounts: list[dict]) -> list[tuple[int, dict]]:
    """Every account the message plausibly names, scored and ranked. Full formal
    name typed out = 100; an account's own acronym (equal to its initials) = 70;
    a passing mention of another org's acronym in the name = 40. Shared by the
    "offer candidates" and "resolve one" callers so they agree."""
    haystack = _normalize(prompt)
    prompt_tokens = set(re.findall(r"[a-z0-9]{2,}", prompt.lower()))
    scored: list[tuple[int, dict]] = []
    for account in accounts:
        name = str(account.get("name", ""))
        if len(_normalize(name)) < 3:
            continue
        score = 0
        if _normalize(name) in haystack:
            score = 100  # the full formal name was typed out
        else:
            acronyms, initials = _account_match_keys(name)
            hit = acronyms & prompt_tokens
            if hit:
                # Its own acronym (equal to its initials) beats a name that only
                # mentions another org's acronym in parentheses.
                score = 70 if initials and initials in hit else 40
        if score:
            scored.append((score, account))
    scored.sort(key=lambda item: -item[0])
    return scored


def candidate_accounts(
    prompt: str, accounts: list[dict], scoped_id: str = ""
) -> list[dict]:
    """The accounts a note could be filed against, narrowed to those the message
    refers to — by full name, or by a short form the user actually types.

    People write "MCIT", not "Ministry of Communications and Information
    Technology (MCIT)"; requiring the full formal name matched nothing and
    dropped the note. So an account also matches by acronym, its OWN acronym
    ranked first. A conversation already scoped to an account needs no naming."""
    if scoped_id:
        scoped = [a for a in accounts if str(a.get("id")) == scoped_id]
        if scoped:
            return scoped
    return [account for _, account in _score_accounts(prompt, accounts)[:8]]


def resolve_account(
    prompt: str, accounts: list[dict], scoped_id: str = ""
) -> tuple[dict | None, list[dict]]:
    """Which single account an *unscoped* message is about, so it can run through
    the customer agent as if it had been scoped.

    Returns ``(account, [])`` when one match is clearly ahead (confident — file
    it and auto-scope), ``(None, [a, b, …])`` when several tie at the top
    (ambiguous — ask which one), and ``(None, [])`` when nothing is named. A
    scoped conversation always resolves to its own account."""
    if scoped_id:
        scoped = [a for a in accounts if str(a.get("id")) == scoped_id]
        if scoped:
            return scoped[0], []
    scored = _score_accounts(prompt, accounts)
    if not scored:
        return None, []
    top_score = scored[0][0]
    tied = [account for score, account in scored if score == top_score]
    if len(tied) == 1:
        return tied[0], []
    return None, tied[:5]


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
    # Accounts a new action may touch: those carrying an open action, plus any
    # the host explicitly offered for a note-capture.
    offered = {str(a.get("id") or "") for a in (offered_accounts or [])}
    accounts = {str(action.get("account_id") or "") for action in actions} | offered

    unmatched = list(proposal.unmatched)

    # The note. Its account must be one the host OFFERED — one the message
    # actually named, or the account the conversation is scoped to. Merely
    # carrying an open action is not being named: every open action's account
    # used to qualify, so a note that named no account at all was filed against
    # whichever account the model happened to see in the action list. A note
    # about Metis's own build coder landed on a Ukrainian bank that way.
    note = proposal.note
    if note is not None and note.account_id not in offered:
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
