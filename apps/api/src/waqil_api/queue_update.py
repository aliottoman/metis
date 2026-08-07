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


def is_queue_update_request(prompt: str) -> bool:
    """True when a message reports finished work or asks to track new work."""
    return bool(_COMPLETION.search(prompt) or _NEW_WORK.search(prompt))


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
    proposal: QueueUpdateV1, actions: list[dict], *, today: datetime | None = None
) -> tuple[QueueUpdateV1, list[dict]]:
    """Keep only what the host itself offered.

    Returns the cleaned proposal and the matched action rows, so the caller
    renders its approval card from validated records rather than from anything
    the model wrote."""
    by_id = {str(action["id"]): action for action in actions}
    accounts = {str(action.get("account_id") or "") for action in actions}

    matched: list[dict] = []
    completed: list[ActionResolutionV1] = []
    unmatched = list(proposal.unmatched)
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
            # is in play; otherwise it needs saying explicitly.
            involved = {str(action.get("account_id") or "") for action in matched}
            if len(involved) == 1:
                account = involved.pop()
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
                "completed": completed,
                "new_actions": new_actions,
                "unmatched": unmatched[:10],
            }
        ),
        matched,
    )


def describe(proposal: QueueUpdateV1, matched: list[dict]) -> str:
    """The approval card's body: exactly what will change, in plain words."""
    by_id = {str(action["id"]): action for action in matched}
    lines: list[str] = []
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
