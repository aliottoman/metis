"""The attention queue — one ranked answer to "what needs me today?".

Metis has excellent workbenches and no front door. Pending memory proposals
live on one page, notes awaiting analysis on another, asset trust on a third,
approvals on a fourth. Nothing is lost, but remembering where work waits is
itself work, and the counts grow until none of them mean anything.

This aggregates all of it once and ranks it by **consequence, not count**. That
distinction is the whole design: fifty assets that have needed setup for months
are not today's problem, and an overdue commitment to a customer is, so a
straight tally would bury the thing that actually matters. Ranking makes the
headline honest — if it says three things need you, those are the three.

Deferral is a first-class outcome beside approve and reject. Without it a queue
becomes a list of things you have decided not to do and cannot silence, which
is how a review surface stops being opened at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import asyncio

from pydantic import Field

from .contracts import AttentionFeedV1, AttentionItemV1, Contract
from .database import Database

# What a delay actually costs, expressed as a base weight. A commitment made to
# a customer outranks the app's own housekeeping — the queue is ordered by who
# is waiting, not by which subsystem produced the row.
_BASE_WEIGHT = {
    "run_approval": 90,  # a run is stopped mid-flight, holding its work
    "customer_action": 80,  # a promise to someone outside this machine
    "customer_opportunity": 70,  # a reviewed need without a recorded follow-up
    "customer_note": 60,  # captured intelligence not yet in the record
    "tool_proposal": 45,  # a capability waiting to become real
    "answer_atom": 40,  # reusable knowledge, one review from being real
    "memory": 35,  # improves future answers; nothing breaks meanwhile
    "asset_trust": 25,  # a reviewed recipe waiting to be trusted
    "stale_source": 20,  # knowledge is thinner than it could be
}

_KIND_LABEL = {
    "run_approval": "Approval",
    "customer_action": "Customer action",
    "customer_opportunity": "Opportunity signal",
    "customer_note": "Customer note",
    "tool_proposal": "Tool proposal",
    "answer_atom": "Answer to keep",
    "memory": "Memory proposal",
    "asset_trust": "Asset trust",
    "stale_source": "Knowledge source",
}


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _age_days(value: Any, now: datetime) -> float:
    created = _parse(value)
    return max((now - created).total_seconds() / 86_400, 0.0) if created else 0.0


_NEXT_STEP = {
    "run_approval": "Open the paused run, inspect the proposed change, then approve or reject it.",
    "customer_note": "Read the captured note, analyze it when ready, and review the proposed updates.",
    "tool_proposal": "Review the tool's purpose, permissions, and proposed behavior before deciding.",
    "answer_atom": "Compare the answer with its source, then keep it if it is reusable and accurate.",
    "memory": "Check the proposed fact against its source before remembering it.",
    "asset_trust": "Inspect the launch commands and trust the recipe only if they match your intent.",
    "stale_source": "Check the source's status and indexing permissions, then retry if needed.",
}


def _neglected(item: AttentionItemV1, now: datetime) -> bool:
    """A future deadline is planned work; an old creation date is not neglect."""
    return item.kind == "customer_action" and (
        item.overdue or (item.due_at is None and _age_days(item.updated_at, now) >= 7)
    )


class AttentionService:
    """Builds the ranked queue. Read-only apart from deferral."""

    def __init__(self, database: Database, assets: Any | None = None) -> None:
        self.database = database
        # Optional: an absent asset library narrows the queue, never fails it.
        self.assets = assets

    async def defer(self, item_key: str, kind: str, days: int, reason: str = "") -> str:
        until = datetime.now(UTC) + timedelta(days=max(days, 1))
        stamp = until.isoformat().replace("+00:00", "Z")
        await self.database.defer_attention_item(item_key, kind, stamp, reason)
        return stamp

    async def undefer(self, item_key: str) -> None:
        await self.database.clear_attention_deferral(item_key)

    async def _assets_awaiting_trust(self) -> list[Any]:
        """Assets whose launch recipe is configured but not approved.

        Only these — "needs setup" is work the user has not started, not a
        decision waiting on them, and putting fifty of those in a review queue
        is how a review queue stops being opened."""
        if self.assets is None:
            return []
        try:
            catalog = await self.assets.list()
        except Exception:  # noqa: BLE001 - the queue must survive a cold library
            return []
        return [
            asset
            for asset in catalog
            if getattr(asset, "launch_configured", False)
            and not getattr(asset, "launch_approved", False)
        ][:10]

    async def feed(self, *, top: int = 3) -> AttentionFeedV1:
        data = await self.database.attention_data()
        now = _parse(data.get("now")) or datetime.now(UTC)
        deferrals: dict[str, str] = data.get("deferrals", {})
        items: list[AttentionItemV1] = []

        def add(
            *,
            key: str,
            kind: str,
            title: str,
            detail: str = "",
            href: str = "",
            created_at: Any = None,
            due_at: Any = None,
            account_id: str | None = None,
            account_name: str = "",
            updated_at: Any = None,
            source_href: str = "",
            why_now: str = "",
            next_step: str = "",
            prepared_prompt: str = "",
            bump: float = 0.0,
        ) -> None:
            due = _parse(due_at)
            overdue = bool(due and due < now)
            # Age breaks ties within a kind so the oldest item surfaces first,
            # but it is capped: waiting three months must not let housekeeping
            # outrank a commitment that came due this morning.
            score = _BASE_WEIGHT.get(kind, 10) + bump
            score += min(_age_days(created_at, now), 14.0) * 0.7
            if overdue:
                score += 25 + min((now - due).total_seconds() / 86_400, 14.0) * 1.5
            elif due:
                hours = (due - now).total_seconds() / 3600
                if hours <= 24:
                    score += 15
                elif hours <= 72:
                    score += 6
            next_step = next_step or _NEXT_STEP.get(
                kind, "Review the source and agree a next step."
            )
            why_now = why_now or (
                "Work is paused until you make this decision."
                if kind == "run_approval"
                else "This review can make existing work useful again."
            )
            source_href = source_href or href
            prepared_prompt = (
                prepared_prompt
                or f"{title}\n\nNext step: {next_step}\nSource: {source_href}"
            )
            items.append(
                AttentionItemV1(
                    key=key,
                    kind=kind,
                    kind_label=_KIND_LABEL.get(kind, kind),
                    title=title[:200],
                    detail=detail[:400],
                    href=href,
                    account_id=account_id,
                    due_at=due,
                    created_at=_parse(created_at),
                    overdue=overdue,
                    priority=round(score, 2),
                    deferred_until=_parse(deferrals.get(key)),
                    account_name=account_name,
                    updated_at=_parse(updated_at),
                    source_href=source_href,
                    why_now=why_now,
                    next_step=next_step,
                    prepared_prompt=prepared_prompt,
                )
            )

        for run in data.get("waiting_runs", []):
            prompt = " ".join(str(run.get("prompt") or "").split())
            add(
                key=f"run_approval:{run['id']}",
                kind="run_approval",
                title=prompt[:120] or "A run is waiting for approval",
                detail="Paused at an approval gate — its work is held until you decide.",
                href=f"/?conversation={run['conversation_id']}&run={run['id']}",
                created_at=run.get("created_at"),
            )

        for action in data.get("open_actions", []):
            owner = str(action.get("owner") or "").strip()
            account_name = str(action.get("account_name") or "")
            description = str(action.get("description") or "Open action")
            href = f"/customers?account={action['account_id']}&tab=actions&action={action['id']}"
            source_href = (
                f"/customers?account={action['account_id']}&tab=sources&source={action['source_id']}"
                if action.get("source_id")
                else href
            )
            due = _parse(action.get("due_at"))
            untouched = int(_age_days(action.get("updated_at"), now))
            why_now = (
                f"Due {due.strftime('%b %d')}; the commitment is still open."
                if due and due < now
                else f"Due {due.strftime('%b %d')}; prepare the next step before the deadline."
                if due
                else f"No recorded update in {untouched} days, and no due date is set."
                if untouched >= 7
                else "An open commitment has no due date yet."
            )
            next_step = (
                f"Review the source, confirm {owner}'s next step, and record completion or a revised due date."
                if owner
                else "Review the source, confirm who owns the next step, and record completion or a due date."
            )
            add(
                key=f"customer_action:{action['id']}",
                kind="customer_action",
                title=description,
                detail=f"{action.get('account_name', '')}"
                + (f" · owner {owner}" if owner else ""),
                href=href,
                created_at=action.get("created_at"),
                due_at=action.get("due_at"),
                account_id=str(action.get("account_id") or "") or None,
                account_name=account_name,
                updated_at=action.get("updated_at"),
                source_href=source_href,
                why_now=why_now,
                next_step=next_step,
                prepared_prompt=(
                    f"{account_name}: {description}\n\n"
                    f"1. Read the original commitment: {source_href}\n"
                    f"2. {next_step}\n"
                    "3. Mark complete only after the work is finished.\n\n"
                    f"Follow-up draft:\nFollowing up on {description.rstrip('.')}. "
                    "Could you confirm the current status and next step? "
                    "If the timeline has changed, let's agree an updated date."
                ),
                bump=8 if not due and untouched >= 7 else 0,
            )

        for signal in data.get("opportunity_signals", []):
            content = str(signal.get("content") or "Recorded customer need")
            account_name = str(signal.get("account_name") or "")
            href = f"/customers?account={signal['account_id']}&tab=facts&fact={signal['id']}"
            source_href = (
                f"/customers?account={signal['account_id']}&tab=sources&source={signal['source_id']}"
                if signal.get("source_id")
                else href
            )
            add(
                key=f"customer_opportunity:{signal['id']}",
                kind="customer_opportunity",
                title=content,
                detail=f"{account_name} · reviewed {str(signal.get('kind') or 'need').replace('_', ' ')}",
                href=href,
                source_href=source_href,
                created_at=signal.get("created_at"),
                account_id=str(signal["account_id"]),
                account_name=account_name,
                why_now="A need was recorded in the last 14 days; its source has no recorded follow-up action.",
                next_step="Confirm the need, identify the decision-maker and timing, then agree one concrete follow-up.",
                prepared_prompt=(
                    f"Opportunity to qualify with {account_name}\n\nRecorded need: {content}\n"
                    f"Source: {source_href}\n\n"
                    "1. Confirm the need is still current and review the source evidence.\n"
                    "2. Ask who owns the decision, what success looks like, and when it is needed.\n"
                    "3. Agree a discovery conversation, demo, or sizing review and record its owner and date.\n\n"
                    "This is a recorded need to qualify; no deal value or likelihood has been inferred."
                ),
            )

        for note in data.get("waiting_notes", []):
            ready_to_review = note.get("status") == "review"
            add(
                key=f"customer_note:{note['id']}",
                kind="customer_note",
                title=str(note.get("title") or "Captured note"),
                detail=f"{note.get('account_name', '')} · "
                + (
                    "analysis ready for review"
                    if ready_to_review
                    else "captured, not yet analyzed"
                ),
                href=f"/customers?account={note['account_id']}&tab=sources&source={note['id']}",
                created_at=note.get("created_at"),
                account_id=str(note.get("account_id") or "") or None,
                account_name=str(note.get("account_name") or ""),
                next_step=(
                    "Open the captured note, check the extracted facts and actions, and save only the changes you approve."
                    if ready_to_review
                    else ""
                ),
                why_now=(
                    "The analysis is ready; reviewing it turns captured information into usable account context."
                    if ready_to_review
                    else "A captured customer conversation has not been analyzed yet."
                ),
                bump=6 if ready_to_review else 0,
            )

        for proposal in data.get("tool_proposals", []):
            add(
                key=f"tool_proposal:{proposal['id']}",
                kind="tool_proposal",
                title=str(proposal.get("summary") or "A tool is awaiting review"),
                detail=f"Risk {proposal.get('risk_level', '?')} · awaiting your decision",
                href="/tools",
                created_at=proposal.get("created_at"),
            )

        for atom in data.get("pending_answers", []):
            add(
                key=f"answer_atom:{atom['id']}",
                kind="answer_atom",
                title=str(atom.get("question") or "A reusable answer"),
                detail="Worked out in a run — keep it and it answers next time",
                href="/memory",
                created_at=atom.get("created_at"),
            )

        for memory in data.get("pending_memories", []):
            confidence = float(memory.get("confidence") or 0.0)
            add(
                key=f"memory:{memory['id']}",
                kind="memory",
                title=str(memory.get("content") or "Proposed memory"),
                detail=f"{memory.get('kind', 'fact')} · {round(confidence * 100)}% confidence",
                href="/memory",
                created_at=memory.get("created_at"),
                # A confident proposal is likelier to be worth keeping, so it
                # should reach the top of its own kind first.
                bump=confidence * 6,
            )

        # Assets are deliberately weighted low and capped. Dozens have needed
        # setup for months without anything breaking, so they belong in the
        # queue as a truthful backlog — never at the top of it, and never in
        # such volume that they bury a commitment that came due today.
        for asset in await self._assets_awaiting_trust():
            add(
                key=f"asset_trust:{asset.id}",
                kind="asset_trust",
                title=asset.name,
                detail="Launch recipe configured but not yet trusted to run",
                href=f"/assets/{quote(asset.id, safe='')}?view=settings",
            )

        for source in data.get("stale_sources", []):
            state = str(source.get("status") or "pending")
            add(
                key=f"stale_source:{source['id']}",
                kind="stale_source",
                title=str(source.get("label") or "Knowledge source"),
                detail=(
                    "Indexing failed — answers are missing this material"
                    if state == "error"
                    else "Added but never indexed, so nothing from it is searchable"
                ),
                href="/knowledge",
                created_at=None,
                bump=10 if state == "error" else 0,
            )

        # The database normally filters expiry; keep adapters and stale snapshots
        # from hiding work after its chosen return time.
        for item in items:
            if item.deferred_until is not None and item.deferred_until <= now:
                item.deferred_until = None
        live = [item for item in items if item.deferred_until is None]
        snoozed = sorted(
            (item for item in items if item.deferred_until is not None),
            key=lambda item: item.deferred_until or now,
        )
        live.sort(key=lambda item: (-item.priority, item.created_at or now))
        counts: dict[str, int] = {}
        for item in live:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return AttentionFeedV1(
            generated_at=now,
            items=live,
            top=live[:top],
            deferred_items=snoozed,
            counts=counts,
            total=len(live),
            deferred=len(snoozed),
            neglected=[item for item in live if _neglected(item, now)],
            opportunities=[
                item for item in live if item.kind == "customer_opportunity"
            ],
        )


class MorningBrief:
    """Composes the day's brief from records; optional prose is explicit.

    The split is the same one the document factory uses, and for the same
    reason. Anything a model writes here is commentary on numbers it was
    handed; it is never the source of them. A brief that says "you closed
    three actions" when you closed one would poison the one surface meant to
    be trusted at a glance.
    """

    def __init__(
        self,
        attention: AttentionService,
        database: Database,
        model: Any = None,
        preference: Any | None = None,
    ) -> None:
        self.attention = attention
        self.database = database
        self.model = model
        # The brief follows the same provider as everything else; without the
        # aliases the routed provider cannot resolve which model to ask.
        self.preference = preference
        self.last_error = ""

    async def compose(self, *, hours: int = 24, refresh: bool = False) -> Any:
        from .contracts import MorningBriefV1

        now = datetime.now(UTC)
        self.last_error = ""
        since = now - timedelta(hours=max(hours, 1))
        stamp = since.isoformat().replace("+00:00", "Z")
        feed, changes = await asyncio.gather(
            self.attention.feed(top=3), self.database.changes_since(stamp)
        )

        changed: list[str] = []
        for win in changes.get("wins", []):
            arr = win.get("yearly_arr")
            amount = f" (${arr:,.0f} ARR)" if arr else ""
            changed.append(
                f"Recorded a win: {win['title']} — {win['account_name']}{amount}"
            )
        for action in changes.get("closed_actions", []):
            changed.append(
                f"Closed: {action['description']} — {action['account_name']}"
            )
        if changes.get("runs_completed"):
            changed.append(f"{changes['runs_completed']} runs completed")
        if changes.get("memories_kept"):
            changed.append(f"{changes['memories_kept']} memories kept")
        if changes.get("actions_created"):
            changed.append(f"{changes['actions_created']} new actions captured")

        brief = MorningBriefV1(
            generated_at=now,
            since=since,
            changed=changed,
            focus=feed.top,
            waiting_total=feed.total,
            narrative=(
                f"{feed.counts.get('customer_action', 0)} open commitments, "
                f"{len(feed.neglected)} needing a follow-up, and "
                f"{len(feed.opportunities)} recent opportunity signals."
                if feed.total
                else ""
            ),
            recommendation=(
                f"Start with {feed.top[0].title}. {feed.top[0].why_now}"
                if feed.top
                else ""
            ),
        )
        # Opening Today, refreshing facts, and listening must never start a
        # reasoning job. The caller must explicitly request written prose.
        if self.model is None or not refresh:
            return brief
        facts = "\n".join(
            [
                f"Waiting on the user: {feed.total} items.",
                "Top of the queue:",
                *(
                    f"- [{item.kind_label}] {item.title}"
                    + (" (OVERDUE)" if item.overdue else "")
                    for item in feed.top
                ),
                f"Changed in the last {hours} hours:",
                *(f"- {line}" for line in changed or ["- nothing recorded"]),
            ]
        )
        aliases: dict[str, str] = {}
        if self.preference is not None:
            try:
                aliases = self.preference.resolve_aliases()
            except Exception:  # noqa: BLE001 - fall back to provider defaults
                aliases = {}
        try:
            written = await cast_any(self.model)._structured(
                BriefProseV1,
                system_prompt=(
                    "You write the two short paragraphs that open someone's "
                    "working day, over facts you are given. Never state a "
                    "number, name, or outcome that is not in those facts, and "
                    "never imply more happened than they show. `narrative` is "
                    "two sentences on where things stand. `recommendation` is "
                    "one sentence naming what to do first and why it matters "
                    "more than the rest. Plain, specific, no cheerleading. Return both "
                    "fields by calling the supplied function exactly once."
                ),
                user_prompt=facts,
                role="planner",
                model_aliases=aliases,
                max_output_tokens=600,
            )
            brief = brief.model_copy(
                update={
                    "narrative": written.narrative,
                    "recommendation": written.recommendation,
                }
            )
        except Exception as error:  # noqa: BLE001 - the facts are the brief; prose is a bonus
            # Surfaced, not swallowed: a brief that quietly loses its prose
            # every morning is indistinguishable from one that never had any.
            self.last_error = f"{type(error).__name__}: {str(error)[:200]}"
        return brief


class BriefProseV1(Contract):
    """The written half of the brief.

    Public name on purpose: providers derive a tool-function name from the
    class, and a leading underscore produced `return__briefprose`, which
    Command A+ answered in prose rather than calling."""

    narrative: str = Field(default="", max_length=600)
    recommendation: str = Field(default="", max_length=300)


def cast_any(value: Any) -> Any:
    """The provider's structured decode is a per-lane capability, not part of
    the shared protocol."""
    return value
