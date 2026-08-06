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

from .contracts import AttentionFeedV1, AttentionItemV1
from .database import Database

# What a delay actually costs, expressed as a base weight. A commitment made to
# a customer outranks the app's own housekeeping — the queue is ordered by who
# is waiting, not by which subsystem produced the row.
_BASE_WEIGHT = {
    "run_approval": 90,      # a run is stopped mid-flight, holding its work
    "customer_action": 80,   # a promise to someone outside this machine
    "customer_note": 60,     # captured intelligence not yet in the record
    "tool_proposal": 45,     # a capability waiting to become real
    "memory": 35,            # improves future answers; nothing breaks meanwhile
    "asset_trust": 25,       # a reviewed recipe waiting to be trusted
    "stale_source": 20,      # knowledge is thinner than it could be
}

_KIND_LABEL = {
    "run_approval": "Approval",
    "customer_action": "Customer action",
    "customer_note": "Note to analyze",
    "tool_proposal": "Tool proposal",
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
            add(
                key=f"customer_action:{action['id']}",
                kind="customer_action",
                title=str(action.get("description") or "Open action"),
                detail=f"{action.get('account_name', '')}"
                + (f" · owner {owner}" if owner else ""),
                href=f"/customers?account={action['account_id']}&tab=actions",
                created_at=action.get("created_at"),
                due_at=action.get("due_at"),
                account_id=str(action.get("account_id") or "") or None,
            )

        for note in data.get("waiting_notes", []):
            add(
                key=f"customer_note:{note['id']}",
                kind="customer_note",
                title=str(note.get("title") or "Captured note"),
                detail=f"{note.get('account_name', '')} · captured, not yet analyzed",
                href=f"/customers?account={note['account_id']}&tab=sources",
                created_at=note.get("created_at"),
                account_id=str(note.get("account_id") or "") or None,
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
                href="/assets",
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
        )
