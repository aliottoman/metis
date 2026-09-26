"""Customer-scoped capture, extraction, review, and Markdown output."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from .contracts import (
    CustomerAccountDetailV1,
    CustomerAccountV1,
    CustomerActionExtractV1,
    CustomerActionV1,
    CustomerDashboardV1,
    CustomerEvidenceV1,
    CustomerRecordEvidenceV1,
    CustomerExtractionV1,
    CustomerFactV1,
    CustomerInteractionV1,
    CustomerNoteV1,
    CustomerOutputV1,
    CustomerPersonV1,
    CustomerSearchHitV1,
    CustomerSearchResultV1,
    CustomerSettingsV1,
    CustomerSourceV1,
    CustomerUpdateProposalV1,
    CustomerWinV1,
    KnowledgeSnippetV1,
)
from .context_links import customer_record_url, meeting_source_url, source_record_url
from .database import Database
from .local_model_session import LocalModelSessionManager
from .model_preference import is_cloud_model
from .model_provider import DeterministicModelProvider
from .queue_update import candidate_accounts


EXTRACTION_PROMPT_VERSION = "customer-extraction-v1"
DEFAULT_ACTIVITY_TEMPLATE = """## {account_name} — Customer Activity

**Date:** {date}
**Interaction:** {title}

### Summary
{summary}

### Decisions, requirements, and signals
{facts}

### Actions
{actions}

### People
{people}

### Source
{source}
"""


def _source(row: dict[str, Any]) -> CustomerSourceV1:
    return CustomerSourceV1(
        **{key: value for key, value in row.items() if key != "content_hash"}
    )


def _evidence(value: Any, *, record: bool = False) -> CustomerEvidenceV1:
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except ValueError:
            value = {}
    contract = CustomerRecordEvidenceV1 if record else CustomerEvidenceV1
    return contract.model_validate(value or {})


def _action(row: dict[str, Any]) -> CustomerActionV1:
    value = dict(row)
    value["evidence"] = _evidence(value.pop("evidence_json", {}), record=True)
    return CustomerActionV1.model_validate(value)


def _snippet(text: str, needle: str, *, width: int = 180) -> str:
    """A one-line excerpt centred on the match, so a hit shows why it matched."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    found = flat.lower().find(needle) if needle else -1
    if found < 0:
        return f"{flat[:width].rstrip()}…"
    start = max(0, found - width // 3)
    end = min(len(flat), start + width)
    return (
        ("…" if start else "")
        + flat[start:end].strip()
        + ("…" if end < len(flat) else "")
    )


def _proposal(row: dict[str, Any]) -> CustomerUpdateProposalV1:
    value = dict(row)
    if "extraction" not in value:
        import json

        value["extraction"] = json.loads(value.pop("extraction_json"))
    return CustomerUpdateProposalV1.model_validate(value)


class CustomerIntelligenceService:
    def __init__(
        self,
        database: Database,
        model: Any,
        model_session: LocalModelSessionManager,
        preference: Any | None = None,
    ) -> None:
        self.database = database
        # The routed provider, not the local one: note analysis follows the
        # same model choice the chat header shows, cloud or local.
        self.model = model
        self.model_session = model_session
        self.preference = preference
        # One auto-analysis at a time: extraction on capture is a convenience,
        # and a rate-limited free tier must not be hit by a burst of them.
        self._auto_analyze_lock = asyncio.Semaphore(1)

    async def accounts(self) -> list[CustomerAccountV1]:
        return [
            CustomerAccountV1.model_validate(item)
            for item in await self.database.list_customer_accounts()
        ]

    async def account(self, account_id: str) -> CustomerAccountDetailV1 | None:
        data = await self.database.customer_account_data(account_id)
        if data is None:
            return None
        interactions = [
            CustomerInteractionV1.model_validate(item) for item in data["interactions"]
        ]
        facts: list[CustomerFactV1] = []
        for item in data["facts"]:
            value = dict(item)
            value["evidence"] = _evidence(value.pop("evidence_json", {}))
            facts.append(CustomerFactV1.model_validate(value))
        people: list[CustomerPersonV1] = []
        for item in data["people"]:
            value = dict(item)
            value["evidence"] = _evidence(value.pop("evidence_json", {}))
            # The row id is kept: editing or removing a contact addresses the
            # record, not the name it currently happens to carry.
            for key in ("account_id", "created_at", "updated_at"):
                value.pop(key, None)
            people.append(CustomerPersonV1.model_validate(value))
        wins = await self._wins(data["wins"], account_name=data["account"]["name"])
        return CustomerAccountDetailV1(
            account=CustomerAccountV1.model_validate(data["account"]),
            interactions=interactions,
            facts=facts,
            actions=[_action(item) for item in data["actions"]],
            people=people,
            sources=[_source(item) for item in data["sources"]],
            wins=wins,
            notes=[CustomerNoteV1.model_validate(item) for item in data["notes"]],
        )

    async def search(self, query: str, *, limit: int = 40) -> CustomerSearchResultV1:
        """Find any customer record mentioning `query`, across every account."""
        rows, truncated = await self.database.search_customer_records(
            query, limit=limit
        )
        needle = query.strip().lower()
        hits = [
            CustomerSearchHitV1(
                kind=row["kind"],
                id=row["id"],
                account_id=row["account_id"],
                account_name=row["account_name"],
                title=str(row["title"] or "").strip() or "Untitled",
                snippet=_snippet(str(row["snippet"] or ""), needle),
                occurred_at=row["at"],
            )
            for row in rows
        ]
        return CustomerSearchResultV1(query=query, hits=hits, truncated=truncated)

    async def dashboard(self) -> CustomerDashboardV1:
        data = await self.database.customer_dashboard_data()
        return CustomerDashboardV1(
            active_accounts=data["active_accounts"],
            open_actions=data["open_actions"],
            overdue_actions=data["overdue_actions"],
            waiting_notes=data["waiting_notes"],
            total_wins=data["total_wins"],
            dac_wins=data["dac_wins"],
            total_yearly_arr=data["total_yearly_arr"],
            wins_by_service=data["wins_by_service"],
            recent_accounts=[
                CustomerAccountV1.model_validate(item)
                for item in data["recent_accounts"]
            ],
            priority_actions=[_action(item) for item in data["priority_actions"]],
            recent_wins=await self._wins(data["recent_wins"]),
        )

    async def _wins(
        self, rows: list[dict[str, Any]], *, account_name: str | None = None
    ) -> list[CustomerWinV1]:
        """Wins with their estimate attached, in one query rather than per win."""
        from .win_valuation import WinValuationService

        valuations = await self.database.win_valuations_for(
            [str(row["id"]) for row in rows]
        )
        wins: list[CustomerWinV1] = []
        for row in rows:
            value = dict(row)
            if account_name is not None:
                value["account_name"] = account_name
            found = valuations.get(str(row["id"]))
            value["valuation"] = (
                WinValuationService.to_contract(found) if found else None
            )
            wins.append(CustomerWinV1.model_validate(value))
        return wins

    async def analyze(self, source_id: str) -> CustomerUpdateProposalV1:
        source = await self.database.get_customer_source(source_id)
        if source is None:
            raise KeyError("customer source not found")
        aliases: dict[str, str] = {}
        if self.preference is not None:
            try:
                aliases = self.preference.resolve_aliases()
            except Exception:  # noqa: BLE001 - fall back to the local session
                aliases = {}
        provider = aliases.get("_provider", "local")
        if provider == "local":
            # The local lane keeps its launch-gate: extraction on a model that
            # is not resident would stall the request for a full load.
            model = self.model_session.selected_model or ""
            await self.model_session.require_ready(model)
            aliases = {
                **aliases,
                "planner": model,
                "coder": model,
                "quality": model,
                "_provider": "local",
            }
        else:
            # Cloud analysis needs no local weights resident at all.
            model = f"{provider}:{aliases.get('planner', '')}"
        if isinstance(self.model, DeterministicModelProvider):
            extraction = self._deterministic_extraction(source)
            model = "deterministic"
        elif hasattr(self.model, "_structured"):
            lines = "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(source["content"].splitlines(), start=1)
            )
            extraction = await self.model._structured(
                CustomerExtractionV1,
                system_prompt=(
                    "You extract customer intelligence from one account-scoped note. "
                    "Do not invent facts. Preserve contradictions as separate facts. "
                    "Each fact, action, and person must carry a short verbatim quote "
                    "and matching line numbers from the supplied note. Return only "
                    "the requested structured object."
                ),
                user_prompt=(
                    f"Account-scoped source title: {source['title']}\n"
                    f"Source id: {source['id']}\n\nNumbered note:\n{lines}"
                ),
                role="planner",
                model_aliases=aliases,
                max_output_tokens=4096,
            )
        else:
            raise RuntimeError("the selected provider cannot extract customer notes")
        enriched = extraction.model_copy(deep=True)
        for collection in (enriched.people, enriched.facts, enriched.actions):
            for item in collection:
                item.evidence.source_id = source_id
        row = await self.database.create_customer_proposal(
            source_id=source_id,
            account_id=source["account_id"],
            extraction=enriched.model_dump(mode="json"),
            model=model,
            prompt_version=EXTRACTION_PROMPT_VERSION,
        )
        return _proposal(row)

    def _cloud_pinned(self) -> bool:
        """Whether the routed model runs in the cloud — the only case auto-
        analysis should fire, since a local pin would force a weight load on
        every capture (exactly what the manual Analyze gate avoids)."""
        if self.preference is None:
            return False
        try:
            aliases = self.preference.resolve_aliases()
        except Exception:  # noqa: BLE001 - no readable preference means no auto-fire
            return False
        if aliases.get("_provider", "local") != "local":
            return True
        return is_cloud_model(aliases.get("planner", ""))

    async def auto_analyze(self, source_id: str) -> None:
        """Best-effort background extraction right after capture: a note you
        file is already extracted and waiting for your review when you open the
        record. It never blocks the capture, never writes facts on its own
        (analyze leaves a review proposal), and is safe to call from anywhere.

        Fires only on a pinned cloud model, serialised so a burst of captures
        cannot trip a rate-limited free tier, idempotent (skips a source already
        moved past 'waiting'), and swallows errors so a failure simply leaves
        the note 'waiting' for the manual Analyze button."""
        if not self._cloud_pinned():
            return
        async with self._auto_analyze_lock:
            source = await self.database.get_customer_source(source_id)
            if source is None or source.get("status") != "waiting":
                return
            try:
                await self.analyze(source_id)
            except Exception:  # noqa: BLE001 - a convenience, never fatal to the app
                pass

    async def ingest_notion_documents(self, documents: list[Any]) -> int:
        """Map synced Notion pages onto customer accounts and file each as a
        source, so a customer's Notion page flows into their record.

        Deduped by content: an unchanged page on the next ~12h sync is a no-op
        (the capture's content hash makes it a duplicate), so the same info is
        never re-ingested. A genuinely-new or changed page is auto-analysed into
        a review proposal — Notion never writes to a record without your
        approval. Only an unambiguous single-account name match becomes a
        proposal; a page that names no account, or several, stays knowledge-only
        exactly as before. Returns how many pages produced a new proposal."""
        if not documents or not self._cloud_pinned():
            return 0
        accounts = await self.database.list_customer_accounts()
        filed = 0
        for document in documents:
            title = (getattr(document, "title", "") or "").strip()
            content = (getattr(document, "markdown", "") or "").strip()
            if not title or not content:
                continue
            matched = candidate_accounts(title, accounts)
            if len(matched) != 1:
                continue
            row, duplicate = await self.database.capture_customer_source(
                account_id=str(matched[0]["id"]),
                source_kind="notion",
                title=title,
                content=content,
                source_ref=str(
                    getattr(document, "url", "") or getattr(document, "page_id", "")
                ),
                occurred_at=None,
            )
            if duplicate or row.get("status") != "waiting":
                continue  # unchanged page → nothing new to review
            await self.auto_analyze(str(row["id"]))
            filed += 1
        return filed

    def _deterministic_extraction(self, source: dict[str, Any]) -> CustomerExtractionV1:
        lines = [
            line.strip() for line in source["content"].splitlines() if line.strip()
        ]
        actions: list[CustomerActionExtractV1] = []
        for index, line in enumerate(lines, start=1):
            lowered = line.lower().lstrip("-* ")
            if lowered.startswith(("action:", "todo:", "follow up:", "follow-up:")):
                actions.append(
                    CustomerActionExtractV1(
                        description=line.split(":", 1)[-1].strip(),
                        evidence=CustomerEvidenceV1(
                            quote=line[:1000],
                            source_id=source["id"],
                            line_start=index,
                            line_end=index,
                        ),
                    )
                )
        return CustomerExtractionV1(
            summary=" ".join(lines)[:4000],
            occurred_at=source.get("occurred_at"),
            actions=actions,
        )

    async def save_proposal(
        self, proposal_id: str, extraction: CustomerExtractionV1
    ) -> CustomerUpdateProposalV1 | None:
        row = await self.database.save_customer_proposal(
            proposal_id, extraction.model_dump(mode="json")
        )
        return _proposal(row) if row else None

    async def open_proposal(self, source_id: str) -> CustomerUpdateProposalV1 | None:
        """The pending review proposal for a source, if any — how the UI opens
        an auto-analyzed note (or a Notion-derived one) for review."""
        row = await self.database.get_open_proposal_for_source(source_id)
        return _proposal(row) if row else None

    async def apply_proposal(self, proposal_id: str) -> CustomerUpdateProposalV1 | None:
        """Apply a pending proposal's own extraction to the record, unchanged —
        the one-click path from the chat's analysis card. It is exactly a review
        that accepted every proposed item: nothing is written until this is
        called, and a proposal already decided (or missing) returns None so the
        caller can say so rather than writing twice."""
        row = await self.database.get_customer_proposal(proposal_id)
        if row is None or row.get("status") != "review":
            return None
        return await self.save_proposal(proposal_id, _proposal(row).extraction)

    async def context(self, account_id: str) -> str:
        detail = await self.account(account_id)
        if detail is None:
            raise KeyError("customer account not found")
        facts = (
            "\n".join(
                f"- [{item.kind}] {item.content}"
                for item in detail.facts
                if item.status in {"active", "disputed"}
            )
            or "- No saved facts"
        )
        actions = (
            "\n".join(
                f"- {item.description} (owner: {item.owner or 'unassigned'})"
                for item in detail.actions
                if item.status == "open"
            )
            or "- No open actions"
        )
        # Pinned notes only. A note is pinned precisely because the user decided
        # it is standing context for the account, so the pin is the consent to
        # spend conversation context on it.
        pinned = "\n".join(
            f"- {item.title or 'Note'}: {' '.join(item.body.split())[:600]}"
            for item in detail.notes
            if item.pinned
        )
        return (
            f"Selected customer: {detail.account.name} ({detail.account.id})\n"
            f"Saved customer facts:\n{facts}\nOpen actions:\n{actions}"
            + (f"\nPinned account notes:\n{pinned}" if pinned else "")
        )[:16_000]

    async def evidence(
        self, account_id: str, *, compact: bool = False
    ) -> list[KnowledgeSnippetV1]:
        """The account's reviewed record as individually citable sources.

        `context()` returns the same material as one prose block, which is what
        let an answer about this account state a figure no record contained:
        with nothing numbered, nothing could be cited, and the grounding gate —
        which measures citations against retrieved evidence — had no evidence to
        measure against and never ran.

        Every returned snippet is one record the user themselves reviewed, so
        each carries the record's own identifier and a score of 1.0. They are
        not fuzzy retrieval hits; they are the account's ledger.

        `compact` caps each kind for an *unscoped* chat, which merely mentioned
        the account and still needs its memories, corpus and history alongside.
        The cap is per kind rather than an overall head, so a ledger with forty
        facts still shows its open actions — the whole point is that an account
        with a record can never be answered as though it had none.
        """
        detail = await self.account(account_id)
        if detail is None:
            raise KeyError("customer account not found")
        name = detail.account.name
        snippets: list[KnowledgeSnippetV1] = []

        def cap(items: list[Any], limit: int) -> list[Any]:
            return items[:limit] if compact else items

        def add(
            label: str,
            symbol: str,
            text: str,
            record_id: str,
            *,
            tab: str,
            evidence: CustomerEvidenceV1 | None = None,
            source_id: str | None = None,
            origin_url: str | None = None,
        ) -> None:
            provenance_url = origin_url or customer_record_url(
                account_id, tab, f"{tab.rstrip('s')}-{record_id}"
            )
            reviewed_source = source_id or (evidence.source_id if evidence else None)
            if reviewed_source:
                provenance_url = source_record_url(account_id, reviewed_source)
            if (
                isinstance(evidence, CustomerRecordEvidenceV1)
                and evidence.source == "meeting"
                and evidence.meeting_id
            ):
                provenance_url = meeting_source_url(
                    evidence.meeting_id, evidence.turn_id
                )
                text += "\nKept from a meeting action reviewed by the user."
            if evidence and evidence.quote:
                # This is the exact quote the user reviewed, never the rest of
                # a raw note or transcript that has not been approved.
                text += f"\nReviewed source excerpt: {evidence.quote[:600]}"
            snippets.append(
                KnowledgeSnippetV1(
                    source_label=label,
                    provider="customer",
                    rel_path=f"{name}#{record_id}",
                    symbol=symbol[:120],
                    text=text.strip()[:2_000],
                    score=1.0,
                    source_url=provenance_url,
                )
            )

        for win in cap(detail.wins, 4):
            parts = [f"Win: {win.title}"]
            if win.yearly_arr is not None:
                parts.append(f"Yearly ARR: ${win.yearly_arr:,.0f}")
            if win.dac_shape:
                parts.append(f"Shape: {win.dac_shape}")
            if win.services:
                parts.append(f"Services: {', '.join(win.services)}")
            if win.won_at:
                parts.append(f"Won: {win.won_at.date().isoformat()}")
            if win.brief:
                parts.append(win.brief)
            add("Recorded win", win.title, ". ".join(parts), win.id, tab="wins")

        for fact in cap(
            [item for item in detail.facts if item.status in {"active", "disputed"}], 10
        ):
            label = "Disputed fact" if fact.status == "disputed" else "Reviewed fact"
            add(
                label,
                f"{fact.kind}",
                f"[{fact.kind}] {fact.content}",
                fact.id,
                tab="facts",
                evidence=fact.evidence,
            )

        for action in cap(
            [item for item in detail.actions if item.status == "open"], 8
        ):
            owner = action.owner or "unassigned"
            due = f", due {action.due_at.date().isoformat()}" if action.due_at else ""
            add(
                "Open action",
                action.description[:60],
                f"Open action: {action.description} (owner: {owner}{due})",
                action.id,
                tab="actions",
                evidence=action.evidence,
            )

        for person in cap(detail.people, 4):
            descriptor = ", ".join(
                part for part in (person.role, person.organization) if part
            )
            add(
                "Known contact",
                person.name,
                f"{person.name}{f' — {descriptor}' if descriptor else ''}",
                person.id,
                tab="people",
                evidence=person.evidence,
            )

        for note in cap([item for item in detail.notes if item.pinned], 3):
            add(
                "Pinned note",
                note.title or "Note",
                f"{note.title or 'Note'}: {' '.join(note.body.split())}",
                note.id,
                tab="notes",
                origin_url=(
                    f"/?conversation={quote(note.origin_ref, safe='')}"
                    if note.origin == "chat" and note.origin_ref
                    else None
                ),
            )

        # Newest first, bounded: an account with a long history must not crowd
        # the prompt with old interactions at the expense of its own ledger.
        for interaction in detail.interactions[: 3 if compact else 8]:
            when = (
                interaction.occurred_at.date().isoformat()
                if interaction.occurred_at
                else "undated"
            )
            summary = interaction.summary or interaction.title
            if not summary:
                continue
            add(
                "Logged interaction",
                f"{interaction.title or 'Interaction'} ({when})",
                f"{when} — {summary}",
                interaction.id,
                tab="timeline",
                source_id=interaction.source_id,
            )
        return snippets

    async def output(
        self, account_id: str, kind: str, interaction_id: str | None
    ) -> CustomerOutputV1:
        detail = await self.account(account_id)
        if detail is None:
            raise KeyError("customer account not found")
        selected = (
            next(
                (item for item in detail.interactions if item.id == interaction_id),
                None,
            )
            if interaction_id
            else (detail.interactions[0] if detail.interactions else None)
        )
        if kind != "activity_tracker":
            raise ValueError(
                "Only the activity tracker output is available in this version."
            )
        settings = CustomerSettingsV1.model_validate(
            await self.database.customer_settings()
        )
        facts = [
            item
            for item in detail.facts
            if selected is None or item.interaction_id == selected.id
        ]
        actions = [
            item
            for item in detail.actions
            if selected is None or item.interaction_id == selected.id
        ]
        source = next(
            (
                item
                for item in detail.sources
                if selected and item.id == selected.source_id
            ),
            None,
        )
        values = {
            "account_name": detail.account.name,
            "date": (
                selected.occurred_at.astimezone(UTC).date().isoformat()
                if selected
                else datetime.now(UTC).date().isoformat()
            ),
            "title": selected.title if selected else "Account update",
            "summary": selected.summary if selected else "No saved interaction yet.",
            "facts": "\n".join(f"- **{item.kind}:** {item.content}" for item in facts)
            or "- None captured",
            "actions": "\n".join(
                f"- [ ] {item.description}"
                + (f" — {item.owner}" if item.owner else "")
                + (f" — due {item.due_at.date().isoformat()}" if item.due_at else "")
                for item in actions
                if item.status == "open"
            )
            or "- [ ] No open actions",
            "people": "\n".join(
                f"- {item.name}" + (f" — {item.role}" if item.role else "")
                for item in detail.people
            )
            or "- None captured",
            "source": (
                f"{source.title}"
                + (f" — {source.source_ref}" if source.source_ref else "")
                if source
                else "Saved customer record"
            ),
        }
        template = settings.activity_template or DEFAULT_ACTIVITY_TEMPLATE
        try:
            content = template.format(**values)
        except (KeyError, ValueError):
            content = DEFAULT_ACTIVITY_TEMPLATE.format(**values)
        row = await self.database.create_customer_output(
            account_id, selected.id if selected else None, kind, content
        )
        return CustomerOutputV1(**row, tracker_url=settings.tracker_url)
