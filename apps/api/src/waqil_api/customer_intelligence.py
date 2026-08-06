"""Customer-scoped capture, extraction, review, and Markdown output."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .contracts import (
    CustomerAccountDetailV1,
    CustomerAccountV1,
    CustomerActionExtractV1,
    CustomerActionV1,
    CustomerDashboardV1,
    CustomerEvidenceV1,
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
)
from .database import Database
from .local_model_session import LocalModelSessionManager
from .model_provider import DeterministicModelProvider


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


def _evidence(value: Any) -> CustomerEvidenceV1:
    if isinstance(value, str):
        import json
        try:
            value = json.loads(value)
        except ValueError:
            value = {}
    return CustomerEvidenceV1.model_validate(value or {})


def _action(row: dict[str, Any]) -> CustomerActionV1:
    value = dict(row)
    value["evidence"] = _evidence(value.pop("evidence_json", {}))
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
    return ("…" if start else "") + flat[start:end].strip() + ("…" if end < len(flat) else "")


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

    def _deterministic_extraction(
        self, source: dict[str, Any]
    ) -> CustomerExtractionV1:
        lines = [line.strip() for line in source["content"].splitlines() if line.strip()]
        actions: list[CustomerActionExtractV1] = []
        for index, line in enumerate(lines, start=1):
            lowered = line.lower().lstrip("-* ")
            if lowered.startswith(("action:", "todo:", "follow up:", "follow-up:")):
                actions.append(
                    CustomerActionExtractV1(
                        description=line.split(":", 1)[-1].strip(),
                        evidence=CustomerEvidenceV1(
                            quote=line[:1000], source_id=source["id"],
                            line_start=index, line_end=index,
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

    async def context(self, account_id: str) -> str:
        detail = await self.account(account_id)
        if detail is None:
            raise KeyError("customer account not found")
        facts = "\n".join(
            f"- [{item.kind}] {item.content}" for item in detail.facts
            if item.status in {"active", "disputed"}
        ) or "- No saved facts"
        actions = "\n".join(
            f"- {item.description} (owner: {item.owner or 'unassigned'})"
            for item in detail.actions if item.status == "open"
        ) or "- No open actions"
        # Pinned notes only. A note is pinned precisely because the user decided
        # it is standing context for the account, so the pin is the consent to
        # spend conversation context on it.
        pinned = "\n".join(
            f"- {item.title or 'Note'}: {' '.join(item.body.split())[:600]}"
            for item in detail.notes if item.pinned
        )
        return (
            f"Selected customer: {detail.account.name} ({detail.account.id})\n"
            f"Saved customer facts:\n{facts}\nOpen actions:\n{actions}"
            + (f"\nPinned account notes:\n{pinned}" if pinned else "")
        )[:16_000]

    async def output(
        self, account_id: str, kind: str, interaction_id: str | None
    ) -> CustomerOutputV1:
        detail = await self.account(account_id)
        if detail is None:
            raise KeyError("customer account not found")
        selected = next(
            (item for item in detail.interactions if item.id == interaction_id), None
        ) if interaction_id else (detail.interactions[0] if detail.interactions else None)
        if kind != "activity_tracker":
            raise ValueError("Only the activity tracker output is available in this version.")
        settings = CustomerSettingsV1.model_validate(
            await self.database.customer_settings()
        )
        facts = [
            item for item in detail.facts
            if selected is None or item.interaction_id == selected.id
        ]
        actions = [
            item for item in detail.actions
            if selected is None or item.interaction_id == selected.id
        ]
        source = next(
            (item for item in detail.sources if selected and item.id == selected.source_id),
            None,
        )
        values = {
            "account_name": detail.account.name,
            "date": (
                selected.occurred_at.astimezone(UTC).date().isoformat()
                if selected else datetime.now(UTC).date().isoformat()
            ),
            "title": selected.title if selected else "Account update",
            "summary": selected.summary if selected else "No saved interaction yet.",
            "facts": "\n".join(f"- **{item.kind}:** {item.content}" for item in facts)
            or "- None captured",
            "actions": "\n".join(
                f"- [ ] {item.description}"
                + (f" — {item.owner}" if item.owner else "")
                + (f" — due {item.due_at.date().isoformat()}" if item.due_at else "")
                for item in actions if item.status == "open"
            ) or "- [ ] No open actions",
            "people": "\n".join(
                f"- {item.name}" + (f" — {item.role}" if item.role else "")
                for item in detail.people
            ) or "- None captured",
            "source": (
                f"{source.title}" + (f" — {source.source_ref}" if source.source_ref else "")
                if source else "Saved customer record"
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
        return CustomerOutputV1(
            **row, tracker_url=settings.tracker_url
        )
