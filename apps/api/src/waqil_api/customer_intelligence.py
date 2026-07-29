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
    CustomerOutputV1,
    CustomerPersonExtractV1,
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
        local_model: Any,
        model_session: LocalModelSessionManager,
    ) -> None:
        self.database = database
        self.local_model = local_model
        self.model_session = model_session

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
        people: list[CustomerPersonExtractV1] = []
        for item in data["people"]:
            value = dict(item)
            value["evidence"] = _evidence(value.pop("evidence_json", {}))
            for key in ("id", "account_id", "created_at", "updated_at"):
                value.pop(key, None)
            people.append(CustomerPersonExtractV1.model_validate(value))
        wins: list[CustomerWinV1] = []
        for item in data["wins"]:
            value = dict(item)
            value["account_name"] = data["account"]["name"]
            wins.append(CustomerWinV1.model_validate(value))
        return CustomerAccountDetailV1(
            account=CustomerAccountV1.model_validate(data["account"]),
            interactions=interactions,
            facts=facts,
            actions=[_action(item) for item in data["actions"]],
            people=people,
            sources=[_source(item) for item in data["sources"]],
            wins=wins,
        )

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
            recent_wins=[
                CustomerWinV1.model_validate(item) for item in data["recent_wins"]
            ],
        )

    async def analyze(self, source_id: str) -> CustomerUpdateProposalV1:
        source = await self.database.get_customer_source(source_id)
        if source is None:
            raise KeyError("customer source not found")
        model = self.model_session.selected_model or ""
        await self.model_session.require_ready(model)
        if isinstance(self.local_model, DeterministicModelProvider):
            extraction = self._deterministic_extraction(source)
            model = "deterministic"
        elif hasattr(self.local_model, "_structured"):
            lines = "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(source["content"].splitlines(), start=1)
            )
            extraction = await self.local_model._structured(
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
                model_aliases={
                    "planner": model, "coder": model, "quality": model,
                    "_provider": "local",
                },
                max_output_tokens=4096,
            )
        else:
            raise RuntimeError("the selected local provider cannot extract customer notes")
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
        return (
            f"Selected customer: {detail.account.name} ({detail.account.id})\n"
            f"Saved customer facts:\n{facts}\nOpen actions:\n{actions}"
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
