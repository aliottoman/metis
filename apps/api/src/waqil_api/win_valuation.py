"""Estimate what a win is worth when nobody wrote down a number.

Several real wins carry no ARR figure — the deal was recorded before the
commercials were known. This turns the account's own notes into a defensible
estimate rather than leaving the tracker blank.

The split of labour is the same one the Sizing tab's recommender uses, and for
the same reason. The model reads the notes and answers a bounded question:
*which of these SKUs, and how many of each?* It picks from a menu it is given
and quotes the note for every line. It never sees a price and never returns a
dollar figure. The host then multiplies the quantities by the rate card and
sums them, so every number in the result is arithmetic over rates the user
owns. A hallucination can put the wrong quantity on a line — which is visible
and editable — but it cannot invent a price, a SKU, or a total.

The result is always a *proposal*. It is stored apart from `yearly_arr` and
only becomes the win's figure when the user accepts it.
"""
from __future__ import annotations

import json
from typing import Any

from .contracts import (
    ModelRequestV1,
    WinValuationLineV1,
    WinValuationV1,
)
from .database import Database
from .sku_catalog import SkuCatalog

VALUATION_PROMPT_VERSION = "win-valuation-v1"

# How a billing metric becomes a yearly figure. Oracle states each SKU's unit in
# the service descriptions; annualizing it is arithmetic, not judgement, so it
# lives here rather than in the prompt.
HOURLY = "per hour"
MONTHLY = "per month"

SYSTEM_PROMPT = """You estimate the size of a closed Oracle Cloud deal from the \
notes taken about that customer.

You are given the evidence and a menu of billable SKUs. Identify what the \
customer actually deployed and return it as line items.

Rules:
- Use only the `sku` values from the menu. Never invent one.
- `quantity` is in the SKU's own unit: for "GPU Per Hour" it is the number of \
GPUs, for "OCPU Per Hour" the number of OCPUs, for a per-month storage SKU the \
number of gigabytes.
- `utilization` is the fraction of the year the resource runs: 1.0 for an \
always-on production deployment, lower only if the notes say it is part-time.
- Quote the evidence for each line in `why`, in a few words.
- Include a line only if the notes support it. An empty list is the right \
answer when the notes describe no infrastructure.
- You have no price list and must not state, guess, or imply any monetary \
amount anywhere in your reply.
- `confidence` is high only when the notes name the hardware and its size \
explicitly; low when you are inferring the deployment from indirect signals.

Reply with JSON only:
{"lines": [{"sku": "...", "quantity": 0, "utilization": 1.0, "why": "..."}],
 "explanation": "two sentences on what was deployed and how you read it",
 "confidence": "low|medium|high"}"""


class WinValuationService:
    def __init__(
        self,
        database: Database,
        catalog: SkuCatalog,
        *,
        model: Any = None,
        preference: Any = None,
    ) -> None:
        self.database = database
        self.catalog = catalog
        self._model = model
        self._preference = preference

    async def estimate(self, win_id: str) -> WinValuationV1:
        win = await self.database.get_customer_win(win_id)
        if win is None:
            raise KeyError("customer win not found")
        evidence = await self._evidence(win)
        parsed, model_used = await self._ask(evidence)
        lines, unpriced = self._price(parsed.get("lines", []))
        total = sum(line.yearly_amount for line in lines)
        row = await self.database.upsert_win_valuation(
            win_id,
            estimated_yearly_arr=total if lines else None,
            currency=self.catalog.card.currency,
            lines=[line.model_dump(mode="json") for line in lines],
            explanation=str(parsed.get("explanation") or "").strip()[:1200],
            confidence=self._confidence(parsed.get("confidence"), lines),
            unpriced=unpriced,
            rates_verified=self._rates_verified(lines),
            model_used=model_used,
            prompt_version=VALUATION_PROMPT_VERSION,
        )
        return self.to_contract(row)

    async def accept(self, win_id: str, yearly_arr: float | None = None) -> WinValuationV1:
        """Promote the estimate — or a corrected figure — to the win's ARR."""
        row = await self.database.get_win_valuation(win_id)
        if row is None:
            raise KeyError("win valuation not found")
        amount = yearly_arr if yearly_arr is not None else row["estimated_yearly_arr"]
        if amount is None:
            raise ValueError("this estimate produced no figure to accept")
        await self.database.set_customer_win_arr(win_id, float(amount))
        return self.to_contract(
            await self.database.set_win_valuation_status(win_id, "accepted")
        )

    async def dismiss(self, win_id: str) -> WinValuationV1:
        row = await self.database.set_win_valuation_status(win_id, "dismissed")
        if row is None:
            raise KeyError("win valuation not found")
        return self.to_contract(row)

    async def get(self, win_id: str) -> WinValuationV1 | None:
        row = await self.database.get_win_valuation(win_id)
        return self.to_contract(row) if row else None

    # -- evidence --------------------------------------------------------

    async def _evidence(self, win: dict[str, Any]) -> str:
        """What the model reads: the win itself, then the account's own record.

        Capped well inside any context window, newest first, because a note
        describing the deployed shape is usually recent and the tail of a long
        account history is rarely what was sold.
        """
        data = await self.database.customer_account_data(win["account_id"])
        parts = [
            f"Account: {(data or {}).get('account', {}).get('name', 'Unknown')}",
            f"Win: {win['title']}",
        ]
        if win.get("brief"):
            parts.append(f"Brief: {win['brief']}")
        if win.get("services"):
            parts.append(f"Services tagged: {', '.join(win['services'])}")
        if win.get("dac_shape"):
            parts.append(f"DAC shape: {win['dac_shape']}")
        if data:
            facts = [
                f"- [{item['kind']}] {item['content']}"
                for item in data.get("facts", [])
                if item.get("status") in (None, "active", "disputed")
            ][:40]
            if facts:
                parts.append("Reviewed account facts:\n" + "\n".join(facts))
            notes = [
                f"- {item['title']}: {str(item.get('content') or '')[:1200]}"
                for item in data.get("sources", [])
            ][:10]
            if notes:
                parts.append("Recent notes:\n" + "\n".join(notes))
        return "\n\n".join(parts)[:24_000]

    # -- model -----------------------------------------------------------

    def _aliases(self) -> dict[str, str]:
        """Route this one call to Grok when OCI is reachable.

        Unlike the rest of the workbench, which deliberately runs on whichever
        local model the user pinned, reading a deployment out of loose meeting
        notes and mapping it onto Oracle's catalog wants the larger cloud model
        — and it is one short call per win, not a per-message cost. If OCI is
        not configured this falls straight back to the user's own selection, so
        the feature still works entirely locally.
        """
        if self._preference is None:
            return {}
        try:
            aliases = self._preference.resolve_aliases()
            if getattr(self._preference, "oci_available", False):
                aliases = {**aliases, "_provider": "oci"}
            return aliases
        except Exception:  # noqa: BLE001 - fall back to provider defaults
            return {}

    async def _ask(self, evidence: str) -> tuple[dict[str, Any], str | None]:
        menu = self.catalog.describe_priced()
        if self._model is None or not menu:
            return {}, None
        aliases = self._aliases()
        # A literal alias match over the evidence, computed by the host. It is a
        # hint, not a constraint: it catches "4xH100" written out, and misses a
        # deployment described only in prose, which is what the model is for.
        payload = {
            "billable_skus": menu,
            "skus_named_in_the_evidence": [
                rate.key for rate in self.catalog.match(evidence)
            ],
            "evidence": evidence,
        }
        try:
            result = await self._model.generate(
                ModelRequestV1(
                    role="planner",
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                ),
                model_aliases=aliases,
            )
        except Exception:  # noqa: BLE001 - an outage yields no estimate, not an error
            return {}, None
        parsed = _parse_json_object(result.content or "")
        return (parsed or {}), (result.model if parsed else None)

    # -- pricing ---------------------------------------------------------

    def _price(
        self, raw_lines: Any
    ) -> tuple[list[WinValuationLineV1], list[str]]:
        """Turn the model's quantities into money. All arithmetic, no judgement."""
        lines: list[WinValuationLineV1] = []
        unpriced: list[str] = []
        if not isinstance(raw_lines, list):
            return lines, unpriced
        for item in raw_lines[:20]:
            if not isinstance(item, dict):
                continue
            sku = str(item.get("sku") or "").strip()
            rate = self.catalog.rate(sku)
            if rate is None:
                # The model named something outside the menu. Surfacing it beats
                # dropping it: it usually means a real component nobody has a
                # rate for yet, which is exactly what the user should see.
                if sku:
                    unpriced.append(sku)
                continue
            try:
                quantity = max(0.0, float(item.get("quantity") or 0.0))
                utilization = min(1.0, max(0.0, float(item.get("utilization", 1.0))))
            except (TypeError, ValueError):
                continue
            if quantity <= 0:
                continue
            amount, basis = self._annualize(rate.unit, rate.value, quantity, utilization)
            entry = self.catalog.entry(rate.part_number) if rate.part_number else None
            lines.append(
                WinValuationLineV1(
                    sku=rate.key,
                    part_number=rate.part_number,
                    name=entry.name if entry else rate.label,
                    unit=rate.unit,
                    quantity=quantity,
                    utilization=utilization,
                    rate=rate.value,
                    rate_verified=rate.verified,
                    yearly_amount=round(amount, 2),
                    basis=basis,
                    why=str(item.get("why") or "").strip()[:300],
                )
            )
        return lines, unpriced

    def _annualize(
        self, unit: str, rate: float, quantity: float, utilization: float
    ) -> tuple[float, str]:
        hours = self.catalog.card.hours_per_year
        lowered = unit.lower()
        if lowered.endswith(HOURLY):
            billable = hours * utilization
            return rate * quantity * billable, (
                f"{quantity:g} × ${rate:,.4f}/hr × {billable:,.0f} hr"
            )
        if lowered.endswith(MONTHLY):
            return rate * quantity * 12, f"{quantity:g} × ${rate:,.4f}/mo × 12"
        # An unrecognised metric is billed once per unit rather than annualized,
        # which understates rather than inflates it.
        return rate * quantity, f"{quantity:g} × ${rate:,.4f}"

    @staticmethod
    def _rates_verified(lines: list[WinValuationLineV1]) -> bool:
        return bool(lines) and all(line.rate_verified for line in lines)

    @staticmethod
    def _confidence(value: Any, lines: list[WinValuationLineV1]) -> str:
        if not lines:
            return "low"
        return value if value in ("low", "medium", "high") else "low"

    @staticmethod
    def to_contract(row: dict[str, Any]) -> WinValuationV1:
        value = dict(row)
        value["lines"] = [
            WinValuationLineV1.model_validate(item)
            for item in _loads(value.pop("lines_json", "[]"), [])
        ]
        value["unpriced"] = _loads(value.pop("unpriced_json", "[]"), [])
        value["rates_verified"] = bool(value.get("rates_verified"))
        return WinValuationV1.model_validate(value)


def _loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw or "")
    except (TypeError, ValueError):
        return default


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object out of a model reply that may be fenced."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
