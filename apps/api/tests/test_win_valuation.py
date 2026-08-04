from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from waqil_api.contracts import ModelResultV1
from waqil_api.database import Database
from waqil_api.sku_catalog import SkuCatalog
from waqil_api.win_valuation import WinValuationService

CATALOG = {
    "entries": [
        {
            "part_number": "B98415",
            "name": "Oracle Cloud Infrastructure - Compute - GPU H100",
            "metric": "GPU Per Hour",
            "category": "Compute",
            "retired": False,
        },
        {
            "part_number": "B91961",
            "name": "Oracle Cloud Infrastructure Block Volume",
            "metric": "Gigabyte Storage Capacity Per Month",
            "category": "Storage",
            "retired": False,
        },
    ]
}
RATES = {
    "currency": "USD",
    "hours_per_year": 8760,
    "rates": [
        {
            "part_number": "B98415", "unit": "GPU Per Hour", "value": 10.0,
            "verified": False, "label": "OCI Compute — GPU H100",
            "aliases": ["H100", "BM.GPU.H100.8"],
        },
        {
            "part_number": "B91961", "unit": "Gigabyte Storage Capacity Per Month",
            "value": 0.0255, "verified": True, "label": "OCI Block Volume",
            "aliases": ["block volume"],
        },
        {
            "part_number": "B00000", "unit": "GPU Per Hour", "value": 99.0,
            "verified": True, "label": "Not in the catalog", "aliases": [],
        },
    ],
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def catalog(tmp_path: Path) -> SkuCatalog:
    (tmp_path / "catalog.json").write_text(json.dumps(CATALOG), encoding="utf-8")
    (tmp_path / "rates.json").write_text(json.dumps(RATES), encoding="utf-8")
    return SkuCatalog(tmp_path)


@pytest.fixture
async def database(tmp_path: Path) -> Any:
    value = Database(tmp_path / "test.db")
    await value.open()
    try:
        yield value
    finally:
        await value.close()


class ScriptedModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.payloads: list[dict[str, Any]] = []

    async def generate(self, request: Any, model_aliases: Any = None) -> ModelResultV1:
        self.payloads.append(json.loads(request.user_prompt))
        return ModelResultV1(model="xai.grok-4.3", content=self.reply)


def test_rate_without_a_catalog_entry_is_dropped(catalog: SkuCatalog) -> None:
    """The catalog is the authority on what Oracle sells, so a rate whose part
    number it does not list cannot become a billable line."""
    assert catalog.rate("B00000") is None
    assert {rate.key for rate in catalog.priced()} == {"B98415", "B91961"}


def test_alias_match_reads_how_notes_are_actually_written(catalog: SkuCatalog) -> None:
    assert [r.key for r in catalog.match("deployed 4xH100 last quarter")] == ["B98415"]
    assert [r.key for r in catalog.match("BM.GPU.H100.8 in Dubai")] == ["B98415"]


@pytest.mark.anyio
async def test_host_prices_the_model_quantities(catalog: SkuCatalog, database: Any) -> None:
    model = ScriptedModel(json.dumps({
        "lines": [
            {"sku": "B98415", "quantity": 4, "utilization": 1.0, "why": "4xH100 deployed"},
            {"sku": "B91961", "quantity": 2000, "why": "2TB block volume"},
            {"sku": "B_INVENTED", "quantity": 5, "why": "hallucinated"},
        ],
        "explanation": "Four H100s and two terabytes.",
        "confidence": "high",
    }))
    account = await database.create_customer_account(
        name="Acme", aliases=[], industry="", region=""
    )
    win = await database.create_customer_win(
        account["id"], title="H100 cluster", brief="They deployed a 4xH100 GPU cluster.",
        services=["DAC"], dac_shape="", yearly_arr=None, won_at=None, source_ref="",
    )
    service = WinValuationService(database, catalog, model=model)
    result = await service.estimate(win["id"])

    # 4 GPUs × $10/hr × 8760 hr = $350,400; 2000 GB × $0.0255 × 12 = $612.
    assert result.estimated_yearly_arr == pytest.approx(351_012.0)
    assert [line.sku for line in result.lines] == ["B98415", "B91961"]
    assert result.lines[0].basis == "4 × $10.0000/hr × 8,760 hr"
    # A SKU outside the menu is surfaced, not silently dropped or valued at zero.
    assert result.unpriced == ["B_INVENTED"]
    # One line rests on an unverified rate, so the whole estimate is unverified.
    assert result.rates_verified is False
    assert result.status == "proposed"
    assert result.model_used == "xai.grok-4.3"

    # The model is never shown a price, and is given the host's literal matches.
    payload = model.payloads[0]
    assert "skus_named_in_the_evidence" in payload
    assert payload["skus_named_in_the_evidence"] == ["B98415"]
    assert "value" not in json.dumps(payload["billable_skus"])
    assert "10.0" not in json.dumps(payload["billable_skus"])


@pytest.mark.anyio
async def test_estimate_never_becomes_the_win_arr_until_accepted(
    catalog: SkuCatalog, database: Any
) -> None:
    model = ScriptedModel(json.dumps({
        "lines": [{"sku": "B98415", "quantity": 1, "why": "one H100"}],
        "explanation": "One GPU.", "confidence": "medium",
    }))
    account = await database.create_customer_account(
        name="Harbor", aliases=[], industry="", region=""
    )
    win = await database.create_customer_win(
        account["id"], title="Pilot", brief="single H100", services=[],
        dac_shape="", yearly_arr=None, won_at=None, source_ref="",
    )
    service = WinValuationService(database, catalog, model=model)
    await service.estimate(win["id"])

    assert (await database.get_customer_win(win["id"]))["yearly_arr"] is None

    accepted = await service.accept(win["id"])
    assert accepted.status == "accepted"
    assert (await database.get_customer_win(win["id"]))["yearly_arr"] == pytest.approx(87_600.0)

    # Re-estimating resets an accepted proposal to unreviewed.
    assert (await service.estimate(win["id"])).status == "proposed"


@pytest.mark.anyio
async def test_a_corrected_figure_wins_over_the_estimate(
    catalog: SkuCatalog, database: Any
) -> None:
    model = ScriptedModel(json.dumps({
        "lines": [{"sku": "B98415", "quantity": 8, "why": "eight"}],
        "explanation": "Eight GPUs.", "confidence": "low",
    }))
    account = await database.create_customer_account(
        name="Simvia", aliases=[], industry="", region=""
    )
    win = await database.create_customer_win(
        account["id"], title="Cluster", brief="8xH100", services=[],
        dac_shape="", yearly_arr=None, won_at=None, source_ref="",
    )
    service = WinValuationService(database, catalog, model=model)
    await service.estimate(win["id"])
    await service.accept(win["id"], yearly_arr=250_000.0)
    assert (await database.get_customer_win(win["id"]))["yearly_arr"] == pytest.approx(250_000.0)


@pytest.mark.anyio
async def test_a_model_outage_yields_no_estimate_rather_than_an_error(
    catalog: SkuCatalog, database: Any
) -> None:
    class Broken:
        async def generate(self, request: Any, model_aliases: Any = None) -> Any:
            raise RuntimeError("provider is down")

    account = await database.create_customer_account(
        name="GlassHub", aliases=[], industry="", region=""
    )
    win = await database.create_customer_win(
        account["id"], title="Deal", brief="", services=[], dac_shape="",
        yearly_arr=None, won_at=None, source_ref="",
    )
    result = await WinValuationService(database, catalog, model=Broken()).estimate(
        win["id"]
    )
    assert result.estimated_yearly_arr is None
    assert result.lines == []
    assert result.model_used is None
    assert result.confidence == "low"


def test_rate_card_endpoints_round_trip(client: TestClient) -> None:
    card = client.get("/api/v1/sku-rates")
    assert card.status_code == 200
    body = card.json()
    assert body["catalog_size"] > 700
    assert body["currency"] == "USD"
    h100 = next(item for item in body["rates"] if item["part_number"] == "B98415")
    assert h100["unit"] == "GPU Per Hour"
    assert h100["verified"] is False

    updated = client.put(
        "/api/v1/sku-rates",
        json={"updates": [{"key": "B98415", "value": 12.5, "verified": True}]},
    )
    assert updated.status_code == 200
    changed = next(
        item for item in updated.json()["rates"] if item["part_number"] == "B98415"
    )
    assert changed["value"] == 12.5
    assert changed["verified"] is True
    # The edit landed in the user's own copy under data_dir, not in the seed
    # vendored with the package — so the suite cannot mutate the shipped rates.
    seed = json.loads(
        (Path(__file__).resolve().parents[1]
         / "src/waqil_api/data/skus/rates.json").read_text(encoding="utf-8")
    )
    assert next(
        item for item in seed["rates"] if item["part_number"] == "B98415"
    )["value"] == 10.0


def test_valuation_endpoint_reports_a_missing_win(client: TestClient) -> None:
    missing = client.post("/api/v1/customers/wins/cwin_0000000000000000000/valuation")
    assert missing.status_code == 404
