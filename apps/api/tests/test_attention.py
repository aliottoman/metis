"""The attention queue: ranked by consequence, not by count.

The property under test is the one the headline depends on. "3 things need
you" is only worth reading if those three are genuinely the top three, so a
promise that came due today must outrank a backlog that has waited months
without anything breaking.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from waqil_api.attention import AttentionService


def _iso(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


class _FakeDatabase:
    def __init__(self, **payload) -> None:
        self.payload = payload
        self.deferred: dict[str, str] = {}

    async def attention_data(self) -> dict:
        return {
            "pending_memories": [],
            "waiting_notes": [],
            "open_actions": [],
            "waiting_runs": [],
            "tool_proposals": [],
            "stale_sources": [],
            "deferrals": self.deferred,
            "now": _iso(datetime.now(UTC)),
            **self.payload,
        }

    async def defer_attention_item(
        self, item_key: str, kind: str, deferred_until: str, reason: str = ""
    ) -> None:
        self.deferred[item_key] = deferred_until

    async def clear_attention_deferral(self, item_key: str) -> None:
        self.deferred.pop(item_key, None)


async def test_a_due_commitment_outranks_a_long_stale_backlog() -> None:
    now = datetime.now(UTC)
    database = _FakeDatabase(
        open_actions=[
            {
                "id": "a1",
                "description": "Send the tenancy walkthrough deck",
                "owner": "Ali",
                "due_at": _iso(now - timedelta(days=1)),
                "created_at": _iso(now - timedelta(days=2)),
                "account_id": "cust_1",
                "account_name": "BAPCO",
            }
        ],
        pending_memories=[
            {
                "id": f"m{index}",
                "kind": "project",
                "content": f"An old proposal {index}",
                "confidence": 0.95,
                "created_at": _iso(now - timedelta(days=180)),
                "source_run_id": None,
            }
            for index in range(16)
        ],
    )
    feed = await AttentionService(database).feed(top=3)
    assert feed.total == 17
    # Seventeen items, and the one with a person waiting on it comes first.
    assert feed.top[0].kind == "customer_action"
    assert feed.top[0].overdue is True
    assert feed.counts["memory"] == 16


async def test_overdue_beats_the_same_item_not_yet_due() -> None:
    now = datetime.now(UTC)
    database = _FakeDatabase(
        open_actions=[
            {
                "id": "later",
                "description": "Not yet due",
                "owner": "",
                "due_at": _iso(now + timedelta(days=10)),
                "created_at": _iso(now - timedelta(days=1)),
                "account_id": "c",
                "account_name": "Acme",
            },
            {
                "id": "overdue",
                "description": "Overdue",
                "owner": "",
                "due_at": _iso(now - timedelta(days=3)),
                "created_at": _iso(now - timedelta(days=1)),
                "account_id": "c",
                "account_name": "Acme",
            },
        ]
    )
    feed = await AttentionService(database).feed()
    assert [item.title for item in feed.items] == ["Overdue", "Not yet due"]


async def test_age_cannot_let_housekeeping_outrank_a_commitment() -> None:
    """Age breaks ties inside a kind; it must never invert the kinds. A memory
    proposal from a year ago is still not more urgent than a customer promise."""
    now = datetime.now(UTC)
    database = _FakeDatabase(
        pending_memories=[
            {
                "id": "ancient",
                "kind": "project",
                "content": "Very old",
                "confidence": 1.0,
                "created_at": _iso(now - timedelta(days=400)),
                "source_run_id": None,
            }
        ],
        open_actions=[
            {
                "id": "fresh",
                "description": "Brand new commitment",
                "owner": "",
                "due_at": None,
                "created_at": _iso(now),
                "account_id": "c",
                "account_name": "Acme",
            }
        ],
    )
    feed = await AttentionService(database).feed()
    assert feed.items[0].kind == "customer_action"


async def test_deferral_removes_from_live_and_can_be_restored() -> None:
    now = datetime.now(UTC)
    database = _FakeDatabase(
        open_actions=[
            {
                "id": "a1",
                "description": "Snooze me",
                "owner": "",
                "due_at": None,
                "created_at": _iso(now),
                "account_id": "c",
                "account_name": "Acme",
            }
        ]
    )
    service = AttentionService(database)
    assert (await service.feed()).total == 1

    await service.defer("customer_action:a1", "customer_action", 7)
    feed = await service.feed()
    assert feed.total == 0
    assert feed.deferred == 1
    # Snoozed, not dismissed: it is still returned, with its return date.
    assert feed.deferred_items[0].deferred_until is not None

    await service.undefer("customer_action:a1")
    assert (await service.feed()).total == 1


async def test_an_empty_queue_is_a_valid_answer() -> None:
    feed = await AttentionService(_FakeDatabase()).feed()
    assert feed.total == 0
    assert feed.top == []
    assert feed.counts == {}


async def test_a_failing_asset_library_narrows_the_queue_but_never_breaks_it() -> None:
    class _Broken:
        async def list(self):
            raise RuntimeError("asset library is cold")

    feed = await AttentionService(_FakeDatabase(), assets=_Broken()).feed()
    assert feed.total == 0
