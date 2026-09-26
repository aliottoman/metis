"""Actionable briefings use recorded evidence, fresh state, and explicit work."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from waqil_api.attention import AttentionService, MorningBrief
from waqil_api.database import Database


NOW = datetime(2026, 9, 10, 9, tzinfo=UTC)


def stamp(days_ago: int = 0) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


class Records:
    def __init__(self, **data) -> None:
        self.data = data

    async def attention_data(self):
        return {"now": stamp(), **self.data}

    async def changes_since(self, _since):
        return {"runs_completed": 2}


def action(identifier: str, **fields):
    return {
        "id": identifier,
        "description": f"Follow up {identifier}",
        "account_id": "c1",
        "account_name": "Acme",
        "created_at": stamp(30),
        "updated_at": stamp(30),
        "due_at": None,
        **fields,
    }


async def test_neglect_uses_last_update_and_does_not_mislabel_future_deadlines():
    feed = await AttentionService(
        Records(
            open_actions=[
                action("neglected"),
                action("touched", updated_at=stamp(1)),
                action("planned", due_at=stamp(-10)),
                action("overdue", updated_at=stamp(), due_at=stamp(1)),
            ]
        )
    ).feed()
    assert {item.key for item in feed.neglected} == {
        "customer_action:neglected",
        "customer_action:overdue",
    }
    assert "30 days" in next(
        item.why_now for item in feed.neglected if item.key.endswith("neglected")
    )


async def test_priority_and_prepared_actions_remain_tied_to_the_real_source():
    records = Records(
        open_actions=[action("urgent", due_at=stamp(2), owner="Sam", source_id="s1")],
        waiting_runs=[
            {
                "id": "r1",
                "conversation_id": "chat1",
                "prompt": "Review a change",
                "created_at": stamp(),
            }
        ],
        opportunity_signals=[
            {
                "id": "f1",
                "account_id": "c2",
                "account_name": "Northwind",
                "kind": "requirement",
                "content": "Needs a sizing review",
                "created_at": stamp(1),
                "source_id": "s2",
            }
        ],
        pending_memories=[
            {"id": "m1", "content": "Old preference", "created_at": stamp(100)}
        ],
    )
    feed = await AttentionService(records).feed()
    assert [item.kind for item in feed.top] == [
        "customer_action",
        "run_approval",
        "customer_opportunity",
    ]
    assert len(feed.top) == 3
    assert feed.top[0].source_href == "/customers?account=c1&tab=sources&source=s1"
    assert "Sam" in feed.top[0].next_step
    assert feed.top[0].source_href in feed.top[0].prepared_prompt
    assert feed.opportunities[0].source_href.endswith("source=s2")
    assert "likelihood" in feed.opportunities[0].prepared_prompt


async def test_deferred_signals_leave_focus_and_expired_deferrals_return():
    key = "customer_action:a1"
    records = Records(open_actions=[action("a1")], deferrals={key: stamp(-1)})
    service = AttentionService(records)
    feed = await service.feed()
    assert feed.top == [] and feed.neglected == [] and feed.deferred == 1
    records.data["deferrals"][key] = stamp(1)
    assert (await service.feed()).top[0].key == key


async def test_opening_and_refreshing_facts_never_invokes_a_model():
    class Model:
        calls = 0

        async def _structured(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("not configured")

    records = Records(open_actions=[action("a1")])
    model = Model()
    service = MorningBrief(AttentionService(records), records, model)
    first = await service.compose()
    assert first.waiting_total == 1 and first.focus[0].next_step
    records.data["open_actions"] = []
    second = await service.compose()
    assert second.waiting_total == 0 and second.focus == []
    assert model.calls == 0
    # Prose is an explicit opt-in, and its failure does not erase the facts.
    records.data["open_actions"] = [action("a1")]
    written = await service.compose(refresh=True)
    assert model.calls == 1 and written.narrative and written.focus


async def test_only_reviewed_recent_needs_without_attached_followups_are_signals(
    tmp_path,
    monkeypatch,
):
    database = Database(tmp_path / "briefing.db")
    await database.open()
    try:
        account = await database.create_customer_account("Acme", [], "", "")
        for title, reviewed, has_action in [
            ("Unreviewed", False, False),
            ("New need", True, False),
            ("Already followed up", True, True),
        ]:
            source, _ = await database.capture_customer_source(
                account_id=account["id"],
                source_kind="manual",
                title=title,
                content=f"Customer conversation: {title}",
                source_ref="",
                occurred_at=None,
            )
            extraction = {
                "facts": [{"kind": "requirement", "content": title}],
                "actions": [{"description": "Schedule discovery"}]
                if has_action
                else [],
            }
            proposal = await database.create_customer_proposal(
                source_id=source["id"],
                account_id=account["id"],
                extraction=extraction,
                model="test",
                prompt_version="1",
            )
            if reviewed:
                await database.save_customer_proposal(proposal["id"], extraction)
        disputed = await database.create_customer_fact(
            account["id"], kind="use_case", content="Disputed need"
        )
        await database.update_customer_fact(
            disputed["id"], kind="use_case", content="Disputed need", status="disputed"
        )
        with monkeypatch.context() as clock:
            clock.setattr(
                "waqil_api.database._now",
                lambda: (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            )
            await database.create_customer_fact(
                account["id"], kind="requirement", content="A need from last month"
            )
        await database.create_customer_fact(
            account["id"], kind="risk", content="Risk is not an opportunity"
        )
        feed = await AttentionService(database).feed()
        assert [item.title for item in feed.opportunities] == ["New need"]
        assert "tab=sources&source=" in feed.opportunities[0].source_href
        assert len([item for item in feed.items if item.kind == "customer_action"]) == 1
        assert any(
            item.kind == "customer_note" and item.title == "Unreviewed"
            for item in feed.items
        )
    finally:
        await database.close()


def test_completing_a_missing_commitment_reports_skipped(client):
    key = "customer_action:missing-action"
    response = client.post(
        "/api/v1/attention/batch", json={"keys": [key], "decision": "approve"}
    )
    assert response.status_code == 200
    assert response.json()["applied"] == []
    assert response.json()["skipped"] == [key]


def test_completing_a_commitment_removes_it_from_the_next_brief(client):
    account = client.post("/api/v1/customers", json={"name": "Acme"}).json()
    action = client.post(
        f"/api/v1/customers/{account['id']}/actions",
        json={"description": "Send the proposal"},
    ).json()
    key = f"customer_action:{action['id']}"
    assert any(
        item["key"] == key for item in client.get("/api/v1/attention").json()["items"]
    )
    result = client.post(
        "/api/v1/attention/batch", json={"keys": [key], "decision": "approve"}
    ).json()
    assert result["applied"] == [key]
    assert not any(item["key"] == key for item in result["feed"]["items"])
    assert not any(
        item["key"] == key
        for item in client.get("/api/v1/attention/brief").json()["focus"]
    )
