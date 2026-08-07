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


async def test_queue_update_only_closes_actions_the_host_offered() -> None:
    """The iron-tight part: a model may pick from the supplied list, never
    compose an id. An invented or foreign id is reported, never applied."""
    from waqil_api.contracts import ActionResolutionV1, NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    actions = [
        {"id": "cact_real", "account_id": "cust_1", "account_name": "BAPCO",
         "description": "Walk through the tenancy"},
    ]
    proposal = QueueUpdateV1(
        completed=[
            ActionResolutionV1(action_id="cact_real", note="done in the session"),
            ActionResolutionV1(action_id="cact_invented"),
        ],
        new_actions=[NewActionV1(description="Send the pricing sheet")],
    )
    cleaned, matched = validate(proposal, actions)
    assert [item.action_id for item in cleaned.completed] == ["cact_real"]
    assert any("cact_invented" in item for item in cleaned.unmatched)
    assert len(matched) == 1
    # The follow-up inherits the only account in play rather than guessing.
    assert cleaned.new_actions[0].account_id == "cust_1"


async def test_a_new_action_with_no_resolvable_account_is_refused() -> None:
    from waqil_api.contracts import NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    actions = [
        {"id": "a", "account_id": "cust_1", "account_name": "A", "description": "x"},
        {"id": "b", "account_id": "cust_2", "account_name": "B", "description": "y"},
    ]
    proposal = QueueUpdateV1(new_actions=[NewActionV1(description="Ambiguous follow-up")])
    cleaned, _ = validate(proposal, actions)
    assert cleaned.new_actions == []
    assert any("no account" in item for item in cleaned.unmatched)


async def test_a_foreign_account_on_a_new_action_is_refused() -> None:
    from waqil_api.contracts import NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    actions = [{"id": "a", "account_id": "cust_1", "account_name": "A", "description": "x"}]
    proposal = QueueUpdateV1(
        new_actions=[NewActionV1(description="Sneaky", account_id="cust_other")]
    )
    cleaned, _ = validate(proposal, actions)
    assert cleaned.new_actions == []
    assert any("unknown account" in item for item in cleaned.unmatched)


def test_queue_update_intent_is_conservative() -> None:
    from waqil_api.queue_update import is_queue_update_request

    assert is_queue_update_request("I completed the meeting with Bank Pivdenny")
    assert is_queue_update_request("we met with BAPCO and walked the tenancy")
    assert is_queue_update_request("add a follow-up to send the pricing sheet")
    # Talking *about* work is not reporting it done.
    assert not is_queue_update_request("what did I commit to for BAPCO?")
    assert not is_queue_update_request("summarize the meeting notes")


async def test_a_new_action_cannot_be_born_overdue() -> None:
    """A model with no clock resolved "next week" to a date in a previous year;
    the queue would then rank that follow-up as overdue, inventing urgency
    ahead of real commitments."""
    from datetime import UTC, datetime

    from waqil_api.contracts import NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    actions = [{"id": "a", "account_id": "c1", "account_name": "A", "description": "x"}]
    today = datetime(2026, 8, 7, tzinfo=UTC)
    proposal = QueueUpdateV1(
        new_actions=[
            NewActionV1(
                description="Send the pricing sheet",
                account_id="c1",
                due_at=datetime(2025, 6, 23, tzinfo=UTC),
            )
        ]
    )
    cleaned, _ = validate(proposal, actions, today=today)
    assert cleaned.new_actions[0].due_at is None
    assert any("past due date" in item for item in cleaned.unmatched)


async def test_a_future_due_date_is_kept() -> None:
    from datetime import UTC, datetime

    from waqil_api.contracts import NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    actions = [{"id": "a", "account_id": "c1", "account_name": "A", "description": "x"}]
    due = datetime(2026, 8, 14, tzinfo=UTC)
    cleaned, _ = validate(
        QueueUpdateV1(
            new_actions=[NewActionV1(description="Later", account_id="c1", due_at=due)]
        ),
        actions,
        today=datetime(2026, 8, 7, tzinfo=UTC),
    )
    assert cleaned.new_actions[0].due_at == due


def test_note_capture_and_cleanup_detection() -> None:
    from waqil_api.queue_update import (
        is_note_capture_request,
        is_queue_update_request,
        wants_cleanup,
    )

    assert is_note_capture_request("add this to DynaAI's notes")
    assert is_note_capture_request("capture this for the account")
    assert is_note_capture_request("note for BAPCO: they want a demo")
    assert is_queue_update_request("save this to their notes")
    # A question about notes is not a request to file one.
    assert not is_note_capture_request("what notes do we have for DynaAI?")
    assert wants_cleanup("add this to their notes but clean it up first")
    assert not wants_cleanup("add this to their notes")


def test_candidate_accounts_matches_named_and_scoped() -> None:
    from waqil_api.queue_update import candidate_accounts

    accounts = [
        {"id": "cust_1", "name": "DynaAI"},
        {"id": "cust_2", "name": "BAPCO"},
    ]
    named = candidate_accounts("please add this to Dyna.AI's notes", accounts)
    assert [a["id"] for a in named] == ["cust_1"]
    # A scoped conversation needs no naming — the account is already chosen.
    scoped = candidate_accounts("just note this", accounts, scoped_id="cust_2")
    assert [a["id"] for a in scoped] == ["cust_2"]
    # An account not named is not offered.
    assert candidate_accounts("note this somewhere", accounts) == []


def test_identifier_guard_catches_an_altered_reference() -> None:
    from waqil_api.queue_update import identifiers_preserved

    original = "SR 4-0003462742 and response chatcmpl-0add1506 for branch 176."
    # Reformatting is fine — normalised comparison ignores spacing/punctuation.
    assert identifiers_preserved(
        "## Case\n- SR4-0003462742\n- chatcmpl-0add1506\n- branch 176", original
    )
    # A single changed digit is caught.
    assert not identifiers_preserved("SR 4-0003462743", original)
    assert not identifiers_preserved("response chatcmpl-0add1507", original)


def test_a_note_is_dropped_when_its_account_was_not_offered() -> None:
    from waqil_api.contracts import CapturedNoteV1, QueueUpdateV1
    from waqil_api.queue_update import validate

    offered = [{"id": "cust_1", "name": "DynaAI"}]
    proposal = QueueUpdateV1(note=CapturedNoteV1(account_id="cust_other", title="X"))
    cleaned, _ = validate(proposal, [], offered_accounts=offered)
    assert cleaned.note is None
    assert any("unknown account" in item for item in cleaned.unmatched)


def test_a_valid_note_is_kept_and_a_todo_inherits_its_account() -> None:
    from waqil_api.contracts import CapturedNoteV1, NewActionV1, QueueUpdateV1
    from waqil_api.queue_update import has_changes, validate

    offered = [{"id": "cust_1", "name": "DynaAI"}]
    proposal = QueueUpdateV1(
        note=CapturedNoteV1(account_id="cust_1", title="Gemma4 issue"),
        new_actions=[NewActionV1(description="Respond to Jarod's email")],
    )
    cleaned, _ = validate(proposal, [], offered_accounts=offered)
    assert cleaned.note is not None and cleaned.note.account_id == "cust_1"
    # The todo had no account of its own; it inherits the note's.
    assert cleaned.new_actions[0].account_id == "cust_1"
    assert has_changes(cleaned)
