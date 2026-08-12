"""The customer-agent routing matrix.

Proves the behaviour the static catalog can't: given the model's chosen tools,
each one routes to exactly one place, every tool is reachable, a bundle routes
sensibly, and a routing failure falls back to answering — never a wrong write.
The model's *choice* is mocked here; that a real message elicits the right
choice is the model's job, exercised separately. This file pins the wiring.
"""

from starlette.testclient import TestClient

from waqil_api import customer_tools as ct
from waqil_api.contracts import CustomerAgentStepV1, CustomerToolCallV1


def _account(client: TestClient, name: str) -> str:
    created = client.post("/api/v1/customers", json={"name": name})
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


async def _swallow(*_args: object, **_kwargs: object) -> None:
    return None


def _step(*names: str) -> CustomerAgentStepV1:
    return CustomerAgentStepV1(
        calls=[
            CustomerToolCallV1(
                name=name,
                title="T" if name in (ct.FILE_NOTE, ct.RECORD_WIN) else "",
            )
            for name in names
        ]
    )


def _route(client: TestClient, account_id: str, step: CustomerAgentStepV1) -> dict:
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    cp.events.emit = _swallow  # type: ignore[attr-defined]

    async def _structured(_schema: object, **_kwargs: object) -> CustomerAgentStepV1:
        return step

    cp.model._structured = _structured  # type: ignore[attr-defined]
    return client.portal.call(  # type: ignore[attr-defined]
        cp._customer_route,
        {
            "run_id": "run_route",
            "conversation_id": "conv_route",
            "prompt": "the user's message",
            "model_aliases": {"_customer_id": account_id},
        },
    )


def test_a_question_routes_to_the_answer_path(client: TestClient) -> None:
    """Empty plan (nothing actionable) → the evidence-gated answer path, unchanged."""
    result = _route(client, _account(client, "Ans A"), CustomerAgentStepV1())
    assert result["route_kind"] == "direct"
    assert "customer_calls" not in result


def test_answer_only_routes_to_the_answer_path(client: TestClient) -> None:
    result = _route(client, _account(client, "Ans B"), _step(ct.ANSWER))
    assert result["route_kind"] == "direct"


def test_file_note_routes_to_execute(client: TestClient) -> None:
    result = _route(client, _account(client, "Note A"), _step(ct.FILE_NOTE))
    assert result["route_kind"] == "customer_execute"
    assert [c["name"] for c in result["customer_calls"]] == [ct.FILE_NOTE]


def test_generate_tracker_routes_to_execute(client: TestClient) -> None:
    result = _route(client, _account(client, "Trk A"), _step(ct.GENERATE_TRACKER))
    assert result["route_kind"] == "customer_execute"
    assert [c["name"] for c in result["customer_calls"]] == [ct.GENERATE_TRACKER]


def test_record_activity_routes_to_the_action_node(client: TestClient) -> None:
    """A message touching the action list goes whole to the node that owns the
    closure approval, not to the additive execute node."""
    result = _route(client, _account(client, "Act A"), _step(ct.RECORD_ACTIVITY))
    assert result["route_kind"] == "queue_update"


def test_activity_bundled_with_a_note_routes_whole_to_the_action_node(
    client: TestClient,
) -> None:
    """queue_update handles the note and the action together, so the bundle
    routes there rather than splitting across two nodes."""
    result = _route(
        client, _account(client, "Act B"), _step(ct.FILE_NOTE, ct.RECORD_ACTIVITY)
    )
    assert result["route_kind"] == "queue_update"


def test_multi_action_without_activity_executes_in_order(client: TestClient) -> None:
    result = _route(
        client, _account(client, "Multi A"), _step(ct.FILE_NOTE, ct.RECORD_WIN)
    )
    assert result["route_kind"] == "customer_execute"
    assert [c["name"] for c in result["customer_calls"]] == [
        ct.FILE_NOTE,
        ct.RECORD_WIN,
    ]


def test_a_routing_failure_falls_back_to_answering(client: TestClient) -> None:
    """The model going down must never route a message to a write — it answers."""
    account_id = _account(client, "Fail A")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    cp.events.emit = _swallow  # type: ignore[attr-defined]

    async def _boom(_schema: object, **_kwargs: object) -> CustomerAgentStepV1:
        raise RuntimeError("model unavailable")

    cp.model._structured = _boom  # type: ignore[attr-defined]
    result = client.portal.call(  # type: ignore[attr-defined]
        cp._customer_route,
        {
            "run_id": "r",
            "conversation_id": "c",
            "prompt": "anything",
            "model_aliases": {"_customer_id": account_id},
        },
    )
    assert result["route_kind"] == "direct"


def test_execute_generates_the_tracker_markdown(client: TestClient) -> None:
    """The generate_tracker handler returns the account's tracker into the thread."""
    account_id = _account(client, "Trk Exec")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    cp.events.emit = _swallow  # type: ignore[attr-defined]
    result = client.portal.call(  # type: ignore[attr-defined]
        cp._customer_execute,
        {
            "run_id": "r",
            "conversation_id": "c",
            "prompt": "tracker please",
            "model_aliases": {"_customer_id": account_id},
            "customer_calls": [{"name": ct.GENERATE_TRACKER}],
        },
    )
    assert "Customer Activity" in result["response_text"]


def _note_with_action(client: TestClient, account_id: str, content: str) -> str:
    """Capture a note and analyse it; returns the pending proposal id. The test
    extraction turns an 'Action:' line into one proposed action."""
    src = client.post(
        "/api/v1/customers/sources",
        json={
            "account_id": account_id,
            "source_kind": "note",
            "title": "Call",
            "content": content,
        },
    )
    assert src.status_code == 201, src.text
    proposal = client.post(f"/api/v1/customers/sources/{src.json()['id']}/analyze")
    assert proposal.status_code == 200, proposal.text
    return str(proposal.json()["id"])


def test_apply_endpoint_commits_an_extraction_to_the_profile(
    client: TestClient,
) -> None:
    """One-click apply: nothing is on the profile until Apply is called, then the
    proposed items land — and a second apply is refused, so it can't write twice."""
    account_id = _account(client, "Apply A")
    proposal_id = _note_with_action(
        client, account_id, "Action: send the pricing by Friday"
    )

    # It's still a proposal — the record is untouched.
    assert client.get(f"/api/v1/customers/{account_id}").json()["actions"] == []

    applied = client.post(f"/api/v1/customers/proposals/{proposal_id}/apply")
    assert applied.status_code == 200, applied.text

    actions = client.get(f"/api/v1/customers/{account_id}").json()["actions"]
    assert any("pricing" in a["description"].lower() for a in actions)

    # Decided now — applying again is a conflict, not a double write.
    assert (
        client.post(f"/api/v1/customers/proposals/{proposal_id}/apply").status_code
        == 409
    )


def test_apply_extraction_handler_commits_the_named_proposal(
    client: TestClient,
) -> None:
    """The agent's apply_extraction tool commits the proposal it was handed."""
    account_id = _account(client, "Apply Agent")
    proposal_id = _note_with_action(client, account_id, "Action: schedule the QBR")

    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    cp.events.emit = _swallow  # type: ignore[attr-defined]
    result = client.portal.call(  # type: ignore[attr-defined]
        cp._customer_execute,
        {
            "run_id": "r",
            "conversation_id": "c",
            "prompt": "yes, apply those",
            "model_aliases": {"_customer_id": account_id},
            "customer_calls": [
                {"name": ct.APPLY_EXTRACTION, "proposal_id": proposal_id}
            ],
        },
    )
    assert "Applied" in result["response_text"]
    actions = client.get(f"/api/v1/customers/{account_id}").json()["actions"]
    assert any("qbr" in a["description"].lower() for a in actions)


def test_filing_a_note_on_cloud_surfaces_a_one_click_apply_card(
    client: TestClient,
) -> None:
    """The keystone: filing a note analyses it in the same turn and emits the
    apply-card event — the write still gated behind the user's tap."""
    account_id = _account(client, "Card A")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]
    cp.customers._cloud_pinned = lambda: (
        True
    )  # analyse in-turn  # type: ignore[attr-defined]

    emitted: list[tuple[str, dict]] = []

    async def _capture(
        run_id: str, thread_id: str, event_type: str, payload=None, *_a, **_k
    ):
        emitted.append((event_type, payload or {}))

    cp.events.emit = _capture  # type: ignore[attr-defined]
    result = client.portal.call(  # type: ignore[attr-defined]
        cp._customer_execute,
        {
            "run_id": "r",
            "conversation_id": "c",
            "prompt": "Action: send the SOW to procurement",
            "model_aliases": {"_customer_id": account_id},
            "customer_calls": [{"name": ct.FILE_NOTE, "title": "SOW"}],
        },
    )
    assert "apply" in result["response_text"].lower()
    cards = [p for (t, p) in emitted if t == "customer.action_suggested"]
    assert cards, "no apply-card event emitted"
    assert cards[0]["proposal_id"]
    assert cards[0]["kind"] == ct.APPLY_EXTRACTION
    assert cards[0]["actions"] >= 1  # the deterministic extraction found the action

    # Still just a proposal — the card was surfaced, nothing was written.
    assert client.get(f"/api/v1/customers/{account_id}").json()["actions"] == []


def test_unscoped_named_account_routes_through_the_agent_auto_scoped(
    client: TestClient,
) -> None:
    """The unification: 'add a note to <acronym>' with no chip resolves the
    account, emits the auto-scope event, and runs the SAME agent scoped to it —
    so scoped and unscoped behave identically from here on."""
    from waqil_api.contracts import CustomerAgentStepV1, CustomerToolCallV1

    account_id = _account(client, "Zephyr Quantum Labs (ZQL)")
    cp = client.app.state.runtime.control_plane  # type: ignore[attr-defined]

    seen: list[tuple[str, dict]] = []

    async def _capture(
        run_id: str, thread_id: str, event_type: str, payload=None, *_a, **_k
    ):
        seen.append((event_type, payload or {}))

    async def _route_file_note(
        _schema: object, **_kwargs: object
    ) -> CustomerAgentStepV1:
        return CustomerAgentStepV1(
            calls=[CustomerToolCallV1(name="file_note", title="Note")]
        )

    cp.events.emit = _capture  # type: ignore[attr-defined]
    cp.model._structured = _route_file_note  # type: ignore[attr-defined]

    result = client.portal.call(  # type: ignore[attr-defined]
        cp._plan,
        {
            "run_id": "r",
            "conversation_id": "c",
            "prompt": "add a note to ZQL: kickoff complete",
            "model_aliases": {},
        },
    )
    # Routed to the agent's execute path, scoped to the resolved account...
    assert result["route_kind"] == "customer_execute"
    assert result["resolved_customer_id"] == account_id
    # ...and the chat auto-scopes (the chip appears from this event).
    assert any(
        t == "customer.scoped" and p.get("account_id") == account_id for t, p in seen
    )
