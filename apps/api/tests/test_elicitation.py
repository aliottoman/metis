from __future__ import annotations

import time

from fastapi.testclient import TestClient

from waqil_api.control_plane import _elicitation_clarification
from waqil_api.main import create_app
from waqil_api.model_provider import (
    PlanningRequestV1,
    default_routing_catalog,
    normalize_plan_semantics,
    validate_plan_semantics,
)
from waqil_api.contracts import PlanEnvelopeV1

# The deterministic planner routes this exact phrase to an ask_user pause with a
# fixed question and two options, so the whole pause/resume path runs network-free.
ASK_PROMPT = "Ask me to choose which sample dataset to load."


def _req(prompt: str) -> PlanningRequestV1:
    return PlanningRequestV1(run_id="r", conversation_id="c", prompt=prompt)


def _wait(client: TestClient, run_id: str, statuses: set[str]) -> dict:
    deadline = time.monotonic() + 5
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/v1/runs/{run_id}").json()
        if latest["status"] in statuses:
            return latest
        time.sleep(0.02)
    raise AssertionError(latest)


# ── Planner semantics ────────────────────────────────────────────────────────


def test_planner_preserves_ask_user_when_warranted() -> None:
    plan = PlanEnvelopeV1(
        summary="ambiguous",
        route="ask_user",
        question="Which environment — staging or prod?",
        options=["staging", "prod"],
    )
    resolved = normalize_plan_semantics(plan, _req("deploy it"))
    validate_plan_semantics(resolved, _req("deploy it"))
    assert resolved.route == "ask_user"
    assert resolved.tool_slug is None
    assert resolved.risk_level.value == "R0"
    assert resolved.question == "Which environment — staging or prod?"
    assert resolved.options == ["staging", "prod"]
    assert [step.id for step in resolved.steps] == ["ask"]


def test_planner_downgrades_ask_user_with_blank_question() -> None:
    plan = PlanEnvelopeV1(summary="x", route="ask_user", question="   ")
    resolved = normalize_plan_semantics(plan, _req("hello"))
    assert resolved.route == "direct"


def test_a_build_follow_up_still_builds_the_pending_tool() -> None:
    """A clear build intent is never interrupted by a question while any
    deterministic route can still take it. "Build it" after a Gate-1 approval
    builds the approved definition — the pause sits after every such route, not
    in front of them."""
    from waqil_api.contracts import RiskLevel
    from waqil_api.model_provider import RoutingCatalog, ToolRoute

    catalog = RoutingCatalog(
        tools=[
            ToolRoute(
                slug="temp-converter",
                existing_risk=RiskLevel.R2,
                factory_risk=RiskLevel.R3,
                input_pipeline="none",
                runnable=False,
                buildable=True,
            )
        ],
        architecture_tool=default_routing_catalog().architecture_tool,
        known_slugs=frozenset({"temp-converter"}),
    )
    plan = PlanEnvelopeV1(summary="x", route="ask_user", question="which?")
    resolved = normalize_plan_semantics(plan, _req("build it now"), catalog)
    assert resolved.route == "tool_factory"
    assert resolved.tool_slug == "temp-converter"


def test_a_pending_tool_is_not_built_for_a_request_about_something_else() -> None:
    """Measured live: with a temperature converter pending, "build me a tool
    that summarises things" built and ran the converter. A request that names a
    job of its own is not a follow-up."""
    from waqil_api.contracts import RiskLevel
    from waqil_api.model_provider import RoutingCatalog, ToolRoute

    catalog = RoutingCatalog(
        tools=[
            ToolRoute(
                slug="temp-converter",
                existing_risk=RiskLevel.R2,
                factory_risk=RiskLevel.R3,
                input_pipeline="none",
                runnable=False,
                buildable=True,
            )
        ],
        architecture_tool=default_routing_catalog().architecture_tool,
        known_slugs=frozenset({"temp-converter"}),
    )
    plan = PlanEnvelopeV1(
        summary="x", route="ask_user", question="What should it take in?"
    )
    resolved = normalize_plan_semantics(
        plan, _req("build me a tool that summarises things"), catalog
    )
    assert resolved.route == "ask_user"
    assert resolved.tool_slug is None


def test_a_build_request_with_no_subject_may_ask_once() -> None:
    """Measured live on both hosted lanes. "Build me a tool that summarises
    things" could never reach the pause — an explicit build was excluded from
    it outright — so the factory drafted a README project-card tool on the bare
    word "summar", ran it against no text, and answered "Untitled Project"
    three times over. Nothing else claimed the request; asking beats guessing."""
    plan = PlanEnvelopeV1(
        summary="x",
        route="ask_user",
        question="What should it take in, and what should it give back?",
    )
    resolved = normalize_plan_semantics(
        plan, _req("build me a tool that summarises things")
    )
    validate_plan_semantics(resolved, _req("build me a tool that summarises things"))
    assert resolved.route == "ask_user"
    assert resolved.tool_slug is None


def test_validate_rejects_ask_user_carrying_a_tool() -> None:
    plan = PlanEnvelopeV1(
        summary="x", route="ask_user", question="q", tool_slug="something"
    )
    try:
        validate_plan_semantics(plan, _req("q"))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "no tool" in str(error)


def test_validate_rejects_ask_user_without_a_question() -> None:
    plan = PlanEnvelopeV1(summary="x", route="ask_user", question="")
    try:
        validate_plan_semantics(plan, _req("q"))
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "question" in str(error)


def test_ask_user_options_and_question_are_normalized() -> None:
    plan = PlanEnvelopeV1(
        summary="x", route="ask_user", question=" pick ", options=["  a  ", "", "b"]
    )
    resolved = normalize_plan_semantics(plan, _req("pick one"))
    assert resolved.question == "pick"
    assert resolved.options == ["a", "b"]


def test_elicitation_clarification_helper() -> None:
    block = _elicitation_clarification(
        {"question": "Which region?"}, {"option": "us-east", "text": None}
    )
    assert "Which region?" in block
    assert "us-east" in block
    # Both an option and free text are joined into one reply.
    both = _elicitation_clarification(
        {"question": "Q"}, {"option": "A", "text": "and also B"}
    )
    assert "A — and also B" in both
    # No pause, or an empty reply, adds nothing to the prompt.
    assert _elicitation_clarification(None, None) == ""
    assert _elicitation_clarification({"question": "Q"}, {"option": "", "text": ""}) == ""


# ── Full pause / resume through the API ──────────────────────────────────────


def _start_ask_run(client: TestClient) -> str:
    conversation = client.post("/api/v1/conversations", json={}).json()
    accepted = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={"content": ASK_PROMPT, "attachment_ids": []},
    ).json()
    return accepted["run_id"]


def test_ask_user_pauses_then_resumes_with_the_answer(client: TestClient) -> None:
    run_id = _start_ask_run(client)
    waiting = _wait(client, run_id, {"awaiting_input", "failed", "completed"})
    assert waiting["status"] == "awaiting_input", waiting

    # The pending question is surfaced for recovery, with the planner's exact text.
    recovery = client.get("/api/v1/runs?status=awaiting_input").json()
    pending = next(
        item["elicitation"] for item in recovery if item["run"]["id"] == run_id
    )
    assert pending is not None
    assert pending["question"] == "Which option would you like?"
    assert pending["options"] == ["Option A", "Option B"]

    answered = client.post(
        f"/api/v1/runs/{run_id}/answers", json={"option": "Option A"}
    )
    assert answered.status_code == 200, answered.text

    completed = _wait(client, run_id, {"completed", "failed"})
    assert completed["status"] == "completed", completed

    events = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
    assert "event: elicitation.requested" in events
    assert "event: run.awaiting_input" in events
    assert "event: elicitation.answered" in events
    assert "event: run.resumed" in events
    assert "event: run.completed" in events
    # The answer was folded into the synthesis prompt as the tool result, so the
    # deterministic echo carries it through to the finished reply.
    assert "They answered" in events
    assert "Option A" in events


def test_answer_is_rejected_once_the_run_is_no_longer_waiting(
    client: TestClient,
) -> None:
    run_id = _start_ask_run(client)
    _wait(client, run_id, {"awaiting_input"})
    first = client.post(f"/api/v1/runs/{run_id}/answers", json={"option": "Option B"})
    assert first.status_code == 200, first.text
    _wait(client, run_id, {"completed", "failed"})
    # A second answer has nothing to resume.
    again = client.post(f"/api/v1/runs/{run_id}/answers", json={"option": "Option A"})
    assert again.status_code == 409


def test_answer_payload_validation(client: TestClient) -> None:
    run_id = _start_ask_run(client)
    _wait(client, run_id, {"awaiting_input"})

    # An option that was never offered is refused.
    bad_option = client.post(
        f"/api/v1/runs/{run_id}/answers", json={"option": "Option Z"}
    )
    assert bad_option.status_code == 409

    # A reply that says nothing is refused.
    empty = client.post(
        f"/api/v1/runs/{run_id}/answers", json={"option": "", "text": ""}
    )
    assert empty.status_code == 422

    # A mismatched elicitation id is refused.
    wrong_id = client.post(
        f"/api/v1/runs/{run_id}/answers",
        json={"elicitation_id": "elic_nope", "option": "Option A"},
    )
    assert wrong_id.status_code == 409

    # The run is still answerable after the rejections.
    ok = client.post(f"/api/v1/runs/{run_id}/answers", json={"text": "the first one"})
    assert ok.status_code == 200, ok.text
    completed = _wait(client, run_id, {"completed", "failed"})
    assert completed["status"] == "completed"


def test_free_text_answer_is_carried_into_the_reply(client: TestClient) -> None:
    run_id = _start_ask_run(client)
    _wait(client, run_id, {"awaiting_input"})
    answered = client.post(
        f"/api/v1/runs/{run_id}/answers",
        json={"text": "load the quarterly figures"},
    )
    assert answered.status_code == 200, answered.text
    _wait(client, run_id, {"completed", "failed"})
    events = client.get(f"/api/v1/runs/{run_id}/events?after=0").text
    assert "load the quarterly figures" in events


def test_pending_question_survives_a_process_restart(settings) -> None:
    with TestClient(create_app(settings)) as first:
        run_id = _start_ask_run(first)
        waiting = _wait(first, run_id, {"awaiting_input", "failed"})
        assert waiting["status"] == "awaiting_input"

    with TestClient(create_app(settings)) as restarted:
        recovery = restarted.get("/api/v1/runs?status=awaiting_input").json()
        pending = next(
            item["elicitation"] for item in recovery if item["run"]["id"] == run_id
        )
        assert pending["question"] == "Which option would you like?"
        answered = restarted.post(
            f"/api/v1/runs/{run_id}/answers", json={"option": "Option B"}
        )
        assert answered.status_code == 200, answered.text
        completed = _wait(restarted, run_id, {"completed", "failed"})
        assert completed["status"] == "completed", completed
        events = restarted.get(f"/api/v1/runs/{run_id}/events?after=0").text
        assert "event: run.resumed" in events
        assert "Option B" in events
