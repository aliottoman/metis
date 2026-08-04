"""Model Broker — the framework of control for a tool's runtime model access."""
from __future__ import annotations

import pytest

from waqil_api.contracts import ModelAccessV1, ModelResultV1
from waqil_api.model_broker import BrokerBudgetExceeded, BrokerError, ModelBroker


class _FakeModel:
    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def generate(self, request, on_token=None, *, model_aliases=None, on_reasoning=None):
        self.calls.append(
            {"role": request.role, "system": request.system_prompt, "user": request.user_prompt}
        )
        return ModelResultV1(model="fake", content=self.reply)


class _FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None):
        self.events.append((event_type, payload or {}))


def _broker(access: ModelAccessV1, model=None, events=None) -> ModelBroker:
    return ModelBroker(
        model=model or _FakeModel(),
        access=access,
        events=events or _FakeEvents(),
        run_id="run_1",
        conversation_id="conv_1",
        tool_slug="reference-architecture-generator",
    )


_ENABLED = ModelAccessV1(
    enabled=True,
    roles=["coder"],
    max_calls_per_run=2,
    max_tokens_per_call=1024,
    prompt_templates={"author_diagram_code": "PINNED SYSTEM PROMPT"},
)


@pytest.mark.asyncio
async def test_call_uses_pinned_prompt_and_audits() -> None:
    model, events = _FakeModel("diagram source"), _FakeEvents()
    broker = _broker(_ENABLED, model, events)
    out = await broker.call(
        template_id="author_diagram_code", role="coder", params={"spec": {"x": 1}}
    )
    assert out == "diagram source"
    # The system prompt is the pinned template — the tool cannot change it.
    assert model.calls[0]["system"] == "PINNED SYSTEM PROMPT"
    assert "spec" in model.calls[0]["user"]  # params go in the user turn
    # Exactly one audit event was emitted.
    assert [t for t, _ in events.events] == ["run.broker_call"]
    payload = events.events[0][1]
    assert payload["tool"] == "reference-architecture-generator"
    assert payload["template"] == "author_diagram_code"
    assert payload["call_index"] == 1


@pytest.mark.asyncio
async def test_budget_is_hard_enforced() -> None:
    broker = _broker(_ENABLED)  # budget = 2
    await broker.call(template_id="author_diagram_code", role="coder", params={})
    await broker.call(template_id="author_diagram_code", role="coder", params={})
    assert broker.calls_remaining == 0
    with pytest.raises(BrokerBudgetExceeded):
        await broker.call(template_id="author_diagram_code", role="coder", params={})


@pytest.mark.asyncio
async def test_disallowed_role_is_rejected() -> None:
    broker = _broker(_ENABLED)
    with pytest.raises(BrokerError, match="role 'planner' is not permitted"):
        await broker.call(template_id="author_diagram_code", role="planner", params={})


@pytest.mark.asyncio
async def test_unknown_template_is_rejected() -> None:
    broker = _broker(_ENABLED)
    with pytest.raises(BrokerError, match="unknown prompt template"):
        await broker.call(template_id="ghost_template", role="coder", params={})


@pytest.mark.asyncio
async def test_disabled_access_rejects_all_calls() -> None:
    broker = _broker(ModelAccessV1())  # disabled by default
    assert broker.enabled is False
    with pytest.raises(BrokerError, match="no model access"):
        await broker.call(template_id="author_diagram_code", role="coder", params={})


@pytest.mark.asyncio
async def test_rejected_call_does_not_spend_budget() -> None:
    model = _FakeModel()
    broker = _broker(_ENABLED, model)
    with pytest.raises(BrokerError):
        await broker.call(template_id="author_diagram_code", role="planner", params={})
    # A rejected call must not have invoked the model or consumed budget.
    assert model.calls == []
    assert broker.calls_made == 0


@pytest.mark.asyncio
async def test_scripted_model_replays_then_empties() -> None:
    # The hermetic build-eval stand-in: replies in order, then empty so the tool's
    # deterministic fallback path is also exercised by evals.
    from waqil_api.model_broker import ScriptedModel

    scripted = ScriptedModel(["first", "second"])
    events = _FakeEvents()
    broker = ModelBroker(
        model=scripted,
        access=ModelAccessV1(
            enabled=True,
            roles=["reviewer"],
            max_calls_per_run=4,
            max_tokens_per_call=512,
            prompt_templates={"summarize": "PINNED"},
        ),
        events=events,
        run_id="run_1",
        conversation_id="conv_1",
        tool_slug="readme-summary",
    )
    assert await broker.call(template_id="summarize", role="reviewer", params={}) == "first"
    assert await broker.call(template_id="summarize", role="reviewer", params={}) == "second"
    assert await broker.call(template_id="summarize", role="reviewer", params={}) == ""
    # Every brokered call — including scripted ones — is audited.
    assert [event for event, _ in events.events].count("run.broker_call") == 3
