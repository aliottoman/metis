"""P9 — the README summary interpreter and the tool-contract checker.

The interpreter is the declarative tool's whole runtime: one brokered call with
a deterministic host fallback, then an output-contract check. These tests prove
it (a) uses a valid model reply, (b) NEVER fails — it degrades to a deterministic
summary on any broker/parse problem, and (c) always emits a contract-valid output.
"""
from __future__ import annotations

import pytest

from waqil_api import readme_summary, tool_authoring, tool_contracts
from waqil_api.contracts import ModelAccessV1, ToolDefinitionDraftV1
from waqil_api.model_broker import BrokerBudgetExceeded, ModelBroker, ScriptedModel


class _Events:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def emit(self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None):
        self.events.append(event_type)


class _BoomModel:
    async def generate(self, request, *, model_aliases=None):
        raise RuntimeError("model is down")


def _definition():
    draft = ToolDefinitionDraftV1(name="Readme Summary", description="summarize a readme")
    return tool_authoring.harden_draft(draft, slug="readme-summary", max_broker_calls=4)


def _broker(model, definition, events):
    return ModelBroker(
        model=model,
        access=definition.capability_profile.model_access,
        events=events,
        run_id="run_1",
        conversation_id="conv_1",
        tool_slug=definition.slug,
    )


_README = "# Widget\nWidget is a CLI.\n## Components\n- parser\n- renderer\n## Built with\nRust\n"


@pytest.mark.asyncio
async def test_valid_model_reply_is_used() -> None:
    definition = _definition()
    events = _Events()
    reply = (
        '{"title": "Widget", "purpose": "A CLI tool.", "components": ["parser", "renderer"], '
        '"stack": ["Rust"], "summary": "Widget is a small CLI written in Rust."}'
    )
    output, meta = await readme_summary.run(
        definition, {"text": _README}, _broker(ScriptedModel([reply]), definition, events)
    )
    assert meta["authored_by"] == "model"
    assert output["title"] == "Widget"
    assert output["components"] == ["parser", "renderer"]
    ok, _ = tool_contracts.matches_contract(output, definition.output_contract)
    assert ok
    assert "run.broker_call" in events.events  # the brokered call was audited


@pytest.mark.asyncio
async def test_malformed_reply_falls_back_to_deterministic() -> None:
    definition = _definition()
    output, meta = await readme_summary.run(
        definition, {"text": _README}, _broker(ScriptedModel(["not json at all"]), definition, _Events())
    )
    assert meta["authored_by"] == "deterministic-fallback"
    # The deterministic summary still satisfies the contract and reads the README.
    ok, _ = tool_contracts.matches_contract(output, definition.output_contract)
    assert ok
    assert output["title"] == "Widget"
    assert "parser" in output["components"]
    assert output["stack"] == ["Rust"]


@pytest.mark.asyncio
async def test_broker_error_never_fails_the_run() -> None:
    definition = _definition()
    output, meta = await readme_summary.run(
        definition, {"text": _README}, _broker(_BoomModel(), definition, _Events())
    )
    assert meta["authored_by"] == "deterministic-fallback"
    assert "broker error" in meta["fallback_reason"]
    ok, _ = tool_contracts.matches_contract(output, definition.output_contract)
    assert ok


@pytest.mark.asyncio
async def test_budget_exhaustion_degrades_cleanly() -> None:
    definition = _definition()
    events = _Events()
    broker = _broker(ScriptedModel(["{}"]), definition, events)
    # Spend the single call, then run — the interpreter must fall back, not raise.
    await broker.call(template_id="summarize", role="reviewer", params={"text": "x"})
    with pytest.raises(BrokerBudgetExceeded):
        await broker.call(template_id="summarize", role="reviewer", params={"text": "y"})
    output, meta = await readme_summary.run(definition, {"text": _README}, broker)
    assert meta["authored_by"] == "deterministic-fallback"
    ok, _ = tool_contracts.matches_contract(output, definition.output_contract)
    assert ok


@pytest.mark.asyncio
async def test_partial_reply_is_coerced_and_completed() -> None:
    definition = _definition()
    # Only a title; the rest must be filled from the deterministic fallback.
    output, _ = await readme_summary.run(
        definition, {"text": _README}, _broker(ScriptedModel(['{"title": "Custom Title"}']), definition, _Events())
    )
    assert output["title"] == "Custom Title"
    ok, _ = tool_contracts.matches_contract(output, definition.output_contract)
    assert ok
    assert output["summary"]  # filled from fallback


@pytest.mark.asyncio
async def test_no_model_access_uses_fallback() -> None:
    definition = _definition()
    disabled = definition.model_copy(
        update={
            "capability_profile": definition.capability_profile.model_copy(
                update={"model_access": ModelAccessV1()}
            )
        }
    )
    output, meta = await readme_summary.run(
        disabled, {"text": _README}, _broker(ScriptedModel(["{}"]), disabled, _Events())
    )
    assert meta["authored_by"] == "deterministic-fallback"
    ok, _ = tool_contracts.matches_contract(output, disabled.output_contract)
    assert ok


# ── tool_contracts.matches_contract (the JSON-Schema subset checker) ──────────


def test_contract_checker_accepts_valid_object() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {"type": "string"}}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    ok, problems = tool_contracts.matches_contract({"a": "x", "b": ["y"]}, schema)
    assert ok and not problems


def test_contract_checker_flags_missing_and_wrong_types() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {"type": "string"}}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    ok, problems = tool_contracts.matches_contract({"a": 1, "b": [2], "c": 3}, schema)
    assert not ok
    joined = " ".join(problems)
    assert "a" in joined and "b" in joined and "c" in joined
