"""P9 — the declarative build's evaluation gate (fail-closed).

`_declarative_build` only proposes a tool for Gate-2 activation when its hermetic
eval passes. These tests exercise the real `ControlPlane._evaluate_declarative`
directly: a valid archetype passes through the host-owned scripted fixtures; a
definition with no fixtures fails closed (never activatable).
"""
from __future__ import annotations

import types

import pytest

from waqil_api import tool_authoring
from waqil_api.contracts import ToolDefinitionDraftV1
from waqil_api.control_plane import ControlPlane


class _Events:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def emit(self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None):
        self.events.append(event_type)


def _cp() -> types.SimpleNamespace:
    cp = types.SimpleNamespace(events=_Events())
    cp._check_properties = ControlPlane._check_properties.__get__(cp)
    return cp


def _definition():
    draft = ToolDefinitionDraftV1(name="Readme Summary", description="summarize a readme")
    return tool_authoring.harden_draft(draft, slug="readme-summary", max_broker_calls=4)


_STATE = {"run_id": "run_1", "conversation_id": "conv_1"}


@pytest.mark.asyncio
async def test_valid_archetype_passes_eval() -> None:
    cp = _cp()
    report = await ControlPlane._evaluate_declarative(cp, _STATE, _definition())
    assert report.passed is True
    assert report.score == 1.0
    assert len(report.results) == len(tool_authoring.get_archetype("text-summary").eval_fixtures)
    # The scripted broker calls during eval were audited.
    assert cp.events.events.count("run.broker_call") >= 1


@pytest.mark.asyncio
async def test_definition_without_fixtures_fails_closed() -> None:
    cp = _cp()
    # An unknown archetype has no eval fixtures — the build must NOT be activatable.
    orphan = _definition().model_copy(update={"archetype": "does-not-exist"})
    report = await ControlPlane._evaluate_declarative(cp, _STATE, orphan)
    assert report.passed is False
    assert report.results[0].case_id == "no-eval-cases"


@pytest.mark.asyncio
async def test_a_text_tool_with_no_text_says_so_instead_of_summarising_nothing() -> None:
    """The deterministic fallback is written never to fail, so an empty input
    produced a card reading "Untitled Project" for its title, its purpose and
    its summary — a summary of nothing, rendered as a summary. Measured live on
    the kimi lane from "build me a tool that summarises things"."""
    definition = _definition()
    cp = _cp()
    cp.settings = types.SimpleNamespace(tool_disabled_slugs=[])
    cp._prepare_tool_input = ControlPlane._prepare_tool_input.__get__(cp)

    async def _guard(_state):
        return None

    async def _stage(*_args, **_kwargs):
        return None

    class _Policy:
        def enforce(self) -> None:
            return None

    async def _policy_gate(*_args, **_kwargs):
        return _Policy()

    class _Database:
        async def get_runnable_definition(self, _slug):
            return definition

    cp._guard = _guard
    cp._stage = _stage
    cp._policy_gate = _policy_gate
    cp.database = _Database()

    state = {
        **_STATE,
        "plan": {
            "schema_version": "1",
            "summary": "run it",
            "route": "existing_tool",
            "tool_slug": definition.slug,
            "risk_level": "R2",
        },
        "attachment_text": "",
    }
    result = await ControlPlane._declarative_execute(cp, state)
    assert "had none" in result["response_text"]
    assert "tool.input_missing" in cp.events.events
    # Nothing was produced, so nothing is offered as output.
    assert "tool_output" not in result
