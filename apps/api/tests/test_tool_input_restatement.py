"""A tool that cannot find values the user plainly gave gets one more try.

Measured live on both hosted lanes: "use the break-even calculator — fixed cost
50000, unit price 40, unit cost 25" returned "Missing required parameters.
Please provide Fixed Costs, Selling Price per Unit, and Variable Cost per Unit."
The tool's own regex only knows the words its author happened to write, so a
request naming every value in different words reads as an empty request. The
host restates the request in the tool's declared vocabulary and runs it once
more — translating labels, never values, which is checked rather than trusted.
"""

from __future__ import annotations

import types

import pytest

from waqil_api.contracts import ToolInputRestatementV1
from waqil_api.control_plane import ControlPlane

MISSING = {"error": "Missing required parameters. Please provide Fixed Costs."}
ANSWERED = {"break_even_units": 1666.67}

PROMPT = (
    "Use the break-even calculator for this: fixed cost 50000, unit price 40, "
    "unit cost 25."
)


class _Events:
    def __init__(self) -> None:
        self.emitted: list[str] = []

    async def emit(
        self, run_id, conversation_id, event_type, payload=None, checkpoint_id=None
    ):
        self.emitted.append(event_type)


def _definition() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        slug="break-even-calculator",
        name="break_even_calculator",
        description="Calculates the break-even point in units.",
        intent_examples=["Break-even for fixed costs, selling price, variable cost"],
        capability_profile=types.SimpleNamespace(
            model_access=types.SimpleNamespace(enabled=False, max_calls_per_run=0)
        ),
    )


def _plane(*, restatement: list[str], outputs: list[dict]) -> types.SimpleNamespace:
    """A control plane stubbed down to the two things this path needs: a model
    that restates, and a tool that answers differently the second time."""
    plane = types.SimpleNamespace(events=_Events(), seen_prompts=[])
    plane.settings = types.SimpleNamespace(
        tool_authored_timeout_seconds=10,
        tool_authored_memory_mb=256,
        tool_authored_model_call_timeout_seconds=300,
    )
    plane._reads_as_missing_input = ControlPlane._reads_as_missing_input
    plane._prepare_authored_inputs = lambda _definition, state: {
        "text": "",
        "prompt": state["prompt"],
    }
    plane._authored_bridge = lambda *_args, **_kwargs: None
    plane._restate_for_tool = ControlPlane._restate_for_tool.__get__(plane)
    plane.model = types.SimpleNamespace()

    async def _structured(_schema, **_kwargs):
        return ToolInputRestatementV1(lines=list(restatement))

    plane.model._structured = _structured
    plane.outputs = list(outputs)
    return plane


async def _run(plane, monkeypatch) -> tuple[dict, dict]:
    from waqil_api import authored_code

    async def _execute(_code, tool_inputs, **_kwargs):
        plane.seen_prompts.append(tool_inputs["prompt"])
        return plane.outputs.pop(0)

    monkeypatch.setattr(authored_code, "execute_authored", _execute)
    return await ControlPlane._run_authored(
        plane,
        {"run_id": "run_1", "conversation_id": "conv_1", "prompt": PROMPT},
        _definition(),
        types.SimpleNamespace(implementation="def run(inputs, model): ..."),
    )


@pytest.mark.asyncio
async def test_a_missing_parameter_error_is_retried_in_the_tools_own_words(
    monkeypatch,
) -> None:
    plane = _plane(
        restatement=[
            "Fixed Costs: 50000",
            "Selling Price per Unit: 40",
            "Variable Cost per Unit: 25",
        ],
        outputs=[MISSING, ANSWERED],
    )
    output, meta = await _run(plane, monkeypatch)
    assert output == ANSWERED
    assert "Fixed Costs: 50000" in plane.seen_prompts[1]
    assert "restated" in (meta["fallback_reason"] or "")
    assert "tool.input_restated" in plane.events.emitted


@pytest.mark.asyncio
async def test_a_restatement_that_changed_a_figure_is_thrown_away(monkeypatch) -> None:
    """The one real risk of relabelling. A tool answering confidently from an
    altered number is worse than a tool saying it found nothing."""
    plane = _plane(
        restatement=["Fixed Costs: 5000", "Selling Price per Unit: 40"],
        outputs=[MISSING, ANSWERED],
    )
    output, meta = await _run(plane, monkeypatch)
    assert output == MISSING
    assert meta["fallback_reason"] is None
    assert plane.seen_prompts == [PROMPT]


@pytest.mark.asyncio
async def test_a_tool_that_succeeds_first_time_is_never_asked_twice(
    monkeypatch,
) -> None:
    plane = _plane(restatement=["Fixed Costs: 50000"], outputs=[ANSWERED])
    output, _meta = await _run(plane, monkeypatch)
    assert output == ANSWERED
    assert plane.seen_prompts == [PROMPT]


@pytest.mark.asyncio
async def test_a_genuinely_absent_value_still_reports_the_first_error(
    monkeypatch,
) -> None:
    plane = _plane(restatement=["Fixed Costs: 50000"], outputs=[MISSING, MISSING])
    output, meta = await _run(plane, monkeypatch)
    assert output == MISSING
    assert meta["fallback_reason"] is None
    assert len(plane.seen_prompts) == 2
