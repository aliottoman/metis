"""P1.2: a build that cannot make progress ends early and says so.

Two fail-fast rules under test. An empty plan — the model asked twice and
naming nothing — ends the turn before a single step is spent, because the
alternative was measured: thirty minutes of budget producing one 35-byte
__init__.py. And a streak of host-refused tool calls ends the turn while it
can still end honestly, instead of pushing useful evidence out of the trace
window one refusal at a time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from waqil_api.control_plane import ControlPlane


async def _noop(*args: object, **kwargs: object) -> None:
    return None


async def _empty_context(project_id: str) -> dict[str, object]:
    return {"manifest": {"file_tree": []}}


def _plane(model: object) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.model = model
    plane.events = SimpleNamespace(emit=_noop)
    plane.projects = SimpleNamespace(context=_empty_context)
    plane.settings = SimpleNamespace(
        project_agent_max_steps=48,
        project_staged_max_files=48,
        # The spec-rewrite stage stands down here; these tests are about the
        # manifest and the guards, not the rewrite (test_spec_rewrite owns that).
        project_spec_rewrite=False,
        project_spec_rewrite_max_chars=1800,
    )
    plane._guard = _noop
    plane._stage = _noop
    return plane


@pytest.mark.asyncio
async def test_an_empty_plan_ends_the_turn_before_any_step_is_spent() -> None:
    class NoPlanModel:
        def __init__(self) -> None:
            self.plan_calls = 0

        async def project_plan_files(self, request, *, model_aliases=None):
            self.plan_calls += 1
            return []

        async def project_step(self, request, *, model_aliases=None):
            raise AssertionError("a planless build turn must not reach the step loop")

    model = NoPlanModel()
    result = await ControlPlane._project_step(
        _plane(model),
        {
            "prompt": "Build an app that tracks supplier invoices and totals",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
        },
    )

    assert model.plan_calls == 2  # one immediate retry, then the verdict
    assert "could not produce a build plan" in result["response_text"]
    assert result["project_pending_call"] == {}


@pytest.mark.asyncio
async def test_a_plan_that_recovers_on_retry_continues_normally() -> None:
    class SecondTryModel:
        def __init__(self) -> None:
            self.plan_calls = 0
            self.step_calls = 0

        async def project_plan_files(self, request, *, model_aliases=None):
            self.plan_calls += 1
            return [] if self.plan_calls == 1 else ["app/main.py"]

        async def project_step(self, request, *, model_aliases=None):
            self.step_calls += 1
            from waqil_api.contracts import ProjectAgentStepV1, ProjectToolCallV1

            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(name="list_files", arguments={}),
            )

    model = SecondTryModel()
    plane = _plane(model)
    # The step request path reads the reference corpus; point it at nothing.
    plane.settings.project_reference_enabled = False
    plane.settings.project_reference_dir = "/nonexistent"
    plane.settings.project_reference_max_chars = 0
    plane.settings.project_reference_max_chars_local = 0
    result = await ControlPlane._project_step(
        plane,
        {
            "prompt": "Build an app that tracks supplier invoices and totals",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
        },
    )

    assert model.plan_calls == 2
    assert model.step_calls == 1
    assert result["project_planned_files"] == ["app/main.py"]


@pytest.mark.asyncio
async def test_a_refusal_streak_ends_the_turn_and_offers_staged_work() -> None:
    class MustNotRun:
        async def project_plan_files(self, request, *, model_aliases=None):
            raise AssertionError("the turn must end before planning")

        async def project_step(self, request, *, model_aliases=None):
            raise AssertionError("the turn must end before another step")

    staged = {"app/main.py": {"content": "x = 1\n", "origin": "create", "bytes": 6}}
    result = await ControlPlane._project_step(
        _plane(MustNotRun()),
        {
            "prompt": "Build an app that tracks supplier invoices and totals",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_refused_streak": 5,
            "project_staged": staged,
        },
    )
    assert "5 consecutive refused" in result["response_text"]
    assert "1 staged" in result["response_text"]
    assert result["project_refused_streak"] == 0

    bare = await ControlPlane._project_step(
        _plane(MustNotRun()),
        {
            "prompt": "Build an app that tracks supplier invoices and totals",
            "run_id": "run_x",
            "conversation_id": "conv_x",
            "model_aliases": {"_project_id": "asset_x"},
            "project_refused_streak": 5,
        },
    )
    assert "nothing staged" in bare["response_text"]
