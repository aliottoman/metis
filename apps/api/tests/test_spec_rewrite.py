"""The prescriptive spec rewrite: loose asks get compiled, real specs never do.

Measured on the same model, same day, same pipeline: a conversational build
prompt produced 38 blocking findings and its prescriptive rewrite produced 11.
This stage makes that rewrite part of the build itself — chosen by the shape
of the request rather than a switch, carried like the manifest, and honest
about every assumption it makes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from waqil_api.contracts import ProjectSpecV1
from waqil_api.control_plane import ControlPlane, _looks_prescriptive

LOOSE = (
    "Build out this project from scratch: a solution to help with onboarding "
    "for a bank, extract information from documents with a multimodal LLM and "
    "assess risk high-low against internal criteria."
)


async def _noop_emit(*args, **kwargs) -> None:
    return None


def _plane(model, *, rewrite=True, max_chars=1800):
    plane = object.__new__(ControlPlane)
    plane.model = model
    plane.events = SimpleNamespace(emit=_noop_emit)
    plane.settings = SimpleNamespace(
        project_spec_rewrite=rewrite,
        project_spec_rewrite_max_chars=max_chars,
        project_agent_max_steps=48,
        project_staged_max_files=48,
        project_reference_enabled=False,
        project_reference_dir=Path("/nonexistent-reference"),
        project_reference_max_chars=14_000,
        project_reference_max_chars_local=6_000,
    )
    return plane


class Rewriter:
    def __init__(self, spec="STACK: FastAPI\nFILES: app/main.py", assumptions=("no web research in v1",)):
        self.calls = 0
        self._spec = spec
        self._assumptions = list(assumptions)

    async def project_spec(self, request, *, model_aliases=None):
        self.calls += 1
        return ProjectSpecV1(spec=self._spec, assumptions=self._assumptions)


def test_structure_is_the_tell_for_an_existing_spec() -> None:
    assert _looks_prescriptive("STACK FastAPI backend (Python 3.13), one runtime")
    assert _looks_prescriptive("intro line\nFILES: app/main.py, app/store.py")
    assert not _looks_prescriptive(LOOSE)
    assert not _looks_prescriptive("please stack the results and file them nicely")


@pytest.mark.asyncio
async def test_a_loose_whole_app_request_is_compiled_once() -> None:
    model = Rewriter()
    plane = _plane(model)
    state = {"prompt": LOOSE, "run_id": "r", "conversation_id": "c"}

    info = await ControlPlane._project_spec_rewrite(plane, state, 0, {})
    assert info is not None
    assert info["spec"].startswith("STACK")
    assert info["assumptions"] == ["no web research in v1"]
    assert model.calls == 1

    # Carried in state afterwards: asked exactly once per turn.
    carried = await ControlPlane._project_spec_rewrite(
        plane, {**state, "project_spec": info}, 3, {"a": {}}
    )
    assert carried == info
    assert model.calls == 1


@pytest.mark.asyncio
async def test_the_rewrite_stands_down_where_it_must() -> None:
    model = Rewriter()
    state = {"prompt": LOOSE, "run_id": "r", "conversation_id": "c"}
    # Mid-turn, staged work, or the setting off: never rewritten.
    assert await ControlPlane._project_spec_rewrite(_plane(model), state, 1, {}) is None
    assert await ControlPlane._project_spec_rewrite(_plane(model), state, 0, {"a": {}}) is None
    assert (
        await ControlPlane._project_spec_rewrite(_plane(model, rewrite=False), state, 0, {})
        is None
    )
    # A long request is already a spec by size; a structured one by shape.
    long_state = {**state, "prompt": LOOSE + " x" * 1000}
    assert await ControlPlane._project_spec_rewrite(_plane(model), long_state, 0, {}) is None
    structured = {**state, "prompt": "Build out this project from scratch: X.\nSTACK: FastAPI\nFILES: app/main.py"}
    assert await ControlPlane._project_spec_rewrite(_plane(model), structured, 0, {}) is None
    # Not an application request at all.
    question = {**state, "prompt": "What does app/main.py do?"}
    assert await ControlPlane._project_spec_rewrite(_plane(model), question, 0, {}) is None
    assert model.calls == 0

    # A provider without the method (deterministic, scripted fakes): no stage.
    bare = _plane(SimpleNamespace())
    assert await ControlPlane._project_spec_rewrite(bare, state, 0, {}) is None

    # A failed call loses only the sharpening.
    class Failing:
        async def project_spec(self, request, *, model_aliases=None):
            raise RuntimeError("provider down")

    assert await ControlPlane._project_spec_rewrite(_plane(Failing()), state, 0, {}) is None


@pytest.mark.asyncio
async def test_the_step_request_works_from_the_spec_but_keeps_the_intent() -> None:
    plane = _plane(Rewriter())
    plane.settings.project_reference_enabled = True
    state = {
        "prompt": LOOSE,
        "run_id": "r",
        "conversation_id": "c",
        "model_aliases": {},
    }
    request = ControlPlane._project_step_request(
        plane, state, {}, [], {}, 0, ["app/main.py"], spec_text="STACK: FastAPI\nFILES: app/main.py"
    )
    assert request["user_request"].startswith("STACK")
    assert request["original_request"] == LOOSE
    # build_turn detection keys off the user's own words, not the spec.
    assert request["build_turn"] is True

    bare = ControlPlane._project_step_request(plane, state, {}, [], {}, 0, ["app/main.py"])
    assert bare["user_request"] == LOOSE
    assert "original_request" not in bare
