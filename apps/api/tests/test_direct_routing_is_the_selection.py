"""The project selection routes the turn. Wording never does.

A frozen `cline_direct` run used to be re-classified at the door: `_project_step`
asked a build-request regex whether the request "was a build", and a real task —
"Add invoice-status filtering to the existing application..." — was read as
conversation and dropped into the legacy per-file loop. The run had already been
frozen to the direct path at submit time; the regex overrode it.

The selection is now the decision. These pin that, and pin the four ways it
could regress: a question must still reach Cline, an unselected conversation must
stay ordinary chat, a checkpoint frozen to `planner_slices` must stay legacy, and
a turn that has already answered must not open a second round.
"""

from __future__ import annotations

from typing import Any

import pytest

from waqil_api.control_plane import ControlPlane


# The exact wording that was misrouted. Kept verbatim: the point of this file is
# that no phrasing decides anything, and a paraphrase would weaken the proof.
BASELINE_REQUEST = (
    "Add invoice-status filtering to the existing application. Add a backend "
    "endpoint that returns invoice counts by status, display those counts and a "
    "status filter in the static frontend, and add regression tests. Preserve "
    "the existing API and stored records."
)

PROJECT_QUESTION = "what does app/store.py do?"

# Every wording classifier in the module. A direct turn must call none of them:
# they exist for non-project chat, tool/build refusal contexts, and the frozen
# legacy loop, and each one is a route a direct run could be pulled off.
CLASSIFIERS = (
    "is_project_build_request",
    "is_project_build_instruction",
    "is_new_application_request",
    "wants_web_ui",
)

# The machinery the direct path replaced.
RETIRED = (
    "validate_effective_plan",
    "normalize_build_plan",
    "synthesize_single_slice",
    "next_vertical_slice",
    "initial_coding_prompt",
    "repair_coding_prompt",
)


@pytest.fixture
def tripwires(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every classifier and retired entry point fail loudly if reached."""

    called: list[str] = []

    for name in CLASSIFIERS + RETIRED:

        def _refuse(*_args: Any, __name: str = name, **_kwargs: Any) -> Any:
            called.append(__name)
            raise AssertionError(f"a direct turn reached {__name}")

        monkeypatch.setattr(f"waqil_api.control_plane.{name}", _refuse, raising=True)
    return called


class _Recorder:
    def __init__(self) -> None:
        self.types: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    async def emit(
        self, run_id: str, thread_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        del run_id, thread_id
        self.types.append(event_type)
        self.payloads.append(payload)


def _state(prompt: str, **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": "run_1",
        "conversation_id": "conv_1",
        "prompt": prompt,
        "model_aliases": {
            "_project_id": "asset_1",
            "_project_mode": "grok_bootstrap_local",
            "_coding_engine": "clinecore",
            "_build_path": "cline_direct",
        },
        "project_context": {"manifest": {"file_tree": ["app/main.py", "app/store.py"]}},
    }
    state.update(overrides)
    return state


class _Plane:
    """The two collaborators `_project_step` needs, and nothing else."""

    def __init__(self, tmp_path: Any) -> None:
        self.events = _Recorder()
        self.rounds = 0
        self.settings = type(
            "_S",
            (),
            {
                "cline_sidecar_max_iterations": 60,
                "cline_sidecar_max_rounds": 4,
                "project_run_check_budget": 12,
            },
        )()
        self.projects = type(
            "_P",
            (),
            {
                "assets": type(
                    "_A", (), {"project_path": staticmethod(_project_path(tmp_path))}
                )(),
                "context": staticmethod(_context),
            },
        )()

    # The real predicate, not a stub: routing is what these tests are about.
    _uses_cline_direct = staticmethod(ControlPlane._uses_cline_direct)
    _uses_clinecore = staticmethod(ControlPlane._uses_clinecore)
    _project_direct_admission = ControlPlane._project_direct_admission

    async def _guard(self, state: Any) -> None:
        del state

    async def _project_clinecore_round(self, state: Any) -> dict[str, Any]:
        self.rounds += 1
        await self.events.emit(
            state["run_id"], state["conversation_id"], "project.coding_started", {}
        )
        return {"response_text": "done", "project_pending_call": {}}

    async def _project_protected_setting(self, project_id: str) -> list[str]:
        del project_id
        return []


def _project_path(tmp_path: Any):
    async def _path(project_id: str) -> Any:
        del project_id
        return tmp_path

    return _path


async def _context(project_id: str) -> dict[str, Any]:
    del project_id
    return {"manifest": {"file_tree": ["app/main.py", "app/store.py"]}}


async def _drive(plane: _Plane, state: dict[str, Any]) -> dict[str, Any]:
    """Admission then round, exactly as the graph re-enters the node."""

    first = await ControlPlane._project_step(plane, state)  # type: ignore[arg-type]
    if "project_contract" not in first:
        return first
    return await ControlPlane._project_step(  # type: ignore[arg-type]
        plane, {**state, **first}
    )


@pytest.mark.asyncio
async def test_the_misrouted_baseline_wording_now_reaches_cline(
    tripwires: list[str], tmp_path: Any
) -> None:
    """1. The exact task text: direct_contract, then coding_started, no legacy."""

    plane = _Plane(tmp_path)
    await _drive(plane, _state(BASELINE_REQUEST))

    assert plane.events.types == ["project.direct_contract", "project.coding_started"]
    assert plane.rounds == 1
    assert tripwires == []


@pytest.mark.asyncio
async def test_a_project_question_also_goes_to_cline_and_needs_no_approval(
    tripwires: list[str], tmp_path: Any
) -> None:
    """2. A question is Cline's to answer, and an empty changeset needs no card."""

    plane = _Plane(tmp_path)
    result = await _drive(plane, _state(PROJECT_QUESTION))

    assert plane.events.types == ["project.direct_contract", "project.coding_started"]
    assert tripwires == []

    # Nothing staged: the router publishes rather than raising an approval card.
    terminal = {**_state(PROJECT_QUESTION), **result, "project_staged": {}}
    assert ControlPlane._route_after_project_step(plane, terminal) == "publish"  # type: ignore[arg-type]

    # The same turn with model-written bytes is the one that must be approved.
    with_writes = {
        **terminal,
        "project_staged": {"app/main.py": {"content": "x", "source": "model"}},
    }
    assert (
        ControlPlane._route_after_project_step(plane, with_writes)  # type: ignore[arg-type]
        == "build_approval"
    )


def test_a_conversation_with_no_project_is_ordinary_chat() -> None:
    """3. No selection, no project routing — the decision is the selection."""

    plane = type("_P", (), {"projects": object()})()
    assert (
        ControlPlane._route_after_retrieve(plane, {"model_aliases": {}})  # type: ignore[arg-type]
        == "plan"
    )
    assert (
        ControlPlane._route_after_retrieve(  # type: ignore[arg-type]
            plane, {"model_aliases": {"_project_id": "asset_1"}}
        )
        == "project"
    )


def test_a_checkpoint_frozen_to_planner_slices_stays_legacy() -> None:
    """4. The frozen alias still decides, in both directions."""

    frozen_legacy = {
        "model_aliases": {
            "_project_id": "asset_1",
            "_coding_engine": "clinecore",
            "_build_path": "planner_slices",
        }
    }
    assert ControlPlane._uses_cline_direct(frozen_legacy) is False  # type: ignore[arg-type]
    # Absent means legacy, never the new path by default.
    assert ControlPlane._uses_cline_direct({"model_aliases": {}}) is False  # type: ignore[arg-type]
    assert (
        ControlPlane._uses_cline_direct(  # type: ignore[arg-type]
            {"model_aliases": {"_build_path": "cline_direct"}}
        )
        is True
    )


@pytest.mark.asyncio
async def test_follow_up_wording_cannot_drop_a_direct_project_into_legacy(
    tripwires: list[str], tmp_path: Any
) -> None:
    """5. Every follow-up shape stays on the direct path, whatever it says."""

    follow_ups = (
        "thanks, now also show the totals",  # no verb the regex knows
        "why did you put the counts endpoint there?",  # a question
        "revert that",  # two words
        "ok",  # not a sentence
        "Preserve the existing API and stored records.",  # pure constraint
    )
    for prompt in follow_ups:
        plane = _Plane(tmp_path)
        await _drive(plane, _state(prompt))
        assert plane.rounds == 1, f"{prompt!r} did not reach Cline"
        assert plane.events.types == [
            "project.direct_contract",
            "project.coding_started",
        ], f"{prompt!r} took a different route"
    assert tripwires == []


@pytest.mark.asyncio
async def test_a_terminal_direct_turn_does_not_open_a_second_round(
    tripwires: list[str], tmp_path: Any
) -> None:
    """6. An answered turn re-entering the node must not start another session."""

    plane = _Plane(tmp_path)
    settled = _state(
        BASELINE_REQUEST,
        project_contract={"writable_roots": ["."], "protected_files": []},
        response_text="I added the counts endpoint.",
        project_pending_call={},
    )

    result = await ControlPlane._project_step(plane, settled)  # type: ignore[arg-type]

    assert result == {}
    assert plane.rounds == 0
    assert plane.events.types == []
    assert tripwires == []
