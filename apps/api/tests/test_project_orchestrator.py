"""The orchestrator/coder split: one model decides scope, another writes.

The case these exist for is on record. Given a correctly planned four-file
conversion, narrowed by the host to ONE file, a capable coder read twenty-one
times and wrote nothing — it was never short of context, it was simply free to
keep looking. Everything here is about removing that freedom without removing
the model's judgement: the orchestrator judges and cannot write, the coder writes
and cannot judge, and the controller between them is ordinary code.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from waqil_api.contracts import (
    ProjectAgentStepV1,
    ProjectDirectionV1,
    ProjectToolCallV1,
)
from waqil_api.control_plane import (
    _FOCUSED_READ_ALLOWANCE,
    ControlPlane,
    _directed_attention,
)


async def _noop(*args: object, **kwargs: object) -> None:
    return None


async def _empty_context(project_id: str) -> dict[str, object]:
    return {"manifest": {"file_tree": []}}


def _plane(model: object, **overrides: Any) -> ControlPlane:
    plane = object.__new__(ControlPlane)
    plane.model = model
    plane.events = SimpleNamespace(emit=_noop)
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=None)
    settings: dict[str, Any] = {
        "project_agent_max_steps": 48,
        "project_staged_max_files": 48,
        "project_spec_rewrite": False,
        "project_spec_rewrite_max_chars": 1800,
        "project_reference_enabled": False,
        "project_reference_dir": Path("/none"),
        "project_reference_max_chars": 0,
        "project_reference_max_chars_local": 0,
        "project_repo_map_enabled": False,
        "project_orchestrator_enabled": True,
        "project_orchestrator_max_attempts": 3,
    }
    settings.update(overrides)
    plane.settings = SimpleNamespace(**settings)
    plane._guard = _noop
    plane._stage = _noop
    return plane


def _state(**extra: Any) -> dict[str, Any]:
    return {
        "prompt": "Convert this app to FastAPI: app/main.py and app/static/index.html.",
        "run_id": "run_x",
        "conversation_id": "conv_x",
        "model_aliases": {"_project_id": "asset_x"},
        "project_planned_files": ["app/main.py", "app/static/index.html"],
        "project_build_intent": "build",
        **extra,
    }


# ── The controller's rules ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_orchestrator_names_the_file_and_the_coder_is_told_what_to_write() -> None:
    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            assert "planned_files" in request
            return ProjectDirectionV1(
                path="app/main.py",
                instruction="Define create_app() returning FastAPI and mount appkit at /static.",
                reuse=["appkit.web.mount_appkit_static"],
            )

    plane = _plane(Model())
    direction = await ControlPlane._project_direct(
        plane, _state(), {}, {}, ["app/main.py", "app/static/index.html"], 3
    )
    assert direction["path"] == "app/main.py"
    assert "create_app()" in direction["instruction"]
    assert direction["reuse"] == ["appkit.web.mount_appkit_static"]
    assert direction["attempts"] == {"app/main.py": 1}

    brief = _directed_attention(direction)
    assert "Write ONLY app/main.py" in brief
    assert "create_app()" in brief          # the orchestrator's words, whole
    assert "appkit.web.mount_appkit_static" in brief
    assert "Reads are closed" in brief


@pytest.mark.asyncio
async def test_a_path_outside_the_plan_is_refused_not_obeyed() -> None:
    """Scope belongs to the plan. Directing outside it is not a decision the
    controller may follow, however confidently it is stated."""
    class Wanderer:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(path="/etc/passwd", instruction="do it")

    direction = await ControlPlane._project_direct(
        _plane(Wanderer()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert direction["path"] == "app/main.py"


@pytest.mark.asyncio
async def test_naming_the_same_path_again_is_a_repair_until_the_cap() -> None:
    class Repairer:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(path="app/main.py", instruction="fix the import")

    plane = _plane(Repairer())
    attempts: dict[str, int] = {}
    for expected in (1, 2, 3):
        direction = await ControlPlane._project_direct(
            plane,
            _state(project_direction_attempts=attempts),
            {}, {}, ["app/main.py"], 3,
        )
        assert direction["path"] == "app/main.py"
        assert direction["attempts"]["app/main.py"] == expected
        attempts = direction["attempts"]
    # Past the cap the host stops offering it, and the turn returns to the
    # ordinary loop rather than repairing one file forever.
    exhausted = await ControlPlane._project_direct(
        _plane(Repairer()),
        _state(project_direction_attempts=attempts),
        {}, {}, ["app/main.py"], 3,
    )
    assert exhausted == {}


@pytest.mark.asyncio
async def test_done_returns_the_turn_to_the_ordinary_loop_and_never_ends_it() -> None:
    """An orchestrator that could end a turn would be deciding scope AND
    completion — the two things this split separates."""
    class Finisher:
        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(done=True, reason="the plan is satisfied")

    direction = await ControlPlane._project_direct(
        _plane(Finisher()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert direction == {"done": True, "reason": "the plan is satisfied"}
    assert "path" not in direction


@pytest.mark.asyncio
async def test_a_staged_file_is_not_directed_again() -> None:
    class Model:
        async def project_direction(self, request, *, model_aliases=None):
            assert request["available_files"] == ["app/static/index.html"]
            return ProjectDirectionV1(path="app/static/index.html", instruction="write it")

    staged = {"app/main.py": {"content": "x", "origin": "create", "bytes": 1}}
    direction = await ControlPlane._project_direct(
        _plane(Model()), _state(), {}, staged,
        ["app/main.py", "app/static/index.html"], 3,
    )
    assert direction["path"] == "app/static/index.html"


@pytest.mark.asyncio
async def test_a_lost_direction_falls_back_to_the_undirected_loop() -> None:
    """The split is an improvement, never a dependency: a lane that cannot
    answer must degrade to the loop that worked before it existed."""
    class Broken:
        async def project_direction(self, request, *, model_aliases=None):
            raise RuntimeError("the planner lane is down")

    direction = await ControlPlane._project_direct(
        _plane(Broken()), _state(), {}, {}, ["app/main.py"], 3
    )
    assert direction == {}


@pytest.mark.asyncio
async def test_the_split_can_be_switched_off() -> None:
    class MustNotRun:
        async def project_direction(self, request, *, model_aliases=None):
            raise AssertionError("the orchestrator must not be consulted")

    direction = await ControlPlane._project_direct(
        _plane(MustNotRun(), project_orchestrator_enabled=False),
        _state(), {}, {}, ["app/main.py"], 3,
    )
    assert direction == {}


# ── The coder's half of the gate ───────────────────────────────────────────


def test_a_directed_request_narrows_to_one_file_and_says_reads_are_closed() -> None:
    plane = _plane(object())
    request = ControlPlane._project_step_request(
        plane,
        _state(
            project_direction={
                "path": "app/main.py",
                "instruction": "Define create_app().",
                "reuse": ["appkit.web.page"],
                "read": [],
            },
            project_focus_path="app/main.py",
        ),
        {"manifest": {"file_tree": []}},
        [], {}, 3,
        ["app/main.py", "app/static/index.html"],
    )
    # One file offered, which is what pins the write target on every lane.
    assert request["files_still_to_write"] == ["app/main.py"]
    assert request["reads_closed"] is True
    assert request["reuse_existing"] == ["appkit.web.page"]
    assert "Define create_app()." in request["attention"]


@pytest.mark.asyncio
async def test_a_directed_coder_may_not_read_at_all() -> None:
    """Undirected, a narrowed turn gets a small read allowance. Directed, it
    gets none: the orchestrator named what was needed and the host fetched it,
    so a read here is the exact drift the split removes."""
    plane = _plane(object())

    async def must_not_read(project_id, call, staged, extra):
        raise AssertionError("a directed coder's read must never reach the workspace")

    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=must_not_read)
    state = _state(
        project_direction={"path": "app/main.py", "instruction": "write it", "reuse": [], "read": []},
        project_focus_path="app/main.py",
        project_consecutive_reads=0,
        project_trace=[],
        project_pending_call={"name": "read_file", "arguments": {"path": "app/other.py"}},
    )
    evidence = await ControlPlane._project_execute(plane, state)  # type: ignore[arg-type]
    trace = evidence["project_trace"]
    assert trace[-1]["result"]["ok"] is False
    assert "reads are closed" in trace[-1]["result"]["error"]
    # The undirected allowance is genuinely different, so this is a real gate
    # rather than the old one renamed.
    assert _FOCUSED_READ_ALLOWANCE > 0


@pytest.mark.asyncio
async def test_the_host_reads_for_the_coder_before_closing_reads() -> None:
    """Closing reads is only fair because the host pays for them first."""
    seen: list[str] = []

    async def execute_staged(project_id, call, staged, extra):
        seen.append(call.arguments["path"])
        return {"content": f"contents of {call.arguments['path']}"}, None

    plane = _plane(object())
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    trace = await ControlPlane._prefetch_for_coder(
        plane,
        _state(),  # type: ignore[arg-type]
        "asset_x",
        {},
        {"path": "app/main.py", "read": ["app/models.py"]},
        [],
    )
    # The target file first — a repair is impossible without it — then what the
    # orchestrator asked for.
    assert seen == ["app/main.py", "app/models.py"]
    assert [entry["arguments"]["path"] for entry in trace] == seen
    assert all(entry["result"]["ok"] for entry in trace)


@pytest.mark.asyncio
async def test_a_prefetch_that_fails_is_evidence_not_an_error() -> None:
    async def execute_staged(project_id, call, staged, extra):
        raise FileNotFoundError("app/ghost.py does not exist")

    plane = _plane(object())
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    trace = await ControlPlane._prefetch_for_coder(
        plane, _state(), "asset_x", {},  # type: ignore[arg-type]
        {"path": "app/ghost.py", "read": []},
        [],
    )
    assert trace[-1]["result"]["ok"] is False
    assert "does not exist" in trace[-1]["result"]["error"]


# ── End to end, with a scripted orchestrator and coder ─────────────────────


@pytest.mark.asyncio
async def test_the_directed_arc_writes_the_file_the_orchestrator_named() -> None:
    class Model:
        def __init__(self) -> None:
            self.brief: str = ""

        async def project_plan_files(self, request, *, model_aliases=None):
            raise AssertionError("the plan is already taken in this state")

        async def project_direction(self, request, *, model_aliases=None):
            return ProjectDirectionV1(
                path="app/main.py",
                instruction="Define create_app() and mount the static files.",
                reuse=["appkit.web.mount_appkit_static"],
                read=["appkit/web.py"],
            )

        async def project_step(self, request, *, model_aliases=None):
            self.brief = request["attention"]
            return ProjectAgentStepV1(
                status="tool",
                tool_call=ProjectToolCallV1(
                    name="create_file",
                    arguments={"path": "app/main.py", "content": "def create_app():\n    ...\n"},
                ),
            )

    async def execute_staged(project_id, call, staged, extra):
        return {"content": "def mount_appkit_static(app): ...\n"}, None

    model = Model()
    plane = _plane(model)
    plane.projects = SimpleNamespace(context=_empty_context, execute_staged=execute_staged)
    result = await ControlPlane._project_step(
        plane,
        _state(project_plan_taken=True, project_iterations=3),  # type: ignore[arg-type]
    )
    # The coder was handed the orchestrator's instruction, not a generic one.
    assert "Define create_app() and mount the static files." in model.brief
    assert "appkit.web.mount_appkit_static" in model.brief
    # And the controller recorded who was directed, so a repair is countable.
    assert result["project_direction"]["path"] == "app/main.py"
    assert result["project_direction_attempts"] == {"app/main.py": 1}
    assert result["project_focus_path"] == "app/main.py"


def test_the_orchestrator_runs_in_its_own_seat_not_the_run_s_lane() -> None:
    """`_provider` is global to a run, so without this the planner ladder was
    decorative and the orchestrator ran on whatever chat was pinned to."""
    from waqil_api.control_plane import _planner_aliases

    aliases = {
        "_provider": "local",
        "coder": "qwen3-coder:30b",
        "_chain_planner": '[{"provider": "cline", "model": "anthropic/claude-opus-4.5"}]',
    }
    routed = _planner_aliases(aliases)
    assert routed["_provider"] == "cline"
    assert routed["_cline_model"] == "anthropic/claude-opus-4.5"
    # The coder alias is untouched: the two seats are independent.
    assert routed["coder"] == "qwen3-coder:30b"


def test_no_planner_chain_leaves_the_run_s_lane_alone() -> None:
    from waqil_api.control_plane import _planner_aliases

    aliases = {"_provider": "cohere", "coder": "x"}
    assert _planner_aliases(aliases) == {"_provider": "cohere", "coder": "x"}
